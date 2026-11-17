#!/usr/bin/env python3
"""Multi-architecture, multi-task supervised training and independent evaluation.

The 200,000,000-byte budget applies to ALL ORIGINAL FP32 portable model files
across tasks. No quantization is applied or used to satisfy the budget.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter,defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, log_loss, confusion_matrix
from architectures import train_neural, probabilities
from corpus import acquire, build_writing, checks, language_records, write_review_queue
from strong_data import LANGS, load_gold

def encode_numpy(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(f'not JSON serializable: {type(x).__name__}')

SEED=20261117
MODEL_NAMES=('linear','complement_nb','cnn','bigru')
BYTE_CAP=200_000_000


def evaluate(y, proba, classes):
    guesses = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (guesses == y)
    # Expected Calibration Error: reported as an observational metric, not a
    # promise of population-level calibration on weakly labeled samples.
    ece = 0.0
    reliability = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = min(1.0, lower + 0.1)
        mask = (conf >= lower) & (conf < upper if upper < 1.0 else conf <= 1.0)
        if not mask.any():
            continue
        acc, avg = float(correct[mask].mean()), float(conf[mask].mean())
        ece += float(mask.mean()) * abs(acc - avg)
        reliability.append({'confidence_low': round(float(lower), 2),
                            'confidence_high': round(float(upper), 2),
                            'n': int(mask.sum()), 'accuracy': acc, 'mean_confidence': avg})
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, guesses)),
        'balanced_accuracy': float(balanced_accuracy_score(y, guesses)),
        'macro_f1': float(f1_score(y, guesses, average='macro', labels=list(range(len(classes))), zero_division=0)),
        'log_loss': float(log_loss(y, np.clip(proba, 1e-8, 1.0), labels=list(range(len(classes))))),
        'expected_calibration_error_10_bins': ece,
        'reliability_bins': reliability,
        'classification_report': classification_report(y, guesses, labels=list(range(len(classes))),
                          target_names=classes, zero_division=0, output_dict=True),
        'confusion_matrix': confusion_matrix(y, guesses, labels=list(range(len(classes)))).tolist(),
        'classes': classes,
    }


def temperature(p,t):
    a=np.log(np.maximum(p,1e-10))/max(float(t),1e-5)
    a-=np.max(a,axis=1,keepdims=True)
    e=np.exp(a)
    return e/np.sum(e,axis=1,keepdims=True)


def fit_temp(p,y):
    vals=[0.5,0.65,0.8,1.,1.25,1.5,2.,3.]
    return min(vals,key=lambda v:log_loss(y,temperature(p,v),labels=list(range(p.shape[1]))))


def choose_weights(calibrated,y):
    """Optimize log-loss with lower bound 0.03 for all four models.
    Held-out weak TEST is never used for blending, temperature or thresholds.
    """
    cube=np.stack(calibrated,axis=1)
    k=cube.shape[1]
    if k==1:return [1.]
    init=np.ones(k)/k
    def obj(w):
        p=np.einsum('nmc,m->nc',cube,w)
        return log_loss(y,p,labels=list(range(p.shape[1]))) + 0.025*np.sum((w-init)**2)
    result=minimize(obj,init,method='SLSQP',bounds=[(0.03,1)]*k,
                    constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1}],
                    options={'maxiter':90,'ftol':1e-9})
    if not result.success or np.any(result.x<0.029):return init.tolist()
    w=np.maximum(result.x,0.03);w=w/w.sum()
    return w.tolist()


def model_bundle_linear(train,val,test,classes,max_features,seed,gold_test=None):
    train_names=[r['name'] for r in train]
    y=np.array([classes.index(r['label']) for r in train],dtype=np.int64)
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,3),lowercase=False,
                               min_df=1 if len(train)<500 else 2,
                               max_features=max_features,dtype=np.float32,
                               norm='l2',smooth_idf=True,sublinear_tf=False)
    trainx=vectorizer.fit_transform(train_names)
    x_val=vectorizer.transform([r['name'] for r in val])
    x_test=vectorizer.transform([r['name'] for r in test])
    common={'type':'tfidf_char_1_3','vocabulary':vectorizer.vocabulary_,
            'idf':vectorizer.idf_.astype(np.float32).tolist(),
            'feature_count':len(vectorizer.vocabulary_)}
    linear=SGDClassifier(loss='log_loss',alpha=1e-5,max_iter=1500,tol=1e-4,
                         class_weight='balanced',random_state=seed,average=True)
    linear.fit(trainx,y)
    nb=ComplementNB(alpha=0.65,norm=False)
    nb.fit(trainx,y)
    export={
        'linear':{'arch':'linear','classes':classes,'tfidf':common,
                  'coef':linear.coef_.astype(np.float32).tolist(),
                  'intercept':linear.intercept_.astype(np.float32).tolist(),
                  'mode':'sigmoid_binary' if len(classes)==2 else 'sigmoid_ovr'},
        'complement_nb':{'arch':'complement_nb','classes':classes,'tfidf':common,
                         'feature_log_prob':nb.feature_log_prob_.astype(np.float32).tolist()}}
    x_gold=vectorizer.transform([r['name'] for r in gold_test]) if gold_test else None
    predictions={
        'linear':{'val':linear.predict_proba(x_val).astype(np.float64),
                  'test':linear.predict_proba(x_test).astype(np.float64),
                  'gold':linear.predict_proba(x_gold).astype(np.float64) if x_gold is not None else np.empty((0,len(classes)))},
        'complement_nb':{'val':nb.predict_proba(x_val).astype(np.float64),
                         'test':nb.predict_proba(x_test).astype(np.float64),
                         'gold':nb.predict_proba(x_gold).astype(np.float64) if x_gold is not None else np.empty((0,len(classes)))}}
    return export,predictions,{'vocabulary_count':len(vectorizer.vocabulary_),
                               'train_samples':len(train_names)}


def subset_names(rows, max_items, seed):
    if len(rows)<=max_items or max_items==0:return rows
    # Fixed subsampling stratified by class for inference-time CPU budget.
    by=defaultdict(list)
    for row in rows:by[row['label']].append(row)
    rng=random.Random(seed)
    samples=[]
    quota=max(1,max_items//len(by))
    for label,values in sorted(by.items()):
        samples.extend(rng.sample(values,min(quota,len(values))))
    return samples


def train_task(task,rows,out,args):
    print(f'\n[task] {task}',flush=True)
    counts=checks(rows,task)
    classes=sorted(set(r['label'] for r in rows))
    if task == 'language' and set(classes) != set(LANGS):
        raise RuntimeError(f'language task requires all eight classes {sorted(LANGS)}; got {classes}')
    parts={name:[r for r in rows if r['split']==name and r.get('tier')!='gold']
           for name in ('train','val','test')}
    # Gold TRAIN and VAL may be used with evidenced labels; GOLD test NEVER in training
    for name in ('train','val'):
        parts[name]+=[r for r in rows if r['split']==name and r.get('tier')=='gold']
    for name in parts:
        if any(Counter(r['label'] for r in parts[name])[c]<2 for c in classes):
            raise RuntimeError(f'{task}: class missing from {name}; counts={counts}')
    train,val,test=(parts[k] for k in ('train','val','test'))
    rng=random.Random(args.seed+177)
    # Independent val halves: temperatures fitted on A, blending fitted on B.
    by_label=defaultdict(list)
    for r in val:by_label[r['label']].append(r)
    cal=[];blend=[]
    for label,group in sorted(by_label.items()):
        rng.shuffle(group)
        mid=max(1,len(group)//2)
        cal.extend(group[:mid]);blend.extend(group[mid:])
    val_all=cal+blend
    index={id(r):i for i,r in enumerate(val_all)}
    cal_indices=np.array([index[id(r)] for r in cal],dtype=np.int64)
    blend_indices=np.array([index[id(r)] for r in blend],dtype=np.int64)
    cal_y=np.array([classes.index(r['label']) for r in cal]);blend_y=np.array([classes.index(r['label']) for r in blend])
    test_y=np.array([classes.index(r['label']) for r in test])
    # Gold test is never fit or calibrated; prediction is held out from training.
    gold_test=[r for r in rows if r.get('tier')=='gold' and r['split']=='test' and r['label'] in classes]
    simple,probs,info=model_bundle_linear(train,val_all,test,classes,args.max_features,args.seed,gold_test)
    exports=dict(simple)
    metrics={}; neural_info={}
    for arch in ('cnn','bigru'):
        net,vocab,exp,detail=train_neural(arch,train,classes,val_all,args.seed+({'cnn':7,'bigru':13}[arch]),
                 per_class=args.neural_per_class,epochs=args.epochs,batch_size=args.batch_size,
                 vocab_limit=args.vocab_limit,emb_dim=args.embedding_dim,cnn_channels=args.cnn_channels,
                 gru_hidden=args.gru_hidden,smoke=args.smoke)
        exports[arch]=exp;neural_info[arch]=detail
        probs[arch]={'val':probabilities(net,[r['name'] for r in val_all],vocab),
                     'test':probabilities(net,[r['name'] for r in test],vocab),
                     'gold':probabilities(net,[r['name'] for r in gold_test],vocab)}
        del net
    temp={}
    for name in MODEL_NAMES:
        t=fit_temp(probs[name]['val'][cal_indices],cal_y)
        temp[name]=t
        exports[name]['temperature']=t
        metrics[name]={'test':evaluate(test_y,temperature(probs[name]['test'],t),classes),
                       'calibration_temperature':t}
        print(f'[eval] {task}/{name}: Accuracy={metrics[name]["test"]["accuracy"]:.4f} '
              f'Macro-F1={metrics[name]["test"]["macro_f1"]:.4f}',flush=True)
    calibrated=[temperature(probs[name]['val'][blend_indices],temp[name]) for name in MODEL_NAMES]
    w=choose_weights(calibrated,blend_y)
    ensemble=np.zeros((len(test),len(classes)),dtype=np.float64)
    for i,name in enumerate(MODEL_NAMES):ensemble+=w[i]*temperature(probs[name]['test'],temp[name])
    metrics['fusion']={'test':evaluate(test_y,ensemble,classes),'weights':dict(zip(MODEL_NAMES,w)),
                       'validation_blend_macro_f1':evaluate(blend_y,sum(w[i]*p for i,p in enumerate(calibrated)),classes)['macro_f1'],
                       'strategy':'nonnegative 4-model stacking, min weight 0.03, optimized on validation-B only'}
    print(f'[fusion] {task}: test Macro-F1={metrics["fusion"]["test"]["macro_f1"]:.4f}',flush=True)
    # NOT a gold test; weak test measures agreement with the source collection labels.
    # Independently evaluate externally sourced / human-reviewed data. This tier is
    # NEVER part of train/validation/weak test nor used to choose fusion weights.
    gold_stats={}
    if gold_test:
        gold_y=np.array([classes.index(r['label']) for r in gold_test],dtype=np.int64)
        gold_names=[r['name'] for r in gold_test]
        gold_vectorizer_cache=None
        # Model inference uses the exact trained sklearn estimators/transformer.
        # See model_bundle_linear's explicit estimator callback.
        for key in ('linear','complement_nb'):
            model=probs[key]['gold']
            gold_stats[key]=evaluate(gold_y,temperature(model,temp[key]),classes)
        for key in ('cnn','bigru'):
            model=probs[key]['gold']
            gold_stats[key]=evaluate(gold_y,temperature(model,temp[key]),classes)
        ensemble_gold=sum(w[i]*temperature(probs[key]['gold'],temp[key]) for i,key in enumerate(MODEL_NAMES))
        gold_stats['fusion']=evaluate(gold_y,ensemble_gold,classes)
    note='Independent human-reviewed gold test is NOT available' if not gold_test else f'{len(gold_test)} independently reviewed rows evaluated'
    # Write weak test errors (sampled) without leaking full upstream data.
    out.mkdir(parents=True,exist_ok=True)
    with (out/f'{task}_errors.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['name','expected_weak_label','predicted_label','confidence'])
        wrong=np.where(ensemble.argmax(1)!=test_y)[0]
        for i in wrong[:3000]:
            writer.writerow([test[int(i)]['name'],classes[int(test_y[i])],classes[int(ensemble[i].argmax())],round(float(ensemble[i].max()),5)])
    # Active review candidates: held-out samples where model decisions conflict
    # or final confidence is low; suggestions are NOT promoted to training labels.
    review=[]
    indiv={key:temperature(probs[key]['test'],temp[key]) for key in MODEL_NAMES}
    for i,row in enumerate(test):
        disagreement=len({int(v[i].argmax()) for v in indiv.values()})
        margin=float(np.sort(ensemble[i])[-1]-np.sort(ensemble[i])[-2])
        priority=3*disagreement+(1-margin)
        review.append((priority,i))
    review.sort(key=lambda pair:(-pair[0],test[pair[1]]['name']))
    with (out/f'{task}_disagreement_review_queue.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['name','writing_type','origin_lang','group_id','reference_url',
                         'reviewer','split','weak_suggested_label','predicted_label',
                         'prediction_confidence','distinct_model_predictions'])
        for _,i in review[:300]:
            row=test[i]
            writer.writerow([row['name'],'','','','','','',row['label'],
                             classes[int(ensemble[i].argmax())],round(float(ensemble[i].max()),4),
                             len({int(v[i].argmax()) for v in indiv.values()})])
    # Save a small stratified sample of this already-held-out weak test split for
    # the post-training quantization loss gate. It stays in .cache, not in release.
    eval_path=args.cache.parent/'quant_eval_samples.json'
    eval_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        eval_data=json.loads(eval_path.read_text(encoding='utf8')) if eval_path.is_file() else {'tasks':{}}
    except (OSError,ValueError,json.JSONDecodeError):
        eval_data={'tasks':{}}
    sample_cap=500 if task=='writing' else 400
    eval_rng=random.Random(args.seed+(1103 if task=='writing' else 2207))
    eval_rows=[]
    for label in classes:
        pool=sorted((r for r in test if r['label']==label),key=lambda r:r['name'])
        take=eval_rng.sample(pool,min(sample_cap,len(pool)))
        eval_rows.extend({'name':r['name'],'label':r['label']} for r in take)
    eval_data.setdefault('tasks',{})[task]=eval_rows
    eval_path.write_text(json.dumps(eval_data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(f'[quantization] saved held-out evaluation sample: {task} n={len(eval_rows)}',flush=True)
    return {'task':task,'classes':classes,'models':exports,'fusion_weights':dict(zip(MODEL_NAMES,w)),
            'score_semantics':('foreign_phonetic_proxy probability ×100, not a real-world probability' if task=='writing' else
                               'Wikidata P103 person-native-language proxy, NOT name etymology')},\
           {'task':task,'data_counts':counts,'weak_test_metrics':metrics,'neural':neural_info,
            'linear':info,'human_reviewed_test_rows':len(gold_test),'human_review_status':note,
            'human_review_metrics':gold_stats},gold_test



def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    gold=load_gold(args.gold)
    if args.smoke:
        # synthetic STRUCTURE test; never published as a trained real-world model.
        from demo_data import demo
        wr,lg=demo(args.seed)
        provenance={'mode':'SYNTHETIC_SMOKE','NOT_REAL_TRAINING':True}
        wmeta={'note':'synthetic'};lmeta={'note':'synthetic','status':'ready'}
    else:
        source,provenance=acquire(args.cache,args.source_dir)
        wr,wmeta=build_writing(source,gold,args.max_per_class,args.seed)
        write_review_queue(wr,args.output/'review_queue.csv',seed=args.seed)
        lg,lmeta=language_records(args.cache/'wikidata',gold,args.wikidata_limit,args.wikidata,args.seed,
                                  min_per_language=args.min_language)
    bundle={'format_version':3,'trainer_version':'3.0.0','generated_utc':datetime.now(timezone.utc).isoformat(),
            'seed':args.seed,'precision':'original-FP32; NO parameter quantization',
            'models':{},'provenance':{'writing':'corpus-membership WEAK supervision',
                                   'language':'8-class Wikidata P103 person native-language PROXY: zh/en/fr/de/es/ru/ja/ko'}}
    report={'trainer_version':'3.0.0','generated_utc':bundle['generated_utc'],'training_mode':'DEMO_ONLY' if args.smoke else 'PUBLIC_WEAK_SUPERVISION',
            'hardware_note':'CPU compatible; parameters are from-scratch small architectures, not pretrained transformers',
            'data_sources':{'writing':wmeta,'language':lmeta,'upstream':provenance},
            'task_reports':{},'gold_test':'NO VERIFIED GOLD TEST; external generalization unproven',
            'language_classes_expected':sorted(LANGS)}
    with (args.output/'train_manifest.json').open('w',encoding='utf8') as f:
        json.dump(report['data_sources'],f,ensure_ascii=False,indent=2,default=encode_numpy)
    wr_report={}; gold_write=[]
    bundle['models']['writing'],wr_report,gold_write=train_task('writing',wr,args.output,args)
    report['task_reports']['writing']=wr_report
    if gold_write:
        report['gold_test']='Human-reviewed gold rows evaluated independently on writing task; see per-task metrics.'
    lang_ready=lmeta.get('status')=='ready' and len(lg)>70
    if args.wikidata == 'required' and not lang_ready:
        raise RuntimeError('The requested 8-class language model is incomplete. '
                           f'Status={lmeta.get("status")}; missing={lmeta.get("missing_languages", [])}; '
                           'rerun with Wikidata available or explicitly choose best_effort.')
    if lang_ready:
        try:
            bundle['models']['language'],l_report,gold_lang=train_task('language',lg,args.output,args)
            report['task_reports']['language']=l_report
        except (ValueError,RuntimeError) as e:
            if args.wikidata == 'required':
                raise
            print('[warn] language proxy training unusable:',str(e)[:500],flush=True)
            report['task_reports']['language']={'status':'unavailable','reason':str(e)}
    else:
        report['task_reports']['language']={'status':'unavailable',
                     'reason':'the complete eight-class proxy set is unavailable; partial classes are not trained or published',
                     'source_status':lmeta}
    # Fully original float32 weights; counted as ALL training-produced deployable models.
    model_path=args.output/'models_fp32.json'
    model_path.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=encode_numpy),encoding='utf8')
    bytes_models=model_path.stat().st_size
    component_sizes={task:{key:len(json.dumps(item,ensure_ascii=False,separators=(',',':'),default=encode_numpy).encode('utf8'))
                      for key,item in t['models'].items()}
                      for task,t in bundle['models'].items()}
    report['size_budget']={'model_file':'models_fp32.json','model_raw_bytes':bytes_models,
                           'max_raw_bytes':BYTE_CAP,'within_budget':bytes_models<=BYTE_CAP,
                           'model_component_raw_bytes':component_sizes,
                           'output_policy':'training package exports model parameters and research reports only; no HTML is generated'}
    if bytes_models > BYTE_CAP:
        raise RuntimeError(f'ORIGINAL models {bytes_models:,} bytes exceed {BYTE_CAP:,} bytes. '
                           'Reduce --max-features or --vocab-limit, not quantization.')
    # This training package deliberately does not generate or bundle an HTML front end.
    # Remove stale HTML from a reused output directory so old runs cannot be mistaken
    # for artifacts of the current training execution.
    for stale in args.output.rglob('*.html'):
        stale.unlink()
    stale_browser=args.output/'browser'
    if stale_browser.exists() and stale_browser.is_dir():
        import shutil
        shutil.rmtree(stale_browser)
    report['artifacts']={'model':str(model_path),'report_json':str(args.output/'REPORT.json'),
                         'report_markdown':str(args.output/'REPORT.md'),
                         'manifest':str(args.output/'train_manifest.json')}
    (args.output/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=encode_numpy),encoding='utf8')
    lines=['# 中文姓名多模型训练报告','',f'训练模式：{report["training_mode"]}',
           f'原精度模型总大小：{bytes_models:,} / {BYTE_CAP:,} 字节',
           '模型参数均为 FP32，未量化；所有分类器独立评分并经验证集融合。',
           '本训练工程只导出模型 JSON、数据清单、评估报告和人工复核队列，不生成 HTML。','']
    for task,tr in report['task_reports'].items():
        lines.append(f'## {task}')
        if 'weak_test_metrics' not in tr:
            lines.append('未完成：'+tr.get('reason','unknown'));continue
        lines.append('类别：'+', '.join(bundle['models'][task]['classes']))
        lines.append('**以下是语料弱标签测试结果，不是人工标注的可靠性验证。**')
        lines.append('| 模型 | Accuracy | Macro-F1 |')
        lines.append('|---|---:|---:|')
        for name,ms in tr['weak_test_metrics'].items():
            s=ms['test'];lines.append(f'| {name} | {s["accuracy"]:.4f} | {s["macro_f1"]:.4f} |')
        lines.append('融合权重：'+json.dumps(tr['weak_test_metrics']['fusion']['weights'],ensure_ascii=False))
        lines.append('FP32 模型体积（字节）：'+json.dumps(component_sizes[task],ensure_ascii=False))
        lines.append('人工测试：'+tr['human_review_status'])
        if tr['human_review_metrics']:
            lines.append(f'**独立人工金标准**：N={tr["human_reviewed_test_rows"]}；融合 Accuracy={tr["human_review_metrics"]["fusion"]["accuracy"]:.4f}，Macro-F1={tr["human_review_metrics"]["fusion"]["macro_f1"]:.4f}。')
        lines.append('')
    lines.append('## 可靠性状态')
    lines.append(('已计算部分人工金标准结果；规模和覆盖度仍须审核，不能直接宣称达到真实世界可靠性能。'
                  if any(r.get('human_review_metrics') for r in report['task_reports'].values())
                  else '尚未提供足量有审核者与可查证来源的人工金标准测试集，**不允许宣称已达到真实世界的可靠性能**。'))
    lines.append('请填写 data/gold_reviewed.csv。语料间来源偏差和同一人物的别名泄漏仍可能存在。')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f'[finished] ORIGINAL FP32 MODELS {bytes_models:,} / {BYTE_CAP:,} bytes',flush=True)
    print('[report] '+str(args.output/'REPORT.md'),flush=True)


def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('outputs'))
    ap.add_argument('--cache',type=Path,default=Path('.cache/data'))
    ap.add_argument('--source-dir',type=Path)
    ap.add_argument('--gold',type=Path,default=Path('data/gold_reviewed.csv'))
    ap.add_argument('--max-per-class',type=int,default=150000)
    ap.add_argument('--neural-per-class',type=int,default=45000)
    ap.add_argument('--max-features',type=int,default=60000)
    ap.add_argument('--vocab-limit',type=int,default=6000)
    ap.add_argument('--embedding-dim',type=int,default=128)
    ap.add_argument('--cnn-channels',type=int,default=96)
    ap.add_argument('--gru-hidden',type=int,default=128)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--wikidata',choices=['required','best_effort','off'],default='required')
    ap.add_argument('--wikidata-limit',type=int,default=3500)
    ap.add_argument('--min-language',type=int,default=80)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if any(x<0 for x in (args.max_per_class,args.neural_per_class)) or args.max_features<200 or args.vocab_limit<30 or args.epochs<1 or args.batch_size<1 or args.embedding_dim<8 or args.gru_hidden<8 or args.cnn_channels<8 or max(args.embedding_dim,args.gru_hidden,args.cnn_channels)>256:
        ap.error('invalid training limits')
    return args

if __name__=='__main__':run(parse())
