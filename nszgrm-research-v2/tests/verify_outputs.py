#!/usr/bin/env python3
"""Validate FP32 training outputs and (for formal runs) quantized acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED_WRITING={'zh_native','ja_kanji_proxy','foreign_phonetic_proxy'}
EXPECTED_LANGUAGE={'zh','en','fr','de','es','ru','ja','ko'}
EXPECTED_MODELS={'linear','complement_nb','cnn','bigru'}
FP32_CAP=200_000_000
QUANT_CAP=15_000_000


def valid_quant_tensor(value):
    if not isinstance(value,dict):return False
    if value.get('quant_format') not in ('sym_int8_per_row_v1','sym_int16_per_row_v1'):return False
    shape=value.get('shape',[])
    if not shape or not isinstance(shape,list):return False
    try:
        import base64, math
        raw=base64.b64decode(value['data_b64'],validate=True)
        is16=value['quant_format']=='sym_int16_per_row_v1'
        if len(raw)!=math.prod(shape)*(2 if is16 else 1):return False
        return len(value.get('scales',[]))==shape[0]
    except Exception:return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--allow-demo',action='store_true')
    ap.add_argument('--require-language',action='store_true')
    ap.add_argument('--skip-quantized',action='store_true',help='Only for synthetic smoke training')
    args=ap.parse_args();out=args.output
    required=['models_fp32.json','REPORT.json','REPORT.md','train_manifest.json']
    if not args.skip_quantized:required+=['models_int8.json','QUANT_AUDIT.json','LOSS_REPORT.json']
    for name in required:
        if not (out/name).is_file() or (out/name).stat().st_size==0:
            raise SystemExit(f'missing or empty output: {name}')
    if list(out.rglob('*.html')):raise SystemExit('training output must not contain HTML')
    fp_path=out/'models_fp32.json'
    if fp_path.stat().st_size>FP32_CAP:raise SystemExit('FP32 model exceeds 200 MB')
    fp=json.loads(fp_path.read_text('utf8'))
    report=json.loads((out/'REPORT.json').read_text('utf8'))
    if fp.get('format_version')!=3 or fp.get('trainer_version')!='3.3.1':
        raise SystemExit('unexpected FP32 model format/trainer version')
    if not args.allow_demo and report.get('training_mode')!='PUBLIC_WEAK_SUPERVISION':
        raise SystemExit('formal run must be PUBLIC_WEAK_SUPERVISION')
    tasks=fp.get('models',{})
    if 'writing' not in tasks:raise SystemExit('missing writing model')
    for task_name,expected in (('writing',EXPECTED_WRITING),('language',EXPECTED_LANGUAGE)):
        if task_name not in tasks:
            if task_name=='language' and not args.require_language:continue
            raise SystemExit(f'missing task model: {task_name}')
        task=tasks[task_name]
        if set(task.get('classes',[]))!=expected:raise SystemExit(f'{task_name} classes mismatch')
        models=task.get('models',{})
        if set(models)!=EXPECTED_MODELS:raise SystemExit(f'{task_name} model set mismatch')
        for name,model in models.items():
            if model.get('classes')!=task['classes'] or float(model.get('temperature',0))<=0:
                raise SystemExit(f'{task_name}/{name} metadata invalid')
        weights=task.get('fusion_weights',{})
        if set(weights)!=EXPECTED_MODELS or any(not 0<=float(v)<=1 for v in weights.values()) or abs(sum(weights.values())-1)>1e-6:
            raise SystemExit(f'{task_name} fusion weights invalid')
        metrics=report.get('task_reports',{}).get(task_name,{}).get('weak_test_metrics',{})
        if set(metrics)!=EXPECTED_MODELS|{'fusion'}:raise SystemExit(f'{task_name} metrics incomplete')
    if args.require_language and set(tasks)!={'writing','language'}:
        raise SystemExit('required FP32 bundle is not complete 3+8 classes')
    print(f'[PASS] FP32 model: {fp_path.stat().st_size:,}/{FP32_CAP:,} bytes')
    if not args.skip_quantized:
        q_path=out/'models_int8.json'
        if q_path.stat().st_size>=QUANT_CAP:raise SystemExit('quantized model must be strictly below 15 MB')
        quant=json.loads(q_path.read_text('utf8'))
        loss=json.loads((out/'LOSS_REPORT.json').read_text('utf8'))
        audit=json.loads((out/'QUANT_AUDIT.json').read_text('utf8'))
        if quant.get('format_version')!=4 or set(quant.get('models',{}))!=set(tasks):
            raise SystemExit('quantized bundle format/tasks mismatch')
        if not loss.get('passed'):raise SystemExit('quantization accuracy/Macro-F1 loss gate failed')
        if audit.get('acceptance_status')!='PASSED_ON_TRAINING_HELDOUT_WEAK_LABEL_SAMPLE':
            raise SystemExit('quantization audit is not marked accepted')
        for task_name,task in quant['models'].items():
            if task.get('classes')!=tasks[task_name]['classes'] or set(task.get('models',{}))!=EXPECTED_MODELS:
                raise SystemExit(f'{task_name} quantized classes/models mismatch')
            for model_name,model in task['models'].items():
                if model.get('classes')!=task['classes'] or float(model.get('temperature',0))<=0:
                    raise SystemExit(f'{task_name}/{model_name} quantized metadata invalid')
                if model_name=='linear' and not valid_quant_tensor(model.get('coef')):
                    raise SystemExit(f'{task_name}/linear coefficient quantization invalid')
                if model_name=='complement_nb' and not valid_quant_tensor(model.get('feature_log_prob')):
                    raise SystemExit(f'{task_name}/complement_nb quantization invalid')
                if model_name in ('cnn','bigru'):
                    state=model.get('state',{})
                    if not state:raise SystemExit(f'{task_name}/{model_name} state missing')
                    for key,value in state.items():
                        if 'quant_format' in value and not valid_quant_tensor(value):
                            raise SystemExit(f'{task_name}/{model_name}/{key} quantized tensor invalid')
            qtask=loss.get('tasks',{}).get(task_name,{})
            if set(qtask.get('models',{}))!=EXPECTED_MODELS|{'fusion'}:
                raise SystemExit(f'{task_name} quantization gate metrics incomplete')
            if not all(item.get('passed') for item in qtask['models'].values()):
                raise SystemExit(f'{task_name} has a model above the 10% loss threshold')
        if args.require_language and set(quant['models'])!={'writing','language'}:
            raise SystemExit('quantized bundle is not complete 3+8 classes')
        qbytes=q_path.stat().st_size
        print(f'[PASS] quantized model: {qbytes:,}/{QUANT_CAP:,} bytes')
        print('[PASS] every model and fusion passed relative Accuracy/Macro-F1 loss <=10%')
    else:
        print('[SKIP] quantized checks skipped for synthetic smoke run')
    print('[PASS] no HTML generated')


if __name__=='__main__':main()
