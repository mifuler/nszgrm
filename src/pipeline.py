#!/usr/bin/env python3
"""Reproducible weakly-supervised Chinese-name classifier training.

No raw downloaded source corpus is redistributed in the training package.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score,
    log_loss)

SOURCE_REPO = 'https://github.com/wainshine/Chinese-Names-Corpus.git'
SOURCE_WEB = 'https://github.com/wainshine/Chinese-Names-Corpus'
SOURCE_FILES = {
    'zh_native': 'Chinese_Names_Corpus/Chinese_Names_Corpus（120W）.txt',
    'foreign_phonetic_proxy': 'English_Names_Corpus/English_Cn_Name_Corpus（48W）.txt',
    'ja_kanji_proxy': 'Japanese_Names_Corpus/Japanese_Names_Corpus（18W）.txt',
}
LABEL_INFO = {
    'zh_native': '中文姓名语料来源，非绝对可靠的人物身份标签',
    'foreign_phonetic_proxy': '翻译名词典的弱标签，混有地名，不能当作全为英语人名',
    'ja_kanji_proxy': '日本姓名语料的汉字表记弱标签，不能等同汉字音译',
}
LANGS = {
    'en': ('Q1860', '英语'), 'ru': ('Q7737', '俄语'),
    'ja': ('Q5287', '日语'), 'fr': ('Q150', '法语'),
    'es': ('Q1321', '西班牙语'), 'de': ('Q188', '德语'),
    'ko': ('Q9176', '韩语'),
}
HAN_RE = re.compile(r'^[\u3400-\u9fff]{2,8}$')
MIN_PER_LANG = 30


def normalize(text: str) -> str | None:
    """Accept one name component; do not silently strip internal punctuation."""
    s = unicodedata.normalize('NFKC', text).strip().replace('\ufeff', '')
    # Some source files have name separated by a tab/comma from metadata.
    s = re.split(r'[\t,，|\s]', s, maxsplit=1)[0].strip()
    if not HAN_RE.fullmatch(s):
        return None
    return s


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_run(*args: str, cwd: str | None = None, timeout: int = 240):
    return subprocess.run(['git', *args], cwd=cwd, text=True, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def safe_extract(tar_path: Path, dest: Path):
    """Refuse links and traversal from an untrusted network archive."""
    with tarfile.open(tar_path, 'r:gz') as archive:
        for item in archive.getmembers():
            p = Path(item.name)
            if item.issym() or item.islnk() or p.is_absolute() or '..' in p.parts:
                raise ValueError('Invalid archive member: ' + item.name)
        archive.extractall(dest, filter='data')


def download_url(url: str, path: Path, timeout: int = 120):
    req = urllib.request.Request(url, headers={'User-Agent': 'ChineseNameResearchTrainer/1.0 (github.com/iqonli)'})
    tmp = path.with_suffix(path.suffix + '.partial')
    with urllib.request.urlopen(req, timeout=timeout) as res, tmp.open('wb') as file:
        shutil.copyfileobj(res, file, 1024 * 1024)
    os.replace(tmp, path)


def acquire_source(cache: Path) -> tuple[Path, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / 'Chinese-Names-Corpus'
    required = [source / f for f in SOURCE_FILES.values()]
    if not all(p.exists() and p.stat().st_size > 1000 for p in required):
        if source.exists():
            shutil.rmtree(source)
        try:
            print('[download] Git clone --depth 1', flush=True)
            git_run('clone', '--depth', '1', '--single-branch', SOURCE_REPO, str(source), timeout=360)
        except Exception as e:
            print('[download] Git clone failed:', repr(e), '\n[download] Try GitHub codeload tarball', flush=True)
            archive = cache / 'source.tar.gz'
            download_url('https://codeload.github.com/wainshine/Chinese-Names-Corpus/tar.gz/refs/heads/master', archive)
            safe_extract(archive, cache)
            extracted = cache / 'Chinese-Names-Corpus-master'
            if not extracted.is_dir():
                raise RuntimeError('codeload archive missing expected root directory')
            extracted.rename(source)
    if not all(p.exists() and p.stat().st_size > 1000 for p in required):
        raise RuntimeError('Source download incomplete: source corpora are missing')
    try:
        rev = git_run('-C', str(source), 'rev-parse', 'HEAD').stdout.strip()
    except Exception:
        rev = 'master tarball (Git SHA unavailable)'
    files = {k: {'relative_path': rel, 'size_bytes': (source / rel).stat().st_size,
                 'sha256': sha256(source / rel)} for k, rel in SOURCE_FILES.items()}
    return source, {'repo': SOURCE_WEB, 'revision': rev, 'files': files,
                    'license_repository': 'Apache-2.0 (upstream origin of some records uncertain)'}


def load_names(path: Path) -> tuple[set[str], dict]:
    names: set[str] = set()
    seen = 0
    invalid = 0
    # Official files are UTF-8 text; replace encoding errors and track invalid rows.
    with path.open('r', encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            seen += 1
            if not line.strip():
                continue
            s = normalize(line)
            if s is None:
                invalid += 1
            else:
                names.add(s)
    return names, {'read_lines': seen, 'invalid_or_filtered': invalid, 'unique_valid': len(names)}


def balanced_cap(names: set[str], maximum: int, seed: int) -> list[str]:
    ordered = sorted(names)
    if maximum > 0 and len(ordered) > maximum:
        ordered = random.Random(seed).sample(ordered, maximum)
    return ordered


def split_by_hash(rows: list[tuple[str, str]], seed: int):
    out = {'train': [], 'val': [], 'test': []}
    for label, name in rows:
        digest = hashlib.sha256(f'{seed}\0{name}'.encode('utf-8')).digest()
        num = int.from_bytes(digest[:8], 'big') / 2**64
        bucket = 'train' if num < .80 else ('val' if num < .90 else 'test')
        out[bucket].append((label, name))
    return out


def stable_class_split(rows: list[tuple[str, str]], seed: int):
    splits = split_by_hash(rows, seed)
    counts = {part: Counter(label for label, _ in entries) for part, entries in splits.items()}
    labels = sorted(set(label for label, _ in rows))
    bad = [lab for lab in labels if any(counts[p][lab] < 2 for p in ('train', 'val', 'test'))]
    if bad:
        raise ValueError(f'Insufficient train/val/test examples for labels {bad}: {counts}. Increase sample sizes.')
    return splits


def wdqs_query(qid: str, limit: int) -> str:
    # Only use explicitly tagged native-language-of-person (P103) as a PROXY, not verified etymology.
    return f'''SELECT DISTINCT ?person ?name WHERE {{
  ?person wdt:P103 wd:{qid} .
  ?person wdt:P31 wd:Q5 .
  ?person rdfs:label ?name .
  FILTER(LANG(?name) = "zh" || LANG(?name) = "zh-hans")
}} LIMIT {limit}'''


def fetch_wikidata(lang: str, qid: str, limit: int, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (lang + '.json')
    if path.exists() and path.stat().st_size > 100:
        payload = json.loads(path.read_text(encoding='utf-8'))
    else:
        q = wdqs_query(qid, limit)
        params = urllib.parse.urlencode({'query': q, 'format': 'json'})
        url = 'https://query.wikidata.org/sparql?' + params
        error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={
                    'Accept': 'application/sparql-results+json',
                    'User-Agent': 'ChineseNameResearchTrainer/1.0 (https://github.com/iqonli)'
                })
                with urllib.request.urlopen(req, timeout=110) as f:
                    payload = json.loads(f.read().decode('utf-8'))
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = str(exc)
                print(f'[wikidata] {lang} attempt {attempt + 1}: {exc}', flush=True)
                if attempt < 2:
                    time.sleep(5 + attempt * 10)
        else:
            raise RuntimeError(f'Wikidata query {lang} failed: {error}')
    result = []
    for row in payload.get('results', {}).get('bindings', []):
        name = normalize(row.get('name', {}).get('value', ''))
        person = row.get('person', {}).get('value', '').rsplit('/', 1)[-1]
        if name and re.fullmatch(r'Q[1-9][0-9]*', person):
            result.append((person, name))
    return result


def assemble_wikidata(cache: Path, limit: int, mode: str):
    status = {'status': 'disabled' if mode == 'off' else 'attempted',
              'language_rows': {}, 'failed': {}, 'label_semantics':
              'Wikidata P103 (native language of person), weak proxy; NOT origin-language ground truth'}
    if mode == 'off':
        return [], status
    raw: dict[str, list[tuple[str, str]]] = {}
    for lang, (qid, _) in LANGS.items():
        try:
            print('[wikidata] query', lang, qid, flush=True)
            raw[lang] = fetch_wikidata(lang, qid, limit, cache)
            status['language_rows'][lang] = len(raw[lang])
        except Exception as exc:
            status['failed'][lang] = str(exc)
            if mode == 'required':
                raise
        time.sleep(1)
    # Remove names that are shared between different languages; no fabricated unique label.
    by_name: dict[str, set[str]] = {}
    for lang, items in raw.items():
        for _, name in items:
            by_name.setdefault(name, set()).add(lang)
    data = []
    for lang, items in raw.items():
        for _, name in items:
            if len(by_name[name]) == 1:
                data.append((lang, name))
    data = sorted(set(data))
    remaining = Counter(l for l, _ in data)
    good = {lang for lang, n in remaining.items() if n >= MIN_PER_LANG}
    data = [(lab, name) for lab, name in data if lab in good]
    status.update({'status': 'ready' if len(good) >= 2 else 'insufficient',
                   'ambiguous_surface_names': sum(len(v) > 1 for v in by_name.values()),
                   'eligible_language_counts': dict(Counter(l for l, _ in data)),
                   'eligible_classes': sorted(good)})
    if mode == 'required' and len(good) < 2:
        raise RuntimeError('Wikidata returned fewer than 2 languages with >=30 distinct usable names')
    return data, status


def encode_name(name: str) -> str:
    return '^' + name + '$'


def get_scores(clf, matrix):
    raw = clf.decision_function(matrix)
    return raw.reshape(-1, 1) if raw.ndim == 1 else raw


def report_stats(classes, true, proba):
    ids = np.argmax(proba, axis=1)
    pred = [classes[i] for i in ids]
    return {
        'n': len(true), 'accuracy': float(accuracy_score(true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(true, pred)),
        'macro_f1': float(f1_score(true, pred, labels=classes, average='macro', zero_division=0)),
        'log_loss': float(log_loss(true, proba, labels=classes)),
        'brier_per_class': {lab: float(brier_score_loss([int(y == lab) for y in true], proba[:, i]))
                            for i, lab in enumerate(classes)},
        'classification_report': classification_report(true, pred, labels=classes, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(true, pred, labels=classes).tolist(),
        'classes': list(classes),
    }


def export_quantized(clf, calibrator, vectorizer, task_name: str, bits: int = 8):
    vocab = [None] * len(vectorizer.vocabulary_)
    for k, v in vectorizer.vocabulary_.items():
        vocab[v] = k
    coef = clf.coef_.astype(np.float32)
    if bits not in (8, 16):
        raise ValueError('quantization bits must be 8 or 16')
    upper = 127 if bits == 8 else 32767
    dtype = np.int8 if bits == 8 else np.dtype('<i2')
    scales = np.maximum(np.max(np.abs(coef), axis=1), 1e-9) / float(upper)
    weights = np.clip(np.rint(coef / scales[:, None]), -upper, upper).astype(dtype)
    decoded = (weights.astype(np.float32) * scales[:, None]).astype(np.float32)
    return {
        'format': 'CN-name-quantized-v1', 'task': task_name, 'weight_dtype': f'int{bits}',
        'input': {'normalize': 'NFKC', 'allowed_han': 'U+3400..U+9FFF', 'length': [2, 8],
                  'ngram': [1, 3], 'boundary': ['^', '$'], 'binary': True},
        'vocabulary': vocab, 'classes': list(clf.classes_),
        'coef_rows': int(coef.shape[0]), 'scales': scales.tolist(),
        'weights_base64': [base64.b64encode(row.tobytes()).decode('ascii') for row in weights],
        'bias': clf.intercept_.astype(float).tolist(),
        'calibration': {'classes': list(calibrator.classes_),
                        'coef': calibrator.coef_.astype(float).tolist(),
                        'intercept': calibrator.intercept_.astype(float).tolist()},
        '_quantized_weights': decoded,  # stripped before writing JSON
    }


def softmax(v: np.ndarray):
    z = v - v.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)


def quantified_predictions(export: dict, X):
    raw = np.asarray(X @ export['_quantized_weights'].T, dtype=float) + np.array(export['bias'])
    cal = export['calibration']
    logit = raw @ np.array(cal['coef']).T + np.array(cal['intercept'])
    if logit.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-np.clip(logit[:, 0], -35, 35)))
        return np.column_stack((1-p, p))
    return softmax(logit)


def save_model(task: str, rows: list[tuple[str, str]], out: Path, seed: int, features: int,
               demo: bool, full_manifest: dict):
    splits = stable_class_split(rows, seed)
    sp = {part: ([name for _, name in data], [lab for lab, _ in data])
          for part, data in splits.items()}
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), binary=True,
                                 lowercase=False, min_df=1 if demo else 2, max_features=features,
                                 preprocessor=encode_name, dtype=np.float32)
    X_train = vectorizer.fit_transform(sp['train'][0]); y_train = sp['train'][1]
    X_val = vectorizer.transform(sp['val'][0]); y_val = sp['val'][1]
    X_test = vectorizer.transform(sp['test'][0]); y_test = sp['test'][1]
    classifier = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=1000,
                               tol=1e-3, class_weight='balanced', random_state=seed,
                               average=True, n_jobs=-1)
    print(f'[train] {task}: train={len(y_train)} val={len(y_val)} test={len(y_test)} '
          f'features={X_train.shape[1]}', flush=True)
    classifier.fit(X_train, y_train)
    calibration = LogisticRegression(max_iter=400, random_state=seed)
    calibration.fit(get_scores(classifier, X_val), y_val)
    classes = list(calibration.classes_)
    assert classes == list(classifier.classes_), 'Unexpected class order mismatch'
    probabilities = calibration.predict_proba(get_scores(classifier, X_test))
    stats = report_stats(classes, y_test, probabilities)
    # Choose precision on validation data, without ever consulting the test results.
    val_gold_proba = calibration.predict_proba(get_scores(classifier, X_val))
    q8 = export_quantized(classifier, calibration, vectorizer, task, bits=8)
    val_q8 = quantified_predictions(q8, X_val)
    orig_val_f1 = report_stats(classes, y_val, val_gold_proba)['macro_f1']
    q8_val_f1 = report_stats(classes, y_val, val_q8)['macro_f1']
    q8_val_delta = float(np.max(np.abs(val_q8 - val_gold_proba)))
    if orig_val_f1 - q8_val_f1 > 0.01 or q8_val_delta > 0.03:
        print(f'[quantize] {task}: INT8 validation drift F1={orig_val_f1 - q8_val_f1:.4f}, '
              f'max probability delta={q8_val_delta:.4f}; use INT16', flush=True)
        quant = export_quantized(classifier, calibration, vectorizer, task, bits=16)
    else:
        quant = q8
    q_proba = quantified_predictions(quant, X_test)
    q_stats = report_stats(classes, y_test, q_proba)
    stats['quantized_test'] = {k: q_stats[k] for k in ('accuracy', 'macro_f1', 'log_loss')}
    stats['quantized_dtype'] = quant['weight_dtype']
    stats['int8_validation_max_probability_delta'] = q8_val_delta
    stats['quantized_max_abs_probability_difference'] = float(np.max(np.abs(q_proba - probabilities)))
    stats['quantized_mean_abs_probability_difference'] = float(np.mean(np.abs(q_proba - probabilities)))
    stats['quantized_prediction_flip_rate'] = float(np.mean(
        np.argmax(q_proba, axis=1) != np.argmax(probabilities, axis=1)))
    stats['quantized_accuracy_delta'] = float(q_stats['accuracy'] - stats['accuracy'])
    stats['quantized_macro_f1_delta'] = float(q_stats['macro_f1'] - stats['macro_f1'])
    stats['split_counts'] = {part: dict(Counter(sp[part][1])) for part in sp}
    stats['feature_count'] = int(X_train.shape[1]); stats['training_role'] = task
    stats['test_no_name_overlap'] = not bool(set(sp['test'][0]) & set(sp['train'][0]))
    out.mkdir(parents=True, exist_ok=True)
    # Reuse the original sklearn implementation for verification / future retraining.
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier,
                 'calibrator': calibration, 'metadata': {'task': task, 'seed': seed,
                 'dataset_summary': full_manifest}}, out / f'{task}.joblib', compress=3)
    quant.pop('_quantized_weights')
    (out / f'{task}.json').write_text(json.dumps(quant, ensure_ascii=False, separators=(',', ':')),
                                      encoding='utf-8')
    (out / f'{task}_metrics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                             encoding='utf-8')
    with (out / f'{task}_test_errors.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['name', 'true_label', 'predicted_label', 'predicted_confidence', 'correct'])
        for i, name in enumerate(sp['test'][0]):
            predicted = classes[int(np.argmax(probabilities[i]))]
            if predicted != y_test[i]:
                writer.writerow([name, y_test[i], predicted, round(float(probabilities[i].max()), 5), False])
    print(f'[eval] {task} test accuracy={stats["accuracy"]:.4f}, macro F1={stats["macro_f1"]:.4f}, '
          f'{quant["weight_dtype"]} macro F1={q_stats["macro_f1"]:.4f}', flush=True)
    return stats, quant


def make_demo():
    # A fixed artificial smoke-test corpus, NEVER used for production or benchmark claims.
    data = {
        'zh_native': '''张伟 王芳 李娜 李白 李明 王强 刘洋 陈静 杨帆 赵丽 张敏 周杰 林涛 郑杰 黄磊 何平 许晴 唐婉 杜甫 柳永 郎平 田亮 陈飞 张婷 李丹 何洁 赵云 朱熹 王浩 司马光 欧阳修 孙悦 周瑜 马超 林峰 陆游 张岚 郭强 谢安 韩梅 陈旭 方平 胡静 秦朗'''.split(),
        'foreign_phonetic_proxy': '''布莱恩 麦克 布鲁斯 迈克尔 约翰 詹姆斯 彼得 马丁 查理 爱德华 理查德 罗伯特 玛丽 安娜 苏珊 艾米丽 戴维 杰克逊 托马斯 卡洛斯 亚历山大 维克多 尼古拉 弗拉基米尔 伊万 阿列克谢 安德烈 莫妮卡 阿尔弗雷德 尤里 娜塔莉 米歇尔 玛格丽特 奥斯卡 弗朗西斯 露易丝 凯瑟琳 帕特里克 亨利 安德森 莱昂纳多 里卡多'''.split(),
        'ja_kanji_proxy': '''山田太郎 佐藤健 铃木一郎 田中美咲 高桥直树 伊藤博文 中村正义 小林幸子 加藤清史郎 吉田茂 山本五十六 渡边谦 松本润 石田三成 前田利家 织田信长 丰臣秀吉 德川家康 黑泽明 宫崎骏 大岛渚 小津安二郎 三岛由纪夫 川端康成 夏目漱石 芥川龙之介 村上春树 东野圭吾 长谷川博己 藤原龙也 井上真央 北川景子 大野智 木村拓哉 石川佳纯 宫本武藏 井上雄彦 森田芳光 宫本茂 远藤保仁 新垣结衣 岸田文雄'''.split(),
    }
    rows = [(lab, name) for lab, names in data.items() for name in names]
    return rows


def create_report(out: Path, all_metrics: dict, manifest: dict, demo: bool):
    parts = ['# 训练与测试报告', '',
             '> **DEMO / 流程烟雾测试：禁止把指标当作真实模型性能。**' if demo else
             '> 结果来自公开语料弱监督标注；测试集指标不等于真实世界准确率。', '',
             f'训练时间：{manifest["timestamp_utc"]}', f'随机种子：{manifest["seed"]}', '',
             '## 模型结果', '']
    for task, m in all_metrics.items():
        parts += [f'### {task}', '',
                  f'- 训练 / 验证 / 测试：{sum(m["split_counts"]["train"].values())} / '
                  f'{sum(m["split_counts"]["val"].values())} / {m["n"]}',
                  f'- 原始模型：Accuracy={m["accuracy"]:.4f}, Macro-F1={m["macro_f1"]:.4f}',
                  f'- {m["quantized_dtype"]} 浏览器模型：Accuracy={m["quantized_test"]["accuracy"]:.4f}, '
                  f'Macro-F1={m["quantized_test"]["macro_f1"]:.4f}',
                  f'- 量化概率最大绝对差：{m["quantized_max_abs_probability_difference"]:.5f}',
                  f'- 量化概率平均绝对差：{m["quantized_mean_abs_probability_difference"]:.5f}',
                  f'- 量化后预测类别变化率：{m["quantized_prediction_flip_rate"]:.5%}',
                  f'- 量化后准确率变化：{m["quantized_accuracy_delta"]:+.5f}',
                  f'- 量化后 Macro-F1 变化：{m["quantized_macro_f1_delta"]:+.5f}', '',
                  '混淆矩阵类别顺序：' + ', '.join(m['classes']), '', '```',
                  '\n'.join(' '.join(str(x) for x in row) for row in m['confusion_matrix']), '```', '']
    parts += ['## 数据有效性及局限', '',
              '- Chinese-Names-Corpus 是汇编语料；英语名目录里的翻译名并非全部来自英语，也有地名。',
              '- 日语姓名汉字表记不能自动等同汉字音译。三个主类别均是弱监督语料类别。',
              '- Wikidata P103 是人物母语，不能直接证明姓名的词源或源语言。',
              '- 按规范化中文姓名划分集合，并删除跨类别同形词；仍可能有同源姓名变体泄漏。',
              '- 0—100 分数是语料内拟合的分类分数，没有外部代表性校准，不可解释为真实人口概率。',
              '- 当前只用汉字字符串进行判断，不能保证每个姓名的历史渊源。',
              '- 指标由程序根据实际输入数据计算；不要将 DEMO 或未训练的语言分类器指标当作验证结论。', '',
              '完整数据清洗计数与下载文件 SHA256：`source_manifest.json`。',
              '机器可读指标：`*_metrics.json`。',
              '错误样本：`*_test_errors.csv`。']
    (out / 'REPORT.md').write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='GitHub Actions Linux Chinese-name supervised training')
    ap.add_argument('--output', type=Path, default=Path('outputs'))
    ap.add_argument('--cache', type=Path, default=Path('.cache/data'))
    ap.add_argument('--wikidata', choices=['best_effort', 'required', 'off'], default='best_effort')
    ap.add_argument('--max-per-class', type=int, default=120000)
    ap.add_argument('--max-features', type=int, default=35000)
    ap.add_argument('--wikidata-limit', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=20261117)
    ap.add_argument('--demo', action='store_true', help='Smoke test only. Synthetic hand-picked examples.')
    args = ap.parse_args()
    if args.max_per_class < 0 or args.max_features < 100 or args.wikidata_limit < 50:
        ap.error('Invalid sample/features/Wikidata limit')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
                'mode': 'DEMO_ONLY' if args.demo else 'PUBLIC_CORPUS_WEAK_SUPERVISION',
                'parameters': {'max_per_class': args.max_per_class, 'max_features': args.max_features,
                               'wikidata': args.wikidata, 'wikidata_limit': args.wikidata_limit},
                'labels': LABEL_INFO}
    if args.demo:
        rows = make_demo()
        manifest['source'] = 'Built-in hand-picked synthetic smoke data, NOT a representative training corpus'
        manifest['source_qc'] = dict(Counter(l for l, _ in rows))
        lang_rows, wd_status = [], {'status': 'disabled in demo'}
    else:
        source, downloaded = acquire_source(args.cache)
        manifest['source'] = downloaded
        names = {}; stats = {}
        for label, rel in SOURCE_FILES.items():
            names[label], stats[label] = load_names(source / rel)
        # Discard exact same Han spelling when it belongs to multiple training classes.
        overlap = set()
        pairs = [('zh_native', 'foreign_phonetic_proxy'),
                 ('zh_native', 'ja_kanji_proxy'),
                 ('foreign_phonetic_proxy', 'ja_kanji_proxy')]
        for a, b in pairs:
            overlap.update(names[a] & names[b])
        for label in names:
            names[label].difference_update(overlap)
        manifest['source_qc'] = stats
        manifest['ambiguous_cross_label_strings_removed'] = len(overlap)
        manifest['post_dedup_names'] = {lab: len(vals) for lab, vals in names.items()}
        rows = []
        for ix, label in enumerate(SOURCE_FILES):
            chosen = balanced_cap(names[label], args.max_per_class, args.seed + ix)
            rows.extend((label, name) for name in chosen)
        if len(rows) < 200:
            raise RuntimeError('Insufficient real names after cleaning. Refusing a fake successful training run.')
        lang_rows, wd_status = assemble_wikidata(args.cache / 'wikidata', args.wikidata_limit,
                                                 args.wikidata)
    manifest['wikidata'] = wd_status
    manifest['selected_training_pool_counts'] = dict(Counter(label for label, _ in rows))
    # Release raw corpus memory after sampling.
    if not args.demo:
        del names
    metrics = {}
    output_models = {}
    metrics['writing_type'], output_models['writing_type'] = save_model(
        'writing_type', rows, args.output, args.seed, args.max_features, args.demo, manifest)
    if lang_rows and len(set(l for l, _ in lang_rows)) >= 2:
        # Names shared across languages were excluded; evaluate native-language proxy with caveats.
        try:
            metrics['language_proxy'], output_models['language_proxy'] = save_model(
                'language_proxy', lang_rows, args.output, args.seed + 1,
                min(20000, args.max_features), args.demo, manifest)
        except ValueError as e:
            if args.wikidata == 'required':
                raise
            manifest['wikidata']['train_warning'] = str(e)
    manifest['trained_models'] = sorted(metrics)
    (args.output / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    create_report(args.output, metrics, manifest, args.demo)
    browser = args.output / 'browser'
    browser.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    shutil.copy2(project_root / 'web' / 'index.html', browser / 'index.html')
    # Local <script src> is file://-compatible; browser never performs network fetch.
    models_json = json.dumps(output_models, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (browser / 'models.js').write_text('window.CHINESE_NAME_TRAINED_MODELS=' + models_json + ';\n', encoding='utf-8')
    (browser / 'README.txt').write_text('离线解压后直接打开 index.html。数据由本次训练生成。\n', encoding='utf-8')
    print('[done] Trained', ', '.join(metrics), 'outputs:', args.output.resolve(), flush=True)
    if not args.demo and args.wikidata == 'best_effort' and 'language_proxy' not in metrics:
        print('::warning::Wikidata language model unavailable. Main supervised writing-type model saved.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'::error::{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise
