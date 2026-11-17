"""Train-only hard cases and Chinese-language supplemental names.

Source labels remain weak/small manual-rule labels; they are not gold test evidence.
"""
from __future__ import annotations
import csv
import random
from collections import Counter
from pathlib import Path
from corpus import FILES
from strong_data import LABELS, norm_name, source_names


def load_hard_cases(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Hard-case training file missing: {path}')
    out={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'name','writing_type','evidence_tier'}.issubset(reader.fieldnames or []):
            raise ValueError('Hard-case CSV columns missing')
        for line,row in enumerate(reader,2):
            name=norm_name(row['name'])
            label=row['writing_type'].strip()
            if not name or label not in LABELS:
                raise ValueError(f'Invalid hard-case entry at line {line}')
            if name in out and out[name]!=label:
                raise ValueError(f'Conflicting hard-case label for {name}')
            out[name]=label
    if len(out)<25:
        raise ValueError('Hard-case CSV must preserve the 25 existing corrections')
    return out


def load_chinese_names(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Chinese supplemental names file missing: {path}')
    names={norm_name(s) for s in path.read_text(encoding='utf-8-sig').splitlines()}
    names.discard('')
    return sorted(names)


def add_writing_hard_cases(rows, hard_cases, gold, seed=20261117, repeats=48):
    if repeats<1 or repeats>512:raise ValueError('hard repeat out of range')
    gold_by_name={g['name']:g for g in gold}
    included={}
    skipped={}
    for name,label in hard_cases.items():
        existing=gold_by_name.get(name)
        if existing:
            if existing['label']!=label:raise ValueError(f'Hard-case/gold conflicting label: {name}')
            skipped[name]='gold_'+existing['split']
        else:
            included[name]=label
    # Remove source-weak duplicates so hard cases cannot enter validation/test.
    names=set(hard_cases)
    out=[r for r in rows if r['name'] not in names or r.get('tier')=='gold']
    for name,label in sorted(included.items()):
        for i in range(repeats):
            out.append({'name':name,'label':label,'split':'train','group':'hard:'+name,
                        'tier':'hard_case','case_repeat':i,'source':'user_reviewed_rule'})
    return out,{'repeated_train_rows':repeats*len(included),
                'unique_train_cases':len(included),'skipped_gold_cases':skipped,
                'note':'Training-only memorization reinforcement; never added to validation or test'}


def add_zh_language_support(rows, root, hard_cases, chinese_names, gold,
                            seed=20261117, cap=8000,hard_repeat=12):
    if cap<0 or hard_repeat<0:raise ValueError('augmentation parameters cannot be negative')
    # Filter any exact ambiguous Han spelling present in Japanese/translation corpora.
    native,_=source_names(root/FILES['zh_native'])
    historical,_=source_names(root/FILES['zh_ancient_extra'])
    japanese,_=source_names(root/FILES['ja_kanji_proxy'])
    translated,_=source_names(root/FILES['foreign_phonetic_proxy'])
    used={r['name'] for r in rows}|{g['name'] for g in gold}
    excluded=used|japanese|translated|{n for n,c in hard_cases.items() if c!='zh_native'}
    candidates=sorted((native|historical)-excluded)
    selected=random.Random(seed+1153).sample(candidates,min(cap,len(candidates)))
    out=list(rows)
    for name in selected:
        out.append({'name':name,'label':'zh','split':'train','group':'zh-supplement:'+name,
                    'tier':'chinese_corpus_train_only','source_property':'Chinese_Names_Corpus',
                    'entity_id':'train-only:'+name})
    curated_added=[]
    curated_skipped=[]
    for name in chinese_names:
        if name in excluded or name in selected:
            curated_skipped.append(name)
            continue
        curated_added.append(name)
        for i in range(hard_repeat):
            out.append({'name':name,'label':'zh','split':'train','group':'zh-curated:'+name,
                        'tier':'chinese_curated_train_only','source_property':'curated_common_chinese_names',
                        'entity_id':'curated:'+name,'case_repeat':i})
    return out,{'corpus_train_only':len(selected),
                'curated_unique':len(curated_added),
                'curated_repeated_rows':len(curated_added)*hard_repeat,
                'curated_excluded_or_duplicate':curated_skipped,
                'filter':'Excluded Japanese and translated-name homographs, all P103/P1412 names and gold names',
                'evaluation':'P103/P1412 validation and held-out tests remain unchanged'}
