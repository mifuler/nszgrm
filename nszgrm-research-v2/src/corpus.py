"""Reproducible acquisition and weakly supervised corpus assembly.

The writing labels denote source-corpus categories. They are NOT verified personal-name
etymologies. P103 is a person's native language, NOT the etymology of their name.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from strong_data import LABELS, LANGS, load_gold, norm_name, source_names

REPO = 'https://github.com/wainshine/Chinese-Names-Corpus.git'
FILES = {
    'zh_native': 'Chinese_Names_Corpus/Chinese_Names_Corpus（120W）.txt',
    'foreign_phonetic_proxy': 'English_Names_Corpus/English_Cn_Name_Corpus（48W）.txt',
    'ja_kanji_proxy': 'Japanese_Names_Corpus/Japanese_Names_Corpus（18W）.txt',
    'zh_ancient_extra': 'Chinese_Names_Corpus/Ancient_Names_Corpus（25W）.txt',
}
def split3(group: str, seed: int):
    h=hashlib.sha256(f'{seed}:{group}'.encode('utf8')).digest()
    x=int.from_bytes(h[:8],'big') / (2**64)
    return 'train' if x<0.80 else ('val' if x<0.90 else 'test')


# The language task has exactly eight labels. For zh, pool Wikidata's
# Chinese macro-language, Mandarin and Standard Chinese native-language values.
# These are P103 person-native-language proxies, NOT verified name etymologies.
WD = {
    'zh': ('Q7850', 'Q9192', 'Q727694'),
    'en': ('Q1860',),
    'ru': ('Q7737',),
    'ja': ('Q5287',),
    'fr': ('Q150',),
    'es': ('Q1321',),
    'de': ('Q188',),
    'ko': ('Q9176',),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def acquire(cache: Path, source_dir: Path | None = None):
    if source_dir is not None:
        root = source_dir
        how = 'user-provided'
    else:
        cache.mkdir(parents=True, exist_ok=True)
        root = cache / 'Chinese-Names-Corpus'
        how = 'cached'
        if not all((root / rel).is_file() for rel in FILES.values()):
            if root.exists(): shutil.rmtree(root)
            try:
                print('[data] cloning public corpus ...', flush=True)
                subprocess.run(['git','clone','--single-branch','--depth','1',REPO,str(root)],
                               check=True, timeout=400, stdout=subprocess.DEVNULL)
                how = 'git-clone'
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
                print('[data] git clone unavailable; codeload fallback:', str(err)[:220], flush=True)
                if root.exists(): shutil.rmtree(root)
                req = urllib.request.Request('https://codeload.github.com/wainshine/Chinese-Names-Corpus/tar.gz/refs/heads/master',
                    headers={'User-Agent':'NSZGRM-ResearchTrainer/2.0 (GitHub Actions)'})
                with urllib.request.urlopen(req, timeout=150) as resp:
                    archive = cache / 'corpus.tar.gz'
                    with archive.open('wb') as dst: shutil.copyfileobj(resp, dst)
                # Only regular files with safe paths are allowed.
                root.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, 'r:gz') as tar:
                    for member in tar:
                        if not member.isfile(): continue
                        parts = Path(member.name).parts
                        if len(parts) < 2 or '..' in parts or Path(member.name).is_absolute():
                            raise RuntimeError('Unsafe archive entry')
                        rel = Path(*parts[1:])
                        dst = root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as s, dst.open('wb') as d:
                            shutil.copyfileobj(s,d)
                archive.unlink(missing_ok=True)
                how = 'codeload-tarball'
    for label, rel in FILES.items():
        if not (root / rel).is_file() or (root / rel).stat().st_size < 2000:
            raise RuntimeError(f'Missing/empty upstream file: {rel}')
    try:
        rev = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],timeout=8,text=True).strip()
    except Exception: rev = 'unavailable (archive or provided checkout)'
    return root, {'url':REPO,'revision':rev,'acquired_via':how,
        'files':{key:{'path':rel,'sha256':digest(root/rel),'bytes':(root/rel).stat().st_size}
            for key,rel in FILES.items()},
        'license_note':'Repository is Apache-2.0; aggregated upstream source licenses may vary. Raw corpora are not republished.'}


def build_writing(root: Path, gold: list[dict], cap: int, seed: int):
    assert cap >= 0
    names = {}; stats = {}
    for label, rel in FILES.items():
        names[label],stats[label] = source_names(root/rel)
    # Augment Chinese-native training pool with distinct historical names.
    names['zh_native'].update(names.pop('zh_ancient_extra'))
    overlap = set()
    for i, a in enumerate(LABELS):
        for b in LABELS[i+1:]: overlap |= names[a] & names[b]
    gold_names = {g['name'] for g in gold}
    out=[]; selected={}
    for i,label in enumerate(LABELS):
        pool = sorted(names[label] - overlap - gold_names)
        if cap and len(pool)>cap: pool = random.Random(seed+i).sample(pool,cap)
        selected[label]=len(pool)
        for name in pool:
            out.append({'name':name,'label':label,'split':split3('write:'+name,seed),
                        'group':'weak:'+name,'tier':'weak'})
    if any(selected[k] < 100 for k in LABELS): raise RuntimeError('Insufficient writing labels')
    # Gold train/val can be used; gold test is NEVER part of a training split.
    for g in gold:
        out.append(dict(g))
    return out,{'source_counts':stats,'ambiguous_spellings_excluded':len(overlap),
                'gold_spellings_excluded':len(gold_names),'sample_counts':selected}


def _wd_query(lang_qid: str, limit: int):
    return f'''SELECT DISTINCT ?person ?name WHERE {{
       ?person wdt:P31 wd:Q5; wdt:P103 wd:{lang_qid} .
       ?person rdfs:label ?name .
       FILTER(LANG(?name) = "zh" || LANG(?name) = "zh-hans")
    }} LIMIT {limit}'''


def fetch_wikidata(lang: str, qid: str, max_rows: int, cache: Path):
    cache.mkdir(parents=True,exist_ok=True)
    path = cache / f'P103_{lang}_{qid}_{max_rows}.json'
    if path.is_file():
        raw = json.loads(path.read_text(encoding='utf8'))
    else:
        url = 'https://query.wikidata.org/sparql?' + urllib.parse.urlencode(
            {'query':_wd_query(qid,max_rows),'format':'json'})
        err = None
        for attempt in range(2):
            try:
                req=urllib.request.Request(url,headers={
                    'User-Agent':'NSZGRMNameResearch/2.0 (https://github.com/mifuler/nszgrm)',
                    'Accept':'application/sparql-results+json'})
                with urllib.request.urlopen(req,timeout=95) as resp:
                    raw=json.loads(resp.read().decode('utf8'))
                path.write_text(json.dumps(raw,ensure_ascii=False),encoding='utf8')
                break
            except (OSError, ValueError) as e:
                err = e
                if attempt == 0: time.sleep(4)
        else: raise RuntimeError(f'{lang}: {err}')
    out=set()
    for row in raw.get('results',{}).get('bindings',[]):
        name=norm_name(row.get('name',{}).get('value',''))
        entity=row.get('person',{}).get('value','').rsplit('/',1)[-1]
        if name and re.fullmatch(r'Q[1-9]\d*',entity): out.add((entity,name))
    return sorted(out)


def language_records(cache: Path, gold: list[dict], max_rows: int, mode: str, seed: int,
                     min_per_language: int = 80):
    """Build P103 proxy labels. Splits group Wikidata entity and same-surface aliases."""
    if mode == 'off': return [],{'status':'disabled','warning':'Language task disabled'}
    raw={}; errors={}; query_items={}
    for lang,qids in WD.items():
        qids = (qids,) if isinstance(qids, str) else tuple(qids)
        query_items[lang] = list(qids)
        merged=set()
        try:
            for qid in qids:
                print(f'[wikidata] {lang} native-language proxy {qid}',flush=True)
                merged.update(fetch_wikidata(lang,qid,max_rows,cache))
                time.sleep(0.4)
            raw[lang] = sorted(merged)
        except Exception as e:
            errors[lang]=str(e)
            if mode == 'required': raise
            raw[lang] = sorted(merged)
    by_name=defaultdict(set); by_entity=defaultdict(set)
    for lang,rows in raw.items():
        for ent,name in rows:
            by_name[name].add(lang)
            by_entity[ent].add(lang)
    # Eliminate ambiguity among source-language proxy labels, instead of making up a label.
    known_gold_names={g['name'] for g in gold}
    records=[]
    for lang,rows in raw.items():
        for ent,name in rows:
            if len(by_name[name])==1 and len(by_entity[ent])==1 and name not in known_gold_names:
                records.append((ent,name,lang))
    # Graph union: all aliases of the same person, and all people sharing a Han label.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for ent,name,_ in records: union('Q:'+ent,'N:'+name)
    # Names from multiple entities now in same split; remove exact duplicate rows.
    records=sorted(set(records))
    count=Counter(lang for _,_,lang in records)
    eligible={lang for lang,n in count.items() if n>=min_per_language}
    rows=[]
    for ent,name,lang in records:
        if lang not in eligible: continue
        group=find('Q:'+ent)
        rows.append({'name':name,'label':lang,'split':split3('lang:'+group,seed),
                     'group':group,'tier':'proxy','entity_id':ent})
    # reviewer-provided origin language included only in matching reviewed split
    for g in gold:
        if g.get('lang') in WD and g['lang'] in eligible:
            rows.append({'name':g['name'],'label':g['lang'],'split':g['split'],
                         'group':'reviewed:'+g['group'],'tier':'gold',
                         'entity_id':g['group']})
    # Cross-split duplicate form from different entities is quarantined.
    name_splits=defaultdict(set)
    for row in rows: name_splits[row['name']].add(row['split'])
    rows=[r for r in rows if len(name_splits[r['name']])==1]
    expected=set(WD)
    missing=sorted(expected-eligible)
    status='ready' if not missing and not errors else ('partial' if eligible else 'insufficient')
    return rows,{'status':status,'failed_languages':errors,'source_rows':dict(count),
        'query_items':query_items,'expected_languages':sorted(expected),
        'eligible_languages':sorted(eligible),'missing_languages':missing,
        'final_rows':dict(Counter(r['label'] for r in rows)),
        'proxy_semantics':'Wikidata P103 is person native language, not name etymology; no verified origin-language claims.',
        'zh_proxy_items':['Q7850 Chinese','Q9192 Mandarin','Q727694 Standard Chinese']}


def checks(rows: list[dict], task: str):
    partitions=defaultdict(set); name_partition=defaultdict(set)
    for r in rows:
        partitions[r['group']].add(r['split'])
        name_partition[r['name']].add(r['split'])
    conflict=[key for key,s in partitions.items() if len(s)>1]
    if conflict: raise ValueError(f'{task}: groups cross splits: {conflict[:5]}')
    conflict=[key for key,s in name_partition.items() if len(s)>1]
    if conflict: raise ValueError(f'{task}: identical names cross splits: {conflict[:5]}')
    return {'counts':{split:dict(Counter(r['label'] for r in rows if r['split']==split))
              for split in ['train','val','test']},
            'gold_count':dict(Counter(r['split'] for r in rows if r.get('tier')=='gold')),
            'group_split_safe':True,'unique_groups':len(partitions)}


def write_review_queue(rows: list[dict], dest: Path, n=210, seed=20261117):
    dest.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    weak=defaultdict(list)
    for r in rows:
        if r['tier']=='weak':weak[r['label']].append(r['name'])
    with dest.open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['name','writing_type','origin_lang','group_id','reference_url',
                   'reviewer','split','weak_suggested_label'])
        for lab in LABELS:
            for name in rng.sample(weak[lab],min(n//3,len(weak[lab]))):
                w.writerow([name,'','','','','','',lab])
