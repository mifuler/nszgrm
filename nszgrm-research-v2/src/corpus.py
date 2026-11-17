"""Reproducible acquisition and weakly supervised corpus assembly.

The writing labels denote source-corpus categories. They are NOT verified personal-name
etymologies. P103 is a person's native language, NOT the etymology of their name.
"""
from __future__ import annotations
import csv
import hashlib
import http.client
import io
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
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
                archive = cache / 'corpus.tar.gz'
                part = cache / 'corpus.tar.gz.part'
                url = 'https://codeload.github.com/wainshine/Chinese-Names-Corpus/tar.gz/refs/heads/master'
                headers = {'User-Agent':'NSZGRM-ResearchTrainer/3.1 (GitHub Actions)',
                           'Accept':'application/vnd.github+json'}
                downloaded = False
                if shutil.which('aria2c'):
                    try:
                        _aria2_download(url,part,headers)
                        with tarfile.open(part,'r:gz') as check_archive:
                            check_archive.next()
                        os.replace(part,archive)
                        downloaded = True
                        print('[data] corpus archive downloaded with aria2',flush=True)
                    except Exception as aria_error:
                        part.unlink(missing_ok=True)
                        print('[data] aria2 archive attempt failed; using urllib fallback:',str(aria_error)[:180],flush=True)
                last_error = None
                for attempt in range(4) if not downloaded else []:
                    try:
                        req = urllib.request.Request(url,headers=headers)
                        with urllib.request.urlopen(req,timeout=180) as resp:
                            payload = resp.read()
                        part.write_bytes(payload)
                        with tarfile.open(part,'r:gz') as check_archive:
                            check_archive.next()
                        os.replace(part,archive)
                        downloaded = True
                        print(f'[data] corpus archive downloaded with urllib on attempt {attempt+1}',flush=True)
                        break
                    except (OSError,ValueError,http.client.HTTPException,urllib.error.URLError,TimeoutError) as download_error:
                        last_error = download_error
                        part.unlink(missing_ok=True)
                        if attempt < 3:
                            delay = min(30,4*(2**attempt))+random.uniform(0,1)
                            print(f'[data] archive retry {attempt+1}/4 in {delay:.1f}s',flush=True)
                            time.sleep(delay)
                if not downloaded:
                    raise RuntimeError(f'Corpus archive download failed: {last_error}')
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


def _read_wikidata_payload(path: Path):
    """Parse SPARQL JSON, tolerating unescaped C0 bytes *inside* JSON strings.

    Wikidata occasionally emits raw control characters in label values. A strict
    parser rejects the entire result, even though all other rows are complete.
    strict=False only relaxes that one JSON-string rule: truncated responses,
    broken structure and non-UTF-8 bytes still cause validation failures.
    The downstream Han-only name filter rejects any contaminated name.
    """
    payload=path.read_bytes()
    try:
        raw=json.loads(payload)
    except json.JSONDecodeError as exc:
        if 'Invalid control character' not in exc.msg:
            raise
        raw=json.loads(payload,strict=False)
        print('[wikidata] accepted structurally complete JSON containing '
              'an unescaped control character; affected names will be filtered',
              flush=True)
    if not isinstance(raw,dict) or not isinstance(raw.get('results'),dict):
        raise ValueError('Wikidata response has no results object')
    bindings=raw['results'].get('bindings')
    if not isinstance(bindings,list):
        raise ValueError('Wikidata response has no bindings list')
    # Do not quietly cache an HTML error page, a partial structure or malformed
    # per-row SPARQL bindings. A valid result may be small, so do not impose a
    # minimum cardinality here; language_records performs eligibility checks.
    for i,row in enumerate(bindings):
        if not isinstance(row,dict):
            raise ValueError(f'Wikidata row {i} is not an object')
        for key in ('person','name'):
            cell=row.get(key)
            if not isinstance(cell,dict) or not isinstance(cell.get('value'),str):
                raise ValueError(f'Wikidata row {i} has no {key}.value')
    return raw


def _read_wikidata_csv(path: Path):
    """Alternative SPARQL serialization, bypassing JSON label-escaping errors."""
    raw=path.read_bytes().decode('utf-8-sig')
    reader=csv.DictReader(io.StringIO(raw,newline=''),strict=True)
    if not reader.fieldnames or not {'person','name'}.issubset(reader.fieldnames):
        raise ValueError('Wikidata CSV has unexpected headers')
    result=set()
    for i,row in enumerate(reader):
        if None in row or row.get('person') is None or row.get('name') is None:
            raise ValueError(f'Wikidata CSV has malformed row {i+2}')
        person=row['person'].rsplit('/',1)[-1]
        if any(ord(ch)<32 for ch in row['name']):
            continue  # Reject control-bearing labels before strip()/NFKC.
        name=norm_name(row['name'])
        if name and re.fullmatch(r'Q[1-9]\d*',person):
            result.add((person,name))
    if not result:
        raise ValueError('Wikidata CSV has no valid Han names; refusing to cache')
    return sorted(result)


def _wikidata_json_rows(raw):
    result=set()
    for row in raw['results']['bindings']:
        if any(ord(ch)<32 for ch in row['name']['value']):
            continue  # strip() would otherwise hide leading/trailing controls.
        name=norm_name(row['name']['value'])
        entity=row['person']['value'].rsplit('/',1)[-1]
        if name and re.fullmatch(r'Q[1-9]\d*',entity):
            result.add((entity,name))
    return sorted(result)


def _aria2_download(url: str, dest: Path, headers: dict[str,str] | None = None):
    """Download with aria2 retry/reconnect handling; caller validates the payload."""
    exe=shutil.which('aria2c')
    if not exe:
        raise FileNotFoundError('aria2c is not installed')
    headers=headers or {'User-Agent':'NSZGRMNameResearch/3.1.1 (https://github.com/mifuler/nszgrm)',
                        'Accept':'application/sparql-results+json'}
    cmd=[exe,'--max-tries=4','--retry-wait=5','--timeout=150',
         '--connect-timeout=30','--file-allocation=none',
         '--allow-overwrite=true','--auto-file-renaming=false',
         '--check-certificate=true','--console-log-level=warn']
    for key,value in headers.items():
        cmd.extend(['--header',f'{key}: {value}'])
    cmd.extend(['--dir',str(dest.parent),'--out',dest.name,url])
    subprocess.run(cmd,check=True,timeout=700,stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


def fetch_wikidata(lang: str, qid: str, max_rows: int, cache: Path):
    cache.mkdir(parents=True,exist_ok=True)
    prefix=f'P103_{lang}_{qid}_{max_rows}'
    json_path=cache/(prefix+'.json')
    csv_path=cache/(prefix+'.csv')
    for path,parser in ((json_path,_read_wikidata_payload),
                        (csv_path,_read_wikidata_csv)):
        if path.is_file():
            try:
                content=parser(path)
                print(f'[wikidata] verified cached {path.name}',flush=True)
                return _wikidata_json_rows(content) if path.suffix=='.json' else content
            except (OSError,ValueError,UnicodeError,csv.Error):
                print(f'[wikidata] discard invalid cache: {path.name}',flush=True)
                path.unlink(missing_ok=True)

    query=_wd_query(qid,max_rows)
    last_error=None
    # Two JSON attempts, then switch serialization to CSV. Retries therefore
    # change the actual data representation, instead of repeating a broken JSON.
    for attempt in range(6):
        csv_mode=(attempt>=2)
        format_='csv' if csv_mode else 'json'
        dest=csv_path if csv_mode else json_path
        parser=_read_wikidata_csv if csv_mode else _read_wikidata_payload
        headers={'User-Agent':'NSZGRMNameResearch/3.1.1 (https://github.com/mifuler/nszgrm)',
                 'Accept':'text/csv' if csv_mode else 'application/sparql-results+json'}
        url='https://query.wikidata.org/sparql?'+urllib.parse.urlencode(
            {'query':query,'format':format_})
        part=dest.with_name(dest.name+'.part')
        part.unlink(missing_ok=True)
        for client in ('aria2','urllib'):
            if client=='aria2' and not shutil.which('aria2c'):
                continue
            try:
                if client=='aria2':
                    _aria2_download(url,part,headers)
                else:
                    req=urllib.request.Request(url,headers=headers)
                    with urllib.request.urlopen(req,timeout=180) as response:
                        payload=response.read()
                    part.write_bytes(payload)
                content=parser(part)
                result=(_wikidata_json_rows(content) if not csv_mode else content)
                if not result:
                    raise ValueError('No valid Han names in Wikidata response')
                os.replace(part,dest)
                print(f'[wikidata] {client} verified {lang}/{qid} '
                      f'({len(result)} names; format={format_})',flush=True)
                return result
            except (OSError,ValueError,UnicodeError,csv.Error,
                    http.client.HTTPException,urllib.error.URLError,
                    subprocess.CalledProcessError,subprocess.TimeoutExpired,TimeoutError) as e:
                last_error=e
                part.unlink(missing_ok=True)
                print(f'[wikidata] {client} {format_} attempt {attempt+1}/6: '
                      f'{str(e)[:190]}',flush=True)
        if attempt<5:
            delay=min(30,3*(2**attempt))+random.uniform(0,1)
            print(f'[wikidata] retry with {"CSV" if attempt>=1 else "JSON"} '
                  f'after {delay:.1f}s',flush=True)
            time.sleep(delay)
    raise RuntimeError(f'{lang}/{qid}: Wikidata JSON and CSV failed after 6 attempts: {last_error}')

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
