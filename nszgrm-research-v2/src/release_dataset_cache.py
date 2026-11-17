#!/usr/bin/env python3
"""Restore/publish verified Wikidata SPARQL snapshots as a non-latest Release asset.

The cache is independent of short-lived Actions artifacts. Only complete,
validated, hash-verified SPARQL responses are republished, never source corpora
with different redistribution terms. Every failure preserves previously saved
Release contents: publication is a single replacement asset containing the
union of restored snapshots and newly completed requests.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from corpus import _read_wikidata_payload, _read_wikidata_csv, _wikidata_json_rows

CACHE_SCHEMA = 2
TAG = 'nszgrm-wikidata-cache-v2'
ASSET = 'nszgrm-wikidata-verified-v2.tar.gz'
NAME = re.compile(r'^P(?:103|1412)_[a-z]{2}_[A-Za-z0-9_-]+_\d+(?:_[A-Za-z0-9_-]+)?\.(?:json|csv)$')
MAX_EACH = 25_000_000
MAX_TOTAL = 200_000_000


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def verified(path: Path):
    if path.suffix == '.json':
        return len(_wikidata_json_rows(_read_wikidata_payload(path))) > 0
    if path.suffix == '.csv':
        return len(_read_wikidata_csv(path)) > 0
    return False


def candidates(root: Path):
    entries = []
    if not root.exists():
        return entries
    for path in sorted(root.iterdir()):
        if not path.is_file() or not NAME.fullmatch(path.name):
            continue
        size = path.stat().st_size
        if size < 20 or size > MAX_EACH:
            continue
        try:
            if verified(path):
                content = path.read_bytes()
                entries.append((path.name,content))
        except (OSError,ValueError,UnicodeError):
            print(f'[cache] skip malformed {path.name}', flush=True)
    return entries


def build_snapshot(root: Path, archive: Path):
    entries = candidates(root)
    if not entries:
        print('[cache] no valid SPARQL responses; will not publish an empty cache',flush=True)
        return 0
    if sum(len(content) for _,content in entries)>MAX_TOTAL:
        raise RuntimeError('Cache snapshot exceeds size cap')
    manifest={
        'schema':CACHE_SCHEMA,
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'upstream':'https://query.wikidata.org/sparql',
        'usage':'Wikidata P103 native-language proxies and explicitly marked P1412-only fallbacks',
        'files':{name:{'sha256':sha256(content),'bytes':len(content)} for name,content in entries},
    }
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'w:gz') as tar:
        for name,content in entries+[('manifest.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode())]:
            info=tarfile.TarInfo(name)
            info.size=len(content)
            info.mode=0o644
            info.mtime=0
            tar.addfile(info,io.BytesIO(content))
    return len(entries)


def restore_snapshot(archive: Path,root: Path):
    if not archive.is_file():
        return 0
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)>1000 or any(not m.isfile() for m in members):
            raise ValueError('Cache archive contains unexpected entries')
        names=[m.name for m in members]
        if len(names)!=len(set(names)) or 'manifest.json' not in names:
            raise ValueError('Cache archive manifest missing or duplicate filenames')
        manifest_bytes=tar.extractfile('manifest.json').read()
        if len(manifest_bytes)>1_000_000:
            raise ValueError('Cache manifest too large')
        manifest=json.loads(manifest_bytes)
        if manifest.get('schema')!=CACHE_SCHEMA or not isinstance(manifest.get('files'),dict):
            raise ValueError('Unknown cache manifest version')
        described=set(manifest['files'])
        if described!=set(names)-{'manifest.json'}:
            raise ValueError('Cache manifest entries mismatch archive')
        if sum(int(meta.get('bytes',0)) for meta in manifest['files'].values())>MAX_TOTAL:
            raise ValueError('Cache manifest exceeds maximum size')
        root.mkdir(parents=True,exist_ok=True)
        restored=0
        for name,meta in manifest['files'].items():
            if not NAME.fullmatch(name):
                raise ValueError(f'Invalid cache asset filename: {name}')
            member=tar.getmember(name)
            declared=int(meta.get('bytes',0))
            if not 20<=declared<=MAX_EACH or member.size!=declared:
                raise ValueError('Cache file size mismatch: '+name)
            content=tar.extractfile(member).read(MAX_EACH+1)
            if len(content)!=declared or sha256(content)!=meta['sha256']:
                raise ValueError('Cache checksum mismatch: '+name)
            path=root/name
            if path.exists():
                try:
                    if verified(path):
                        continue  # Prefer a valid (possibly newer) Actions cache entry.
                except (OSError,ValueError,UnicodeError):
                    pass
            # Reuse a private temporary directory to preserve exact extension.
            with tempfile.TemporaryDirectory() as tempdir:
                temp_path=Path(tempdir)/name
                temp_path.write_bytes(content)
                if not verified(temp_path):
                    raise ValueError('Structurally invalid cache payload: '+name)
            temp=path.with_name(path.name+'.release-part')
            temp.write_bytes(content)
            os.replace(temp,path)
            restored+=1
    return restored


def run_gh(*arguments,allow_error=False):
    result=subprocess.run(['gh',*arguments],capture_output=True,text=True)
    if result.returncode and not allow_error:
        raise RuntimeError(f'gh {arguments[0]} failed: {(result.stderr or result.stdout)[-400:]}')
    return result


def restore(root: Path,tag: str,asset: str):
    with tempfile.TemporaryDirectory() as tempdir:
        res=run_gh('release','download',tag,'--pattern',asset,'--dir',tempdir,allow_error=True)
        file=Path(tempdir)/asset
        if res.returncode or not file.is_file():
            print('[cache] no existing Release snapshot; Actions cache and live download remain available',flush=True)
            return 0
        count=restore_snapshot(file,root)
        print(f'[cache] restored {count} validated new snapshots from Release {tag}',flush=True)
        return count


def publish(root: Path,tag: str,asset: str):
    with tempfile.TemporaryDirectory() as tempdir:
        file=Path(tempdir)/asset
        count=build_snapshot(root,file)
        if count==0:return 0
        view=run_gh('release','view',tag,allow_error=True)
        if view.returncode:
            run_gh('release','create',tag,'--title','NSZGRM Wikidata eight-language query cache',
                   '--notes','Validated SPARQL data only. P103 = person native language; '
                   'P1412 fallback = language ability, not name etymology. '
                   'Files are SHA256-checked and cache format versioned.',
                   '--latest=false','--prerelease')
        run_gh('release','upload',tag,str(file),'--clobber')
        print(f'[cache] uploaded {count} validated response files to Release {tag}',flush=True)
        return count


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['restore','publish'])
    ap.add_argument('--root',type=Path,default=Path('.cache/nszgrm/wikidata'))
    ap.add_argument('--tag',default=TAG)
    ap.add_argument('--asset',default=ASSET)
    args=ap.parse_args()
    try:
        if args.command=='restore':restore(args.root,args.tag,args.asset)
        else:publish(args.root,args.tag,args.asset)
    except Exception as exc:
        # Cache must not mask a successful training result or replace the
        # independent Actions cache. Print precise failure for diagnosis.
        print(f'[cache] Release {args.command} failed: {exc}',file=sys.stderr)
        if args.command=='restore':return 0
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
