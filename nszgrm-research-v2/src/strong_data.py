"""Dataset assembly for Chinese-name phonetic transliteration research.

Two disjoint evidence tiers are kept separate:
- automatic corpus membership = weak supervision, never gold
- manually reviewed and sourced rows = gold. No automatic promotion.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from opencc_bridge import OPENCC_JS_VERSION, convert_many as opencc_convert_many

LABELS = ('zh_native', 'foreign_phonetic_proxy', 'ja_kanji_proxy')
LANGS = ('zh', 'en', 'ru', 'ja', 'fr', 'es', 'de', 'ko')
HAN = re.compile(r'^[\u3400-\u9fff]{2,8}$')
HAN_ANY = re.compile(r'[\u3400-\u9fff]')
GOLD_COLUMNS = ('name', 'writing_type', 'origin_lang', 'group_id', 'reference_url', 'reviewer', 'split')


def normalize_names(values) -> list[str]:
    """NFKC-normalize and convert every Han-only name to Simplified Chinese.

    OpenCC runs before deduplication, label-conflict checks and data splitting.
    Japanese kanji names pass through jp2cn and t2s converters while retaining
    their original corpus label.
    """
    prepared = [unicodedata.normalize('NFKC', str(value or '')).strip() for value in values]
    # Run OpenCC on every text field containing Han, including mixed-script
    # candidates; retain the original accepted-name filter afterwards.
    eligible = [i for i, value in enumerate(prepared) if HAN_ANY.search(value)]
    converted = opencc_convert_many([prepared[i] for i in eligible])
    output = [''] * len(prepared)
    for index, value in zip(eligible, converted):
        output[index] = value if HAN.fullmatch(value) else ''
    return output


def norm_name(text: str) -> str:
    return normalize_names([text])[0]


def hash_partition(group: str, seed: int, valid_fraction: float = 0.1) -> str:
    h = hashlib.sha256(f'{seed}:{group}'.encode()).digest()
    return 'val' if int.from_bytes(h[:8], 'big') / 2**64 < valid_fraction else 'train'


def group_norm(s: str) -> str:
    value = unicodedata.normalize('NFKC', str(s or '')).strip()
    if HAN_ANY.search(value):
        value = opencc_convert_many([value])[0]
    return value.lower()


def source_names(path: Path):
    # Convert unique raw spellings in batches, before deduplication and label overlap checks.
    raw_counts = Counter()
    counted = 0
    with path.open(encoding='utf-8-sig', errors='replace') as f:
        for raw in f:
            counted += 1
            first = re.split(r'[\t,，|\s]', raw.strip(), maxsplit=1)[0]
            value = unicodedata.normalize('NFKC', first).strip()
            if HAN_ANY.search(value):
                raw_counts[value] += 1

    raw_names = list(raw_counts)
    simplified_names = normalize_names(raw_names)
    names = set()
    sources_by_result = defaultdict(set)
    converted_rows = 0
    converted_results = set()
    for raw_name, simplified in zip(raw_names, simplified_names):
        if not simplified:
            continue
        names.add(simplified)
        sources_by_result[simplified].add(raw_name)
        if simplified != raw_name:
            converted_rows += raw_counts[raw_name]
            converted_results.add(simplified)

    collisions = [
        {'simplified': result, 'sources': sorted(sources)}
        for result, sources in sources_by_result.items() if len(sources) > 1
    ]
    collisions.sort(key=lambda item: item['simplified'])
    return names, {
        'lines': counted,
        'unique_raw_valid': sum(bool(HAN.fullmatch(raw)) for raw in raw_names),
        'unique_valid': len(names),
        'opencc_js_version': OPENCC_JS_VERSION,
        'opencc_converted_rows': converted_rows,
        'opencc_converted_unique_names': len(converted_results),
        'simplified_name_collisions': len(collisions),
        'simplified_name_collision_examples': collisions[:10],
        '_opencc_converted_targets': converted_results,
        '_opencc_source_forms': {result: sorted(sources) for result, sources in sources_by_result.items()
                                 if any(source != result for source in sources)},
    }


def load_gold(path: Path):
    """Only trust CSV rows with explicit reviewer + reference URL + split.

    User-provided labels are never silently inferred. Foreign provenance can be
    ambiguous: these rows should not be included as gold until adjudicated.
    """
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        if not r.fieldnames or not set(GOLD_COLUMNS).issubset(r.fieldnames):
            raise ValueError('Gold CSV missing required headers: ' + ','.join(GOLD_COLUMNS))
        rows = []
        for i, item in enumerate(r, 2):
            if not any(v.strip() for v in item.values() if isinstance(v, str)):
                continue
            name = norm_name(item.get('name', ''))
            label = item.get('writing_type', '').strip()
            group = group_norm(item.get('group_id', ''))
            lang = item.get('origin_lang', '').strip()
            split = item.get('split', '').strip().lower()
            url = item.get('reference_url', '').strip()
            reviewer = item.get('reviewer', '').strip()
            if not name or label not in LABELS or not group or split not in ('train', 'val', 'test'):
                raise ValueError(f'Invalid gold CSV line {i}; missing name/group/label/split')
            if lang and lang not in LANGS and lang != 'unknown':
                raise ValueError(f'Invalid origin_lang at line {i}: {lang}')
            if not url.startswith(('https://', 'http://')) or not reviewer:
                raise ValueError(f'Gold CSV line {i} requires reviewer and evidence URL')
            rows.append({'name': name, 'label': label, 'group': group, 'lang': lang,
                         'split': split, 'url': url, 'reviewer': reviewer, 'tier': 'gold'})
    # Forbid cross-split aliases, duplicate names, contradictory labels.
    group_splits = defaultdict(set)
    name_splits = defaultdict(set)
    name_labels = defaultdict(set)
    for r in rows:
        group_splits[r['group']].add(r['split'])
        name_splits[r['name']].add(r['split'])
        name_labels[r['name']].add(r['label'])
    ambiguous = [x for x, s in group_splits.items() if len(s) > 1]
    ambiguous += [x for x, s in name_splits.items() if len(s) > 1]
    ambiguous += [x for x, s in name_labels.items() if len(s) > 1]
    if ambiguous:
        raise ValueError('Gold: cross-split/group conflicts or contradictory labels: ' + str(ambiguous[:10]))
    return rows


def build_weak(source_dir: Path, gold_rows: list[dict], max_per_class: int, seed: int):
    paths = {
        'zh_native': 'Chinese_Names_Corpus/Chinese_Names_Corpus（120W）.txt',
        'foreign_phonetic_proxy': 'English_Names_Corpus/English_Cn_Name_Corpus（48W）.txt',
        'ja_kanji_proxy': 'Japanese_Names_Corpus/Japanese_Names_Corpus（18W）.txt',
    }
    all_names = {}
    raw_info = {}
    for label, rel in paths.items():
        path = source_dir / rel
        if not path.exists():
            raise FileNotFoundError(f'Missing source {path}: download Chinese-Names-Corpus first')
        all_names[label], raw_info[label] = source_names(path)
    overlaps = (all_names[LABELS[0]] & all_names[LABELS[1]]) | \
               (all_names[LABELS[0]] & all_names[LABELS[2]]) | \
               (all_names[LABELS[1]] & all_names[LABELS[2]])
    opencc_cross_label = []
    for i, left in enumerate(LABELS):
        for right in LABELS[i+1:]:
            for name in sorted(all_names[left] & all_names[right]):
                if (name in raw_info[left].get('_opencc_converted_targets', set()) or
                        name in raw_info[right].get('_opencc_converted_targets', set())):
                    opencc_cross_label.append({
                        'simplified': name,
                        'labels': [left, right],
                        'source_forms': {
                            left: raw_info[left].get('_opencc_source_forms', {}).get(name, [name]),
                            right: raw_info[right].get('_opencc_source_forms', {}).get(name, [name]),
                        },
                    })
    zh_ja_collisions = [item for item in opencc_cross_label
                        if item['labels'] == ['zh_native', 'ja_kanji_proxy']]
    for source_stats in raw_info.values():
        source_stats.pop('_opencc_converted_targets', None)
        source_stats.pop('_opencc_source_forms', None)
    gold_names = {r['name'] for r in gold_rows}
    by_label = {}
    for i, label in enumerate(LABELS):
        candidates = sorted(all_names[label] - overlaps - gold_names)
        if max_per_class and len(candidates) > max_per_class:
            candidates = random.Random(seed + i).sample(candidates, max_per_class)
        by_label[label] = candidates
    rows = []
    for label, names in by_label.items():
        for name in names:
            rows.append({'name': name, 'label': label, 'group': name,
                         'split': hash_partition(name, seed), 'tier': 'weak',
                         'lang': '', 'url': '', 'reviewer': ''})
    return rows, {'source_counts': raw_info, 'cross_label_spellings_excluded': len(overlaps),
                  'opencc_cross_label_collision_count': len(opencc_cross_label),
                  'opencc_cross_label_collision_examples': opencc_cross_label[:20],
                  'chinese_japanese_same_name_overlap_after_normalization_count': len(all_names['zh_native'] & all_names['ja_kanji_proxy']),
                  'opencc_chinese_japanese_collision_count': len(zh_ja_collisions),
                  'opencc_chinese_japanese_collision_examples': zh_ja_collisions[:20],
                  'gold_spellings_excluded': len(gold_names),
                  'weak_selected': {x: len(by_label[x]) for x in LABELS}}


def combine(weak: list[dict], gold: list[dict]):
    # A gold test name is forbidden from weak training, even with unrelated metadata.
    verified_names = {r['name'] for r in gold}
    weak = [r for r in weak if r['name'] not in verified_names]
    return weak + gold


def audit_partition(rows: list[dict]):
    groups = defaultdict(set)
    labels = Counter()
    for r in rows:
        groups[r['group']].add(r['split'])
        labels[(r['tier'], r['split'], r['label'])] += 1
    offenders = [g for g, s in groups.items() if len(s) != 1]
    if offenders:
        raise ValueError('Groups span multiple partitions: ' + str(offenders[:8]))
    return {'counts': [{'tier': t, 'split': s, 'label': l, 'count': n}
                       for (t,s,l), n in sorted(labels.items())],
            'unique_groups': len(groups), 'disjoint_group_splits': True}


def write_jsonl(rows, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_review_queue(rows: list[dict], dest: Path, size: int = 150, seed: int = 20261117):
    """Human labeling queue deliberately has BLANK target labels."""
    randomizer = random.Random(seed)
    candidates = [x for x in rows if x['tier'] == 'weak']
    bylabel = defaultdict(list)
    for r in candidates:
        bylabel[r['label']].append(r)
    queue = []
    for label in LABELS:
        pool = bylabel[label]
        queue.extend(randomizer.sample(pool, min(len(pool), size // len(LABELS))))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'writing_type', 'origin_lang', 'group_id', 'reference_url',
                    'reviewer', 'split', 'weak_suggested_label'])
        for r in queue:
            w.writerow([r['name'], '', '', '', '', '', '', r['label']])
    return len(queue)
