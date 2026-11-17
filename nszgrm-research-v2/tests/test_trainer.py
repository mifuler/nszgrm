import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from strong_data import LANGS, norm_name, load_gold
from corpus import WD, fetch_wikidata, language_records, split3, checks
from train_all import BYTE_CAP,choose_weights,temperature,train_task
import numpy as np

class TrainerTests(unittest.TestCase):
    def test_exact_byte_budget(self):
        self.assertEqual(BYTE_CAP,200_000_000)

    def test_corpus_name_normalization(self):
        self.assertEqual(norm_name('布莱恩'),'布莱恩')
        self.assertEqual(norm_name('山田太郎'),'山田太郎')
        self.assertEqual(norm_name('张·伟'),'')
        self.assertEqual(norm_name('Mary'),'')

    def test_gold_requires_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'gold.csv'
            file.write_text('name,writing_type,origin_lang,group_id,reference_url,reviewer,split\n'
                            '林肯,foreign_phonetic_proxy,en,lincoln,,,test\n',encoding='utf8')
            with self.assertRaises(ValueError):load_gold(file)

    def test_partition_is_stable(self):
        for s in ['林肯','布莱恩','王小明']:
            self.assertEqual(split3(s,20261117),split3(s,20261117))

    def test_group_overlap_is_error(self):
        with self.assertRaises(ValueError):
            checks([dict(name='林肯',label='a',group='lincoln',split='train'),
                    dict(name='林肯二',label='a',group='lincoln',split='test')],'test')


    def test_truncated_wikidata_cache_is_discarded_and_retried(self):
        import json
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self):
                return json.dumps({'results':{'bindings':[{
                    'person':{'value':'https://www.wikidata.org/entity/Q123'},
                    'name':{'value':'林肯'}}]}}).encode('utf-8')
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)
            cached=cache/'P103_en_Q1860_10.json'
            cached.write_text('{"results":',encoding='utf-8')
            with patch('corpus.shutil.which',return_value=None), \
                 patch('corpus.urllib.request.urlopen',return_value=FakeResponse()):
                rows=fetch_wikidata('en','Q1860',10,cache)
            self.assertEqual(rows,[('Q123','林肯')])
            self.assertEqual(json.loads(cached.read_text(encoding='utf-8'))['results']['bindings'][0]['name']['value'],'林肯')

    def test_wikidata_unescaped_control_character_parses_but_is_filtered(self):
        import json
        from corpus import _read_wikidata_payload,_wikidata_json_rows
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'bad_control.json'
            rows=[{'person':{'value':'https://www.wikidata.org/entity/Q7'},
                   'name':{'value':'汉\x0b字'}},
                  {'person':{'value':'https://www.wikidata.org/entity/Q8'},
                   'name':{'value':'林肯'}}]
            # Insert actual VT, not the legal escaped JSON representation.
            raw=json.dumps({'results':{'bindings':rows}},ensure_ascii=False)
            file.write_text(raw.replace('\\u000b','\x0b'),encoding='utf8')
            with self.assertRaises(json.JSONDecodeError):
                json.loads(file.read_bytes())
            output=_wikidata_json_rows(_read_wikidata_payload(file))
            self.assertEqual(output,[('Q8','林肯')])
            rows[0]['name']['value']='\x0b林肯'
            self.assertEqual(_wikidata_json_rows({'results':{'bindings':rows}}),[('Q8','林肯')])

    def test_wikidata_truncated_json_still_rejected(self):
        from corpus import _read_wikidata_payload
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'truncated.json'
            file.write_bytes(b'{"results":{"bindings":[{"person":{"value":"x"}')
            with self.assertRaises(ValueError):
                _read_wikidata_payload(file)

    def test_wikidata_csv_alternative_parsing(self):
        from corpus import _read_wikidata_csv
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'response.csv'
            file.write_text('person,name\nhttps://www.wikidata.org/entity/Q10,"林,肯"\n'
                            'https://www.wikidata.org/entity/Q11,林肯\n',encoding='utf8')
            self.assertEqual(_read_wikidata_csv(file),[('Q11','林肯')])

    def test_wikidata_invalid_json_falls_back_to_csv(self):
        import json
        from corpus import fetch_wikidata
        class FakeResponse:
            def __init__(self,payload):self.payload=payload
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self):return self.payload
        csv_payload=('person,name\nhttps://www.wikidata.org/entity/Q12,林肯\n').encode()
        calls=[]
        def mock_get(request, timeout):
            calls.append(request.full_url)
            if 'format=csv' in request.full_url:
                return FakeResponse(csv_payload)
            return FakeResponse(b'{"results":')
        with tempfile.TemporaryDirectory() as temp:
            with patch('corpus.shutil.which',return_value=None), \
                 patch('corpus.urllib.request.urlopen',side_effect=mock_get), \
                 patch('corpus.time.sleep',return_value=None):
                rows=fetch_wikidata('de','Q188',5000,Path(temp))
            self.assertEqual(rows,[('Q12','林肯')])
            self.assertTrue(any('format=csv' in u for u in calls))
            self.assertTrue((Path(temp)/'P103_de_Q188_5000.csv').is_file())

    def test_language_model_has_eight_classes_including_chinese(self):
        self.assertEqual(set(LANGS), {'zh','en','fr','de','es','ru','ja','ko'})
        self.assertEqual(len(LANGS), 8)

    def test_chinese_native_language_uses_three_wikidata_items(self):
        self.assertEqual(WD['zh'], ('Q7850','Q9192','Q727694'))

    def test_language_records_merge_chinese_qids_and_require_all_eight(self):
        def fake_fetch(lang, qid, max_rows, cache):
            if lang == 'zh':
                idx = WD['zh'].index(qid)
                start, count = idx * 40, 40
            else:
                start, count = 0, 120
            rows = []
            lang_index = list(WD).index(lang)
            for i in range(start, start + count):
                code = lang_index * 1000 + i
                name = chr(0x4e00 + code) + chr(0x6000 + code)
                rows.append((f'Q{100000 + code}', name))
            return rows

        with tempfile.TemporaryDirectory() as temp:
            with patch('corpus.fetch_wikidata', side_effect=fake_fetch), patch('corpus.time.sleep', return_value=None):
                rows, meta = language_records(Path(temp), [], 120, 'required', 20261117, min_per_language=80)
        self.assertEqual(meta['status'], 'ready')
        self.assertEqual(set(meta['eligible_languages']), set(LANGS))
        self.assertEqual(meta['source_rows']['zh'], 120)
        self.assertEqual(set(r['label'] for r in rows), set(LANGS))


    def test_gold_accepts_chinese_language_code(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'gold.csv'
            file.write_text(
                'name,writing_type,origin_lang,group_id,reference_url,reviewer,split\n'
                '林肯,foreign_phonetic_proxy,zh,lincoln,https://example.org/source,reviewer-a,test\n',
                encoding='utf8')
            rows=load_gold(file)
            self.assertEqual(rows[0]['lang'],'zh')

    def test_language_training_rejects_partial_class_set(self):
        rows=[{'name':'张伟','label':lab,'group':lab,'split':'train'} for lab in LANGS[:-1]]
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(RuntimeError,'requires all eight classes'):
                train_task('language',rows,Path(t),None)

    def test_fusion_is_simplex(self):
        y=np.array([0,0,1,1,2,2])
        base=np.eye(3)[y]*0.8+0.2/3
        alt=np.roll(base,1,axis=1)
        w=choose_weights([base,base,alt,alt],y)
        self.assertEqual(len(w),4)
        self.assertTrue(all(x>=.029 for x in w))
        self.assertAlmostEqual(sum(w),1.0,places=6)

if __name__=='__main__':unittest.main()
