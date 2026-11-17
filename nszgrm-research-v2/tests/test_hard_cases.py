import sys
import csv
import tempfile
import unittest
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hard_cases import load_hard_cases, add_writing_hard_cases, add_zh_language_support
from corpus import checks, FILES
from architectures import train_neural

ROOT=Path(__file__).resolve().parents[1]

class HardCaseTests(unittest.TestCase):
    def test_exact_25_user_examples_present(self):
        cases=load_hard_cases(ROOT/'data/hard_cases.csv')
        self.assertEqual(len(cases),25)
        self.assertEqual(cases['路易'],'foreign_phonetic_proxy')
        self.assertEqual(cases['皮日休'],'zh_native')
        self.assertEqual(cases['郁达夫'],'zh_native')

    def test_train_only_case_never_leaks_to_validation(self):
        cases={'路易':'foreign_phonetic_proxy'}
        rows=[dict(name='路易',label='zh_native',split='test',group='weak:路易',tier='weak'),
              dict(name='王伟',label='zh_native',split='test',group='weak:王伟',tier='weak')]
        result,info=add_writing_hard_cases(rows,cases,[],repeats=5)
        self.assertEqual(len([r for r in result if r['name']=='路易']),5)
        self.assertEqual({r['split'] for r in result if r['name']=='路易'},{'train'})
        self.assertEqual(info['repeated_train_rows'],5)
        checks(result,'writing')

    def test_gold_test_remains_excluded_from_training(self):
        cases={'路易':'foreign_phonetic_proxy'}
        gold=[dict(name='路易',label='foreign_phonetic_proxy',split='test',group='gold:louie',tier='gold')]
        result,info=add_writing_hard_cases(gold,cases,gold,repeats=10)
        self.assertEqual(result,gold)
        self.assertEqual(info['skipped_gold_cases']['路易'],'gold_test')

    def test_conflicting_hard_case_vs_gold_raises(self):
        with self.assertRaises(ValueError):
            add_writing_hard_cases([],{'路易':'foreign_phonetic_proxy'},
                [dict(name='路易',label='zh_native',split='test',group='gold',tier='gold')])

    def test_zh_aux_excludes_japanese_foreign_and_test_names(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,rel in FILES.items():
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                data={'zh_native':'中国人\n张三\n王伟\n','zh_ancient_extra':'诸葛亮\n',
                      'ja_kanji_proxy':'中国人\n',
                      'foreign_phonetic_proxy':'张三\n'}[name]
                p.write_text(data,encoding='utf-8')
            origin=[dict(name='王伟',label='zh',split='test',group='WD:王伟',tier='proxy',entity_id='Q2')]
            output,meta=add_zh_language_support(origin,root,{},['诸葛亮','张三'],[],seed=7,cap=0,hard_repeat=4)
            supplemental=[r for r in output if r['name']!='王伟']
            self.assertTrue(all(r['label']=='zh' and r['split']=='train' for r in supplemental))
            self.assertTrue(all(r['name'] not in {'中国人','张三','王伟'} for r in supplemental))
            self.assertEqual(meta['curated_unique'],1)
            checks(output,'language')

if __name__=='__main__':unittest.main()

class ReleaseCacheTests(unittest.TestCase):
    def test_cached_p1412_language_expansion_is_permitted(self):
        from release_dataset_cache import NAME
        self.assertIsNotNone(NAME.fullmatch('P103_ru_Q7737_5000_P103-han-alias-v2.json'))
        self.assertIsNotNone(NAME.fullmatch('P1412_ru_Q7737_5000_P1412-no-P103-v2.csv'))
        self.assertIsNone(NAME.fullmatch('P999_ru_Q7737_5000.json'))
