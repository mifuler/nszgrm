"""Regression tests for the missing-Russian-class failure in run 44050312633."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from corpus import WD, _wd_query_expanded, language_records


def rows(lang,start,count):
    labels=list(WD)
    index=labels.index(lang)
    result=[]
    for n in range(start,start+count):
        code=index*3500+n
        # Han characters in the supported U+4E00..U+9FFF range
        result.append((f'Q{900000+code}',chr(0x4e00+code)+chr(0x6600+code)))
    return result


class EnrichmentTests(unittest.TestCase):
    def test_8_languages_exact(self):
        self.assertEqual(set(WD),{'zh','ja','ko','en','fr','de','es','ru'})

    def test_primary_alias_query_preserves_property_and_chinese_variants(self):
        q=_wd_query_expanded('Q7737',5000)
        self.assertIn('wdt:P103 wd:Q7737',q)
        self.assertIn('skos:altLabel',q)
        self.assertIn('"zh-hant"',q)
        self.assertNotIn('wdt:P1412',q)

    def test_p1412_fallback_excludes_people_with_other_native_languages(self):
        q=_wd_query_expanded('Q7737',5000,source_property='P1412')
        self.assertIn('wdt:P1412 wd:Q7737',q)
        self.assertIn('FILTER NOT EXISTS { ?person wdt:P103 ?nativeLanguage . }',q)
        with self.assertRaises(ValueError):
            _wd_query_expanded('Q7737',5000,source_property='P27')

    def test_russian_native_aliases_fill_without_auxiliary_label(self):
        def fake_fetch(lang,qid,limit,cache,*,query=None,cache_suffix=''):
            if lang=='ru':
                if 'P103-han-alias' in cache_suffix:
                    return rows('ru',74,120)
                return rows('ru',0,74)
            return rows(lang,0,160)
        with tempfile.TemporaryDirectory() as t:
            with patch('corpus.fetch_wikidata',side_effect=fake_fetch),patch('corpus.time.sleep'):
                records,meta=language_records(Path(t),[],5000,'required',20261117,80)
        self.assertEqual(meta['status'],'ready')
        self.assertEqual(meta['missing_languages'],[])
        self.assertGreaterEqual(meta['final_rows']['ru'],80)
        self.assertEqual(meta['auxiliary_training_rows'].get('ru',0),0)
        self.assertEqual(set(r['label'] for r in records),set(WD))

    def test_only_when_necessary_russian_spoken_language_tier_is_separated(self):
        def fake_fetch(lang,qid,limit,cache,*,query=None,cache_suffix=''):
            if lang=='ru':
                if 'P1412' in cache_suffix:
                    return rows('ru',74,90)
                return rows('ru',0,74)
            return rows(lang,0,170)
        with tempfile.TemporaryDirectory() as t:
            with patch('corpus.fetch_wikidata',side_effect=fake_fetch),patch('corpus.time.sleep'):
                records,meta=language_records(Path(t),[],5000,'required',20261117,80)
        self.assertEqual(meta['status'],'ready')
        self.assertEqual(meta['missing_languages'],[])
        weak=[r for r in records if r['label']=='ru' and r['tier']=='proxy_auxiliary']
        self.assertEqual(len(weak),90)
        self.assertTrue(all(r['source_property']=='P1412' for r in weak))
        self.assertEqual(meta['auxiliary_training_rows']['ru'],90)

    def test_insufficient_russian_still_rejected(self):
        def fake_fetch(lang,qid,limit,cache,*,query=None,cache_suffix=''):
            if lang=='ru':return rows(lang,0,74)
            return rows(lang,0,160)
        with tempfile.TemporaryDirectory() as t:
            with patch('corpus.fetch_wikidata',side_effect=fake_fetch),patch('corpus.time.sleep'):
                _,meta=language_records(Path(t),[],5000,'required',20261117,80)
        self.assertEqual(meta['status'],'partial')
        self.assertEqual(meta['missing_languages'],['ru'])


if __name__=='__main__':unittest.main()
