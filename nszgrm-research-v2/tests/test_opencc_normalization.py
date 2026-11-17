import tempfile
import unittest
from pathlib import Path

from opencc_bridge import OPENCC_JS_VERSION, OpenCCBridge
from unittest.mock import patch
from strong_data import build_weak, group_norm, norm_name, source_names
from corpus import FILES, _wikidata_json_rows, build_writing


class OpenCCNormalizationTests(unittest.TestCase):
    def test_pinned_version_and_traditional_to_simplified(self):
        self.assertEqual(OPENCC_JS_VERSION, '1.4.2')
        self.assertEqual(norm_name('陳小明'), '陈小明')
        self.assertEqual(norm_name('臺灣'), '台湾')
        self.assertEqual(norm_name('國際山田'), '国际山田')
        self.assertEqual(norm_name('澤田'), '泽田')

    def test_japanese_kanji_goes_through_the_same_converter(self):
        self.assertEqual(norm_name('山田太郎'), '山田太郎')
        self.assertEqual(norm_name('澤田'), '泽田')
        self.assertEqual(norm_name('國際山田'), '国际山田')
        self.assertEqual(norm_name('澤田'), '泽田')

    def test_collision_statistics_are_computed_after_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'names.txt'
            path.write_text('陳小明\n陈小明\n陳小明\n國際山田\n山田太郎\n', encoding='utf-8')
            names, stats = source_names(path)
        self.assertEqual(names, {'陈小明', '国际山田', '山田太郎'})
        self.assertEqual(stats['lines'], 5)
        self.assertEqual(stats['unique_raw_valid'], 4)
        self.assertEqual(stats['unique_valid'], 3)
        self.assertEqual(stats['opencc_converted_rows'], 3)
        self.assertEqual(stats['opencc_converted_unique_names'], 2)
        self.assertEqual(stats['simplified_name_collisions'], 1)
        self.assertEqual(stats['simplified_name_collision_examples'], [
            {'simplified': '陈小明', 'sources': ['陈小明', '陳小明']}
        ])

    def test_han_characters_in_group_ids_are_normalized(self):
        self.assertEqual(group_norm('Group-國際-42'), 'group-国际-42')

    def test_wikidata_name_labels_are_normalized_before_language_records(self):
        payload = {'results': {'bindings': [
            {'person': {'value': 'https://www.wikidata.org/entity/Q123'},
             'name': {'value': '陳小明'}}
        ]}}
        self.assertEqual(_wikidata_json_rows(payload), [('Q123', '陈小明')])

    def test_weak_training_reports_chinese_japanese_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in FILES.values():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / FILES['zh_native']).write_text(
                '\n'.join(['陳小明'] + ['中' + chr(0x4e00 + i) for i in range(105)]) + '\n',
                encoding='utf-8')
            (root / FILES['ja_kanji_proxy']).write_text(
                '\n'.join(['陈小明'] + ['日' + chr(0x4e00 + i) for i in range(105)]) + '\n',
                encoding='utf-8')
            (root / FILES['foreign_phonetic_proxy']).write_text(
                '\n'.join(['外' + chr(0x4e00 + i) for i in range(105)]) + '\n',
                encoding='utf-8')
            rows, metadata = build_weak(root, [], max_per_class=0, seed=20261117)
        self.assertEqual(metadata['chinese_japanese_same_name_overlap_after_normalization_count'], 1)
        self.assertEqual(metadata['opencc_chinese_japanese_collision_count'], 1)
        self.assertEqual(metadata['opencc_chinese_japanese_collision_examples'][0]['simplified'], '陈小明')
        self.assertEqual(metadata['cross_label_spellings_excluded'], 1)
        self.assertFalse(any(row['name'] == '陈小明' for row in rows))

    def test_chinese_japanese_collision_is_counted_after_opencc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in FILES.values():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
            zh_path = root / FILES['zh_native']
            ja_path = root / FILES['ja_kanji_proxy']
            foreign_path = root / FILES['foreign_phonetic_proxy']
            ancient_path = root / FILES['zh_ancient_extra']
            zh_names = ['陳小明'] + ['中' + chr(0x4e00 + i) for i in range(105)]
            ja_names = ['陈小明'] + ['日' + chr(0x4e00 + i) for i in range(105)]
            foreign_names = ['外' + chr(0x4e00 + i) for i in range(105)]
            zh_path.write_text('\n'.join(zh_names) + '\n', encoding='utf-8')
            ja_path.write_text('\n'.join(ja_names) + '\n', encoding='utf-8')
            foreign_path.write_text('\n'.join(foreign_names) + '\n', encoding='utf-8')
            ancient_path.write_text('古' + chr(0x4e00) + '\n', encoding='utf-8')
            rows, metadata = build_writing(root, [], cap=0, seed=20261117)
        self.assertEqual(metadata['chinese_japanese_same_name_overlap_after_normalization_count'], 1)
        self.assertEqual(metadata['opencc_chinese_japanese_collision_count'], 1)
        self.assertEqual(metadata['opencc_chinese_japanese_collision_examples'][0]['simplified'], '陈小明')
        self.assertEqual(metadata['ambiguous_spellings_excluded'], 1)
        self.assertFalse(any(row['name'] == '陈小明' for row in rows))

    def test_cache_eviction_does_not_corrupt_a_large_batch(self):
        # A real corpus can contain more unique names than the LRU cache limit.
        bridge = OpenCCBridge()
        try:
            with patch('opencc_bridge.CACHE_LIMIT', 3):
                input_names = ['陳小明', '國際山田', '山田太郎', '林肯',
                               '張三', '李四', '王五', '臺灣', '澤田']
                results = bridge.convert_many(input_names)
            self.assertEqual(len(results), len(input_names))
            self.assertEqual(results[0], '陈小明')
            self.assertEqual(results[-1], '泽田')
        finally:
            bridge.close()

    def test_non_han_or_mixed_name_is_rejected_as_before(self):
        self.assertEqual(norm_name('ABC'), '')
        self.assertEqual(norm_name('A陳'), '')


if __name__ == '__main__':
    unittest.main()
