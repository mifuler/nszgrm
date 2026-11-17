import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from strong_data import norm_name,load_gold
from corpus import split3,checks
from train_all import BYTE_CAP,choose_weights,temperature
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

    def test_fusion_is_simplex(self):
        y=np.array([0,0,1,1,2,2])
        base=np.eye(3)[y]*0.8+0.2/3
        alt=np.roll(base,1,axis=1)
        w=choose_weights([base,base,alt,alt],y)
        self.assertEqual(len(w),4)
        self.assertTrue(all(x>=.029 for x in w))
        self.assertAlmostEqual(sum(w),1.0,places=6)

if __name__=='__main__':unittest.main()
