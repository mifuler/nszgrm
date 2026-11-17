import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quantize_existing import qmatrix,dequant


class QuantizationTests(unittest.TestCase):
    def test_int8_per_row_roundtrip(self):
        values=np.random.default_rng(20261117).normal(0,0.15,size=(8,128)).astype(np.float32)
        packed=qmatrix(values,bits=8)
        restored=dequant(packed)
        self.assertEqual(packed['quant_format'],'sym_int8_per_row_v1')
        self.assertEqual(restored.shape,values.shape)
        self.assertLess(float(np.max(np.abs(restored-values))),0.01)

    def test_int16_per_row_has_lower_or_equal_error(self):
        values=np.random.default_rng(17).normal(0,1,size=(4,64)).astype(np.float32)
        q8=dequant(qmatrix(values,bits=8))
        q16=dequant(qmatrix(values,bits=16))
        self.assertLessEqual(float(np.max(np.abs(q16-values))),float(np.max(np.abs(q8-values))))

    def test_rejects_non_finite_weights(self):
        with self.assertRaises(ValueError):qmatrix(np.array([[np.nan,1.0],[0.0,1.0]]))


if __name__=='__main__':unittest.main()
