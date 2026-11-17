import json
import re
import shutil
import joblib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pipeline as p

class PipelineTests(unittest.TestCase):
    def test_normalization_and_rejection(self):
        self.assertEqual(p.normalize(' 李白\tM '), '李白')
        self.assertEqual(p.normalize('李白,男'), '李白')
        self.assertIsNone(p.normalize('张'))
        self.assertIsNone(p.normalize('王·小明'))
        self.assertIsNone(p.normalize('Paul'))
        self.assertEqual(p.normalize(' 布莱恩'), '布莱恩')

    def test_split_determinism_and_no_overlap(self):
        a = [(lab, name) for lab, name in p.make_demo()]
        x = p.stable_class_split(a, 20261117)
        y = p.stable_class_split(a, 20261117)
        self.assertEqual(x, y)
        s = {k: set(name for _, name in v) for k,v in x.items()}
        self.assertFalse(s['train'] & s['val'])
        self.assertFalse(s['train'] & s['test'])
        self.assertFalse(s['val'] & s['test'])

    def test_binary_quantized_export_roundtrip(self):
        # A separate 2-class test covers binary sigmoid calibrator branches.
        rows = []
        for lab, base in [('aa','林'),('bb','布')]:
            for i in range(35):
                rows.append((lab, base + chr(0x4e00 + i)))
        with tempfile.TemporaryDirectory() as td:
            m, quant = p.save_model('binary', rows, Path(td), 17, 1000, True, {})
            self.assertIn('quantized_test', m)
            self.assertIn('quantized_prediction_flip_rate', m)
            self.assertTrue(0 <= m['quantized_prediction_flip_rate'] <= 1)
            self.assertEqual(quant['coef_rows'], 1)
            self.assertIn(quant['weight_dtype'], ['int8','int16'])
            self.assertTrue((Path(td) / 'binary.joblib').exists())

    @unittest.skipUnless(shutil.which('node'), 'Node.js is optional for local smoke test')
    def test_browser_quantized_inference_matches_python(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            m, exported = p.save_model('writing_type', p.make_demo(), out, 20261117, 1000, True, {})
            html = (ROOT / 'web' / 'index.html').read_text('utf-8')
            scripts = re.findall(r'<script(?:\s[^>]*)?>([\s\S]*?)</script>', html)
            self.assertTrue(scripts[-1].strip())
            ui_js = scripts[-1].replace('render();\n})();', 'window.__test_infer=infer;\n})();')
            self.assertIn('window.__test_infer=infer;', ui_js)
            node_script = r"""
const fs=require('fs'),vm=require('vm');
const elements={};
const ctx={window:{}, atob, document:{
  getElementById:(id)=>(elements[id]??=( {value:'',textContent:'',innerHTML:'',addEventListener:()=>{}} )),
  querySelectorAll:()=>[]
}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8')+fs.readFileSync(process.argv[2], 'utf8'), ctx);
const result=ctx.window.__test_infer('布莱恩');
process.stdout.write(JSON.stringify(result.writing.items));
"""
            jsfile = out / 'browser-test.js'
            jsfile.write_text(ui_js, encoding='utf-8')
            loaded = joblib.load(out / 'writing_type.joblib')
            x = loaded['vectorizer'].transform(['布莱恩'])
            for bits in (8, 16):
                reference = p.export_quantized(loaded['classifier'], loaded['calibrator'],
                                               loaded['vectorizer'], 'writing_type', bits=bits)
                exported_for_js = dict(reference)
                exported_for_js.pop('_quantized_weights')
                (out / 'models.js').write_text('window.CHINESE_NAME_TRAINED_MODELS='+json.dumps({'writing_type': exported_for_js},ensure_ascii=False)+';',encoding='utf-8')
                cp = subprocess.run(['node','-e',node_script,str(out/'models.js'),str(jsfile)],capture_output=True,text=True,check=True)
                got = {d['label']:d['prob'] for d in json.loads(cp.stdout)}
                expected = p.quantified_predictions(reference, x)[0]
                for lab,prob in zip(reference['classes'],expected):
                    self.assertAlmostEqual(got[lab], float(prob), places=6)

    def test_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'model'
            cmd = [sys.executable, str(ROOT / 'src' / 'pipeline.py'), '--demo',
                   '--output', str(dest), '--wikidata', 'off', '--max-features', '800']
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            self.assertTrue((dest / 'writing_type.joblib').is_file())
            self.assertTrue((dest / 'writing_type.json').is_file())
            self.assertTrue((dest / 'browser' / 'models.js').is_file())
            self.assertTrue((dest / 'browser' / 'index.html').is_file())
            m = json.loads((dest / 'writing_type_metrics.json').read_text('utf-8'))
            self.assertTrue(m['test_no_name_overlap'])
            self.assertEqual(m['training_role'], 'writing_type')
            self.assertTrue(0 <= m['accuracy'] <= 1)
            self.assertIn('DEMO', (dest / 'REPORT.md').read_text('utf-8'))

if __name__ == '__main__':
    unittest.main()
