#!/usr/bin/env python3
"""Check FP32 model JSON -> JS math against original PyTorch FP32 math.
Requires Node.js and PyTorch, does not retrain the model.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from architectures import TextCNN,BiGRU,probabilities


def main(path: Path):
    bundle=json.loads(path.read_text(encoding='utf8'))
    result=subprocess.run(['node',str(ROOT/'tests/parity_output.js'),str(path.resolve())],
                          capture_output=True,text=True,check=True,timeout=120)
    js=json.loads(result.stdout)
    for task, data in bundle['models'].items():
        for arch in ('cnn','bigru'):
            exported=data['models'][arch]
            hp=exported['hyperparameters']
            net=(TextCNN(len(exported['vocab']),len(exported['classes']),
                         emb=hp['emb'],channels=hp['channels'])
                 if arch=='cnn' else
                 BiGRU(len(exported['vocab']),len(exported['classes']),
                       emb=hp['emb'],hidden=hp['hidden']))
            net.load_state_dict({name:torch.tensor(np.asarray(item['values'],dtype=np.float32).reshape(item['shape']))
                                 for name,item in exported['state'].items()})
            py=probabilities(net,js['names'],exported['vocab'])
            np.testing.assert_allclose(py,js['results'][task][arch],rtol=1e-4,atol=2e-5,
                err_msg=f'{task}/{arch}: JS is not mathematically equivalent to PyTorch FP32')
            print('[PARITY PASS]',task,arch,'max delta',float(np.max(abs(py-np.asarray(js['results'][task][arch])))))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: python tests/verify_parity.py path/to/models_fp32.json')
    main(Path(sys.argv[1]))
