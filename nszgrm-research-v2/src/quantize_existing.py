#!/usr/bin/env python3
"""Post-training per-row INT8/INT16 conversion of an NSZGRM v3 FP32 ensemble.

The FP32 source remains untouched. The quantized bundle keeps the same task/model
layout so the independent HTML front end can use a matching quantized runtime.
"""
from __future__ import annotations
import argparse
import base64
import copy
import datetime
import json
from pathlib import Path
import numpy as np

MAX_BYTES = 15_000_000


def qmatrix(values, shape=None, bits=8):
    a=np.asarray(values,dtype=np.float32)
    if shape is not None:a=a.reshape(shape)
    if bits not in (8,16):raise ValueError('Quant precision must be 8 or 16 bits')
    if a.ndim<2:raise ValueError('Only 2-D or higher tensors may be row-quantized')
    if not np.isfinite(a).all():raise ValueError('Non-finite model weight encountered')
    rows=a.reshape(a.shape[0],-1)
    qmax=127 if bits==8 else 32767
    scales=np.where(np.max(np.abs(rows),axis=1)>0,
                    np.max(np.abs(rows),axis=1)/np.float32(qmax),np.float32(1)).astype(np.float32)
    qdtype=np.int8 if bits==8 else np.dtype('<i2')
    quant=np.clip(np.rint(rows/scales[:,None]),-qmax,qmax).astype(qdtype)
    return {'quant_format':('sym_int8_per_row_v1' if bits==8 else 'sym_int16_per_row_v1'),
            'shape':list(a.shape),'scales':[float(v) for v in scales],
            'data_b64':base64.b64encode(quant.tobytes(order='C')).decode('ascii')}


def dequant(q):
    if not isinstance(q,dict) or q.get('quant_format') not in ('sym_int8_per_row_v1','sym_int16_per_row_v1'):
        raise ValueError('Unexpected quantization format')
    shape=tuple(q['shape'])
    raw=base64.b64decode(q['data_b64'],validate=True)
    bits16=q['quant_format']=='sym_int16_per_row_v1'
    dtype=np.dtype('<i2') if bits16 else np.int8
    expected=int(np.prod(shape))*(2 if bits16 else 1)
    if len(raw)!=expected:raise ValueError('Quantized tensor byte count mismatch')
    a=np.frombuffer(raw,dtype=dtype).reshape(shape[0],-1).astype(np.float32)
    scales=np.asarray(q['scales'],dtype=np.float32).reshape(-1,1)
    if len(scales)!=shape[0]:raise ValueError('Quantized row scale count mismatch')
    return (a*scales).reshape(shape)


def quantize_bundle(original,int16_models=None):
    int16_models=set(int16_models or [])
    bundle=copy.deepcopy(original)
    if bundle.get('format_version')!=3:raise ValueError('Expected NSZGRM FP32 model format_version=3')
    summary={}
    for task,block in bundle['models'].items():
        summary[task]={}
        for name,model in block['models'].items():
            bits=16 if f'{task}/{name}' in int16_models else 8
            fields=['coef'] if name=='linear' else (['feature_log_prob'] if name=='complement_nb' else [])
            for key in fields:
                original_weights=np.asarray(model[key],dtype=np.float32)
                packed=qmatrix(original_weights,bits=bits)
                model[key]=packed
                restored=dequant(packed)
                err=np.abs(restored-original_weights)
                summary[task][name+'/'+key]={'bits':bits,'count':int(original_weights.size),
                    'max_abs_weight_error':float(err.max()),'mean_abs_weight_error':float(err.mean())}
            if 'state' in model:
                for key,item in model['state'].items():
                    if len(item['shape'])<2:continue  # biases remain FP32
                    original_weights=np.asarray(item['values'],dtype=np.float32).reshape(item['shape'])
                    packed=qmatrix(original_weights,bits=bits)
                    model['state'][key]=packed
                    restored=dequant(packed);err=np.abs(restored-original_weights)
                    summary[task][name+'/'+key]={'bits':bits,'count':int(original_weights.size),
                        'max_abs_weight_error':float(err.max()),'mean_abs_weight_error':float(err.mean())}
    bundle['format_version']=4
    bundle['precision']='INT8/INT16 symmetric per-row weights; FP32 biases, IDF, temperatures and fusion weights'
    bundle['quantization']={'algorithm':'sym_int_per_row_v1',
        'int16_fallback_models':sorted(int16_models),
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_format_version':3,
        'note':'Per-row post-training quantization. Acceptance requires the held-out weak-label loss gate.'}
    return bundle,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--int16-models',default='',help='Comma-separated task/model pairs selectively stored as INT16')
    p.add_argument('--max-bytes',type=int,default=MAX_BYTES)
    a=p.parse_args()
    original=json.loads(a.source.read_text('utf-8'))
    fallbacks={x.strip() for x in a.int16_models.split(',') if x.strip()}
    valid={f'{task}/{name}' for task,block in original['models'].items() for name in block['models']}
    if not fallbacks.issubset(valid):raise ValueError('Invalid INT16 fallback model(s): '+str(fallbacks-valid))
    quant,summary=quantize_bundle(original,fallbacks)
    a.out.mkdir(parents=True,exist_ok=True)
    model_path=a.out/'models_int8.json'
    model_path.write_text(json.dumps(quant,ensure_ascii=False,separators=(',',':'),allow_nan=False),'utf-8')
    size=model_path.stat().st_size
    audit={'source':str(a.source),'source_size_bytes':a.source.stat().st_size,
        'quantized_size_bytes':size,'max_bytes':a.max_bytes,
        'compression_ratio':a.source.stat().st_size/max(size,1),
        'weight_error':summary,'fallback_int16_models':sorted(fallbacks),
        'acceptance_status':'PENDING_HELDOUT_LOSS_GATE',
        'thresholds':{'max_relative_accuracy_loss':0.10,'max_relative_macro_f1_loss':0.10}}
    (a.out/'QUANT_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),'utf-8')
    print(f'[quantize] FP32={a.source.stat().st_size:,} bytes; quantized={size:,}/{a.max_bytes:,} bytes; ratio={audit["compression_ratio"]:.2f}x')
    if size>=a.max_bytes:raise SystemExit(f'Quantized model must be strictly below {a.max_bytes:,} bytes')


if __name__=='__main__':main()
