#!/usr/bin/env python3
"""Attach the independently measured quantization gate to the training reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

MAX_BYTES=15_000_000


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    out=args.output
    model=out/'models_int8.json'
    audit_path=out/'QUANT_AUDIT.json'
    loss_path=out/'LOSS_REPORT.json'
    report_path=out/'REPORT.json'
    md_path=out/'REPORT.md'
    for path in (model,audit_path,loss_path,report_path,md_path):
        if not path.is_file():raise SystemExit(f'missing quantization report input: {path}')
    size=model.stat().st_size
    if size>=MAX_BYTES:raise SystemExit(f'quantized model must be strictly below {MAX_BYTES:,} bytes')
    audit=json.loads(audit_path.read_text('utf-8'))
    loss=json.loads(loss_path.read_text('utf-8'))
    report=json.loads(report_path.read_text('utf-8'))
    if not loss.get('passed'):raise SystemExit('quantization accuracy/Macro-F1 gate did not pass')
    audit['acceptance_status']='PASSED_ON_TRAINING_HELDOUT_WEAK_LABEL_SAMPLE'
    audit['gate_report']='LOSS_REPORT.json'
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),'utf-8')
    report['quantization']={
        'model_file':'models_int8.json','model_raw_bytes':size,'max_raw_bytes':MAX_BYTES,
        'within_budget':size<MAX_BYTES,'loss_gate_passed':True,
        'evaluation':loss.get('evaluation'), 'thresholds':loss.get('thresholds'),
        'fallback_int16_models':audit.get('fallback_int16_models',[]),
        'tasks':loss.get('tasks',{}), 'weight_error':audit.get('weight_error',{}),
        'loss_report':'LOSS_REPORT.json','audit':'QUANT_AUDIT.json'}
    report.setdefault('artifacts',{})['quantized_model']=str(model)
    report['artifacts']['quantization_loss_report']=str(loss_path)
    report['artifacts']['quantization_audit']=str(audit_path)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    lines=md_path.read_text('utf-8').rstrip()+ '\n\n## INT8/INT16 量化验收\n\n'
    lines+=f'量化模型大小：**{size:,} / {MAX_BYTES:,} bytes**（严格小于 15 MB）。\n\n'
    lines+='评测使用训练流程保留的分层弱标签测试样本，不是人工金标准。Accuracy 与 Macro-F1 的相对下降均要求不超过 10%。\n\n'
    for task,result in loss.get('tasks',{}).items():
        lines+=f'### {task}\n\n'
        lines+='| 模型 | FP32 Accuracy | 量化 Accuracy | Accuracy 损耗 | FP32 Macro-F1 | 量化 Macro-F1 | Macro-F1 损耗 | 结果 |\n'
        lines+='|---|---:|---:|---:|---:|---:|---:|---|\n'
        for name,item in result.get('models',{}).items():
            lines+=(f"| {name} | {item['fp32']['accuracy']:.4f} | {item['quantized']['accuracy']:.4f} "
                    f"| {item['relative_accuracy_loss']:.2%} | {item['fp32']['macro_f1']:.4f} "
                    f"| {item['quantized']['macro_f1']:.4f} | {item['relative_macro_f1_loss']:.2%} "
                    f"| {'PASS' if item['passed'] else 'FAIL'} |\n")
        lines+='\n'
    lines+='INT16 回退模型：'+(', '.join(audit.get('fallback_int16_models',[])) or '无')+'。\n'
    md_path.write_text(lines,'utf-8')
    print(f'[quantization] accepted: {size:,}/{MAX_BYTES:,} bytes; all models <=10% relative loss')


if __name__=='__main__':main()
