'use strict';
// Official post-training INT8 relative loss gate, not a claim of external validity.
const fs=require('fs');
const original=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const quant=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const evalset=JSON.parse(fs.readFileSync(process.argv[4],'utf8'));
const output=process.argv[5]||'LOSS_REPORT.json';
const limit=0.10;
const old=require('./runtime_quant.js');
const now=require('./runtime_quant.js');
const keys=['linear','complement_nb','cnn','bigru','fusion'];
function evaluate(expected,predicted,classes){
 let correct=0;
 for(let i=0;i<expected.length;i++)if(expected[i]===predicted[i])correct++;
 const fscores=[];
 for(const cls of classes){
  let tp=0,fp=0,fn=0;
  for(let i=0;i<expected.length;i++){
   if(expected[i]===cls && predicted[i]===cls)tp++;
   else if(expected[i]!==cls && predicted[i]===cls)fp++;
   else if(expected[i]===cls && predicted[i]!==cls)fn++;
  }
  fscores.push((2*tp+fp+fn)?2*tp/(2*tp+fp+fn):0);
 }
 return {accuracy:correct/expected.length,macro_f1:fscores.reduce((a,b)=>a+b,0)/fscores.length};
}
const report={model_source:process.argv[2],quantized_model:process.argv[3],evaluation:'stratified sample from the training run held-out weak-label test split; not a human gold set',thresholds:{accuracy_relative_loss:limit,macro_f1_relative_loss:limit},tasks:{},passed:true};
for(const [task,rows] of Object.entries(evalset.tasks)){
 if(!original.models[task]||!quant.models[task])throw Error('Missing task '+task);
 const classes=original.models[task].classes;
 let expected=[],p0=Object.fromEntries(keys.map(k=>[k,[]])),p1=Object.fromEntries(keys.map(k=>[k,[]]));
 let drift=Object.fromEntries(keys.map(k=>[k,{top1_flips:0,max_abs_probability_delta:0,mean_abs_probability_delta:0}]));
 for(const {name,label} of rows){
  if(!old.normalize(name)||!classes.includes(label))throw Error('Invalid test label '+name+': '+label);
  const a=old.predictTask(name,original.models[task]);
  const b=now.predictTask(name,quant.models[task]);
  expected.push(label);
  for(const key of keys){
   const x=key==='fusion'?a.probabilities:a.model_scores[key].probabilities;
   const y=key==='fusion'?b.probabilities:b.model_scores[key].probabilities;
   const maxi=z=>z.indexOf(Math.max(...z));
   p0[key].push(classes[maxi(x)]);p1[key].push(classes[maxi(y)]);
   const stats=drift[key];if(maxi(x)!==maxi(y))stats.top1_flips++;
   for(let j=0;j<classes.length;j++){
    const d=Math.abs(x[j]-y[j]);
    stats.max_abs_probability_delta=Math.max(stats.max_abs_probability_delta,d);
    stats.mean_abs_probability_delta+=d;
   }
  }
 }
 if(expected.length<20)throw Error('Insufficient evaluation cases for task '+task);
 report.tasks[task]={n:expected.length,models:{}};
 for(const key of keys){
  const fp32=evaluate(expected,p0[key],classes),quantized=evaluate(expected,p1[key],classes);
  const loss=(metric)=>Math.max(0,fp32[metric]-quantized[metric])/Math.max(fp32[metric],1e-12);
  const relative_accuracy_loss=loss('accuracy');
  const relative_macro_f1_loss=loss('macro_f1');
  const pass=relative_accuracy_loss<=limit && relative_macro_f1_loss<=limit;
  drift[key].prediction_flip_rate=drift[key].top1_flips/expected.length;
  drift[key].mean_abs_probability_delta/=expected.length*classes.length;
  report.tasks[task].models[key]={fp32,quantized,relative_accuracy_loss,relative_macro_f1_loss,passed:pass,...drift[key]};
  if(!pass)report.passed=false;
 }
}
for(const task of Object.keys(original.models)){
 if(!report.tasks[task]){report.passed=false;report.tasks[task]={error:'Missing representative weak-label test evaluation'};}
}
fs.writeFileSync(output,JSON.stringify(report,null,2));
for(const [task,v] of Object.entries(report.tasks)){
 console.log('[quant gate]',task,v.n||0);
 for(const [m,x] of Object.entries(v.models||{}))console.log(m,'FP32 Acc',x.fp32.accuracy.toFixed(4),'quantized Acc',x.quantized.accuracy.toFixed(4),'FP32 MacroF1',x.fp32.macro_f1.toFixed(4),'quantized MacroF1',x.quantized.macro_f1.toFixed(4),'pass',x.passed);
}
console.log('[quant gate] passed:',report.passed);
if(!report.passed)process.exitCode=2;
