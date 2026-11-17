'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert');
const api=require('../web/runtime.js');
const file=process.argv[2];
if(!file)throw Error('usage: node tests/verify_browser.js path/to/models_fp32.json');
const bundle=JSON.parse(fs.readFileSync(file,'utf8'));
assert(bundle.models.writing);
for(const [taskName, task] of Object.entries(bundle.models)){
  assert.deepStrictEqual(Object.keys(task.models).sort(), ['bigru','cnn','complement_nb','linear']);
  const weights=Object.values(task.fusion_weights);
  assert(Math.abs(weights.reduce((a,b)=>a+b,0)-1)<1e-7);
  for(const name of ['张伟','布莱恩','山田太郎','林肯','安娜','金正恩']){
    const result=api.predictTask(name,task);
    assert.strictEqual(result.probabilities.length,task.classes.length);
    assert(Math.abs(result.probabilities.reduce((a,b)=>a+b,0)-1)<1e-6);
    for(const [modelName, info] of Object.entries(result.model_scores)){
      assert(Math.abs(info.probabilities.reduce((a,b)=>a+b,0)-1)<1e-6);
      assert(info.probabilities.every(v=>Number.isFinite(v)&&v>=0&&v<=1),modelName+' probabilities');
    }
  }
  console.log('[JS PASS]',taskName,task.classes.join(','));
}
console.log('[JS PASS] all FP32 models independently evaluated; fusion probabilities normalized.');
