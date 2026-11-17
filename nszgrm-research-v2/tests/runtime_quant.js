/* NSZGRM full-precision offline inference. Every model always contributes. */
(function(root) {
'use strict';
const minuscule=1e-30;

// Symmetric per-row INT8 decoder, cached for the lifetime of the model.
// float32 scales are applied per output row. No network requests or WASM.
function decodeWeights(t){
 if(Array.isArray(t))return t.map(row=>Float32Array.from(row));
 if(!['sym_int8_per_row_v1','sym_int16_per_row_v1'].includes(t.quant_format))throw Error('Unsupported quant format');
 if(t._decoded)return t._decoded;
 const n=t.shape.reduce((a,b)=>a*b,1), width=n/t.shape[0];
 const raw=atob(t.data_b64),int16=t.quant_format==='sym_int16_per_row_v1';
 if(raw.length!==n*(int16?2:1))throw Error('Quantized tensor byte count mismatch');
 const rows=[];
 for(let i=0;i<t.shape[0];i++){
  const row=new Float32Array(width),scale=t.scales[i],off=i*width;
  for(let j=0;j<width;j++){
   let x;if(int16){const v=2*(off+j);x=raw.charCodeAt(v)|(raw.charCodeAt(v+1)<<8);x=x>32767?x-65536:x;}
   else{x=raw.charCodeAt(off+j);x=x>127?x-256:x;}
   row[j]=Math.fround(x*scale);
  }
  rows.push(row);
 }
 Object.defineProperty(t,'_decoded',{value:rows,enumerable:false});
 return rows;
}

function softmax(a){const max=Math.max(...a);const e=a.map(x=>Math.exp(x-max));const s=e.reduce((p,c)=>p+c,0);return e.map(v=>v/s);}
function sigmoid(x){return x>=0 ? 1/(1+Math.exp(-x)) : Math.exp(x)/(1+Math.exp(x));}
function calibrate(p,t){return softmax(p.map(v=>Math.log(Math.max(v,minuscule))/t));}
function normalize(name){const s=String(name||'').normalize('NFKC').trim();return /^[\u3400-\u9fff]{2,8}$/u.test(s)?s:null;}
function tfidf(name,m){
 const terms=m.tfidf.vocabulary;const idf=m.tfidf.idf;
 const counts=new Map();const chars=[...name];
 for(let n=1;n<=3;n++)for(let i=0;i<=chars.length-n;i++){
  const key=chars.slice(i,i+n).join('');const idx=terms[key];
  if(idx!==undefined)counts.set(idx,(counts.get(idx)||0)+1);
 }
 const x=new Float64Array(idf.length);let norm=0;
 for(const [idx,cnt] of counts){const v=cnt*idf[idx];x[idx]=v;norm+=v*v;}
 if(norm>0){norm=Math.sqrt(norm);for(const idx of counts.keys())x[idx]/=norm;}
 return x;
}
function linear(name,m,cache){
 const x=cache||tfidf(name,m);const logits=[];
 const coef=decodeWeights(m.coef);
 for(let c=0;c<coef.length;c++){
  const w=coef[c];let z=m.intercept[c];
  for(let j=0;j<x.length;j++)if(x[j])z+=x[j]*w[j];
  logits.push(z);
 }
 if(m.mode==='sigmoid_binary'){const q=sigmoid(logits[0]);return [1-q,q];}
 const p=logits.map(sigmoid);const sum=p.reduce((a,b)=>a+b,0);
 return p.map(v=>v/sum);
}
function complement(name,m,cache){
 const x=cache||tfidf(name,m);const logits=[];
 for(const row of decodeWeights(m.feature_log_prob)){let z=0;
  for(let j=0;j<x.length;j++)if(x[j])z+=row[j]*x[j];
  logits.push(z);
 }
 return softmax(logits);
}
function modelState(m){
 if(m._unpacked)return m._unpacked;
 const o={};for(const [key,value] of Object.entries(m.state)){
  if(['sym_int8_per_row_v1','sym_int16_per_row_v1'].includes(value.quant_format)){
   const rows=decodeWeights(value),w=rows[0].length,buf=new Float32Array(rows.length*w);
   rows.forEach((row,i)=>buf.set(row,i*w));o[key]=buf;
  }else o[key]=Float32Array.from(value.values);
 }
 Object.defineProperty(m,'_unpacked',{value:o,enumerable:false});return o;
}
function neuralTokens(name,m){
 const c=[...name].slice(0,m.max_len),v=m.vocab;
 return c.map(ch=>v[ch]===undefined?1:v[ch]);
}
function vecEmbedding(tokens,state,m){
 const w=state['embedding.weight'];const emb=m.hyperparameters.emb;
 return tokens.map(idx=>w.subarray(idx*emb,(idx+1)*emb));
}
function cnn(name,m){
 const s=modelState(m),seq=neuralTokens(name,m),embedding=vecEmbedding(seq,s,m),E=m.hyperparameters.emb,C=m.hyperparameters.channels;
 const z=[];
 for(let i=0;i<3;i++){
  const kernel=i+2,w=s['convs.'+i+'.weight'],b=s['convs.'+i+'.bias'];
  const activations=new Float64Array(C);activations.fill(-Infinity);
  const nPos=Math.max(1,seq.length-kernel+1);
  for(let p=0;p<nPos;p++)for(let c=0;c<C;c++){
   let a=b[c],offset=c*E*kernel;
   for(let j=0;j<E;j++)for(let k=0;k<kernel;k++){
    if(p+k<embedding.length)a+=w[offset+j*kernel+k]*embedding[p+k][j];
   }
   if(a<0)a=0;
   if(a>activations[c])activations[c]=a;
  }
  for(let c=0;c<C;c++)z.push(activations[c]);
 }
 const head=s['head.weight'],bias=s['head.bias'],logits=[];
 for(let c=0;c<m.classes.length;c++){
  let sum=bias[c];for(let j=0;j<z.length;j++)sum+=head[c*z.length+j]*z[j];
  logits.push(sum);
 }
 return softmax(logits);
}
function rnnStep(vec,prev,wih,whh,bih,bhh,H){
 const E=vec.length;const gates=new Float64Array(H*6);
 for(let i=0;i<H*3;i++){
  let a=bih[i],b=bhh[i];
  for(let e=0;e<E;e++)a+=wih[i*E+e]*vec[e];
  for(let h=0;h<H;h++)b+=whh[i*H+h]*prev[h];
  gates[i]=a;
  // scratch second summand to respect GRU reset-gate formula
  gates[i+H*3]=b;
 }
 const out=new Float64Array(H);
 for(let h=0;h<H;h++){
  const r=sigmoid(gates[h]+gates[h+H*3]);
  const z=sigmoid(gates[H+h]+gates[H+h+H*3]);
  const n=Math.tanh(gates[2*H+h]+r*gates[2*H+h+H*3]);
  out[h]=(1-z)*n+z*prev[h];
 }
 return out;
}
function bigru(name,m){
 const state=modelState(m),tokens=neuralTokens(name,m),emb=vecEmbedding(tokens,state,m),H=m.hyperparameters.hidden;
 const paths=[];
 for(const suffix of ['', '_reverse']){
  let h=new Float64Array(H);
  const wih=state['rnn.weight_ih_l0'+suffix],whh=state['rnn.weight_hh_l0'+suffix],
        bih=state['rnn.bias_ih_l0'+suffix],bhh=state['rnn.bias_hh_l0'+suffix];
  const timesteps=suffix?[...emb].reverse():emb;
  for(const v of timesteps)h=rnnStep(v,h,wih,whh,bih,bhh,H);
  paths.push(...h);
 }
 const weights=state['head.weight'],b=state['head.bias'];const logits=[];
 for(let c=0;c<m.classes.length;c++){
  let z=b[c];for(let h=0;h<2*H;h++)z+=weights[c*2*H+h]*paths[h];
  logits.push(z);
 }
 return softmax(logits);
}
function predictTask(name,task){
 const weights=task.fusion_weights,models=task.models;
 const cache=tfidf(name,models.linear);
 const results={};const final=new Array(task.classes.length).fill(0);
 for(const key of ['linear','complement_nb','cnn','bigru']){
  const model=models[key];if(!model)throw Error('Required model missing: '+key);
  const p=(key==='linear'?linear(name,model,cache):
          key==='complement_nb'?complement(name,model,cache):
          key==='cnn'?cnn(name,model):bigru(name,model));
  const calibrated=calibrate(p,model.temperature);
  const weight=weights[key];
  results[key]={probabilities:calibrated,weight,top:task.classes[calibrated.indexOf(Math.max(...calibrated))]};
  for(let i=0;i<final.length;i++)final[i]+=weight*calibrated[i];
 }
 return {classes:task.classes,model_scores:results,probabilities:final,
   label:task.classes[final.indexOf(Math.max(...final))],
   confidence:Math.max(...final)};
}
function predict(name,bundle){
 const normalized=normalize(name);
 if(!normalized)return {error:'请输入 2–8 个连续汉字；标点、间隔号或非中文姓名请先拆分。'};
 const writing=predictTask(normalized,bundle.models.writing);
 const pidx=writing.classes.indexOf('foreign_phonetic_proxy');
 const score=pidx<0?null:100*writing.probabilities[pidx];
 const result={name:normalized,phonetic_index:score,writing_type:writing,
               note:'分数为弱监督模型融合结果，未经代表性人群校准；不是词源证据。'};
 if(bundle.models.language){result.language_proxy=predictTask(normalized,bundle.models.language);}
 return result;
}
const api={predict,predictTask,normalize,tfidf,linear,complement,cnn,bigru,calibrate};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
root.NSZGRM=api;
})(typeof globalThis!=='undefined'?globalThis:this);
