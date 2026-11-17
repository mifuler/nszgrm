'use strict';
const fs=require('fs'),api=require('../web/runtime.js');
const models=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).models;
const names=['布莱恩','张明','山田太郎','安娜','林肯'];
const result={};
for(const [task,bundle] of Object.entries(models)){
  result[task]={};
  for(const arch of ['cnn','bigru'])result[task][arch]=names.map(n=>api[arch](n,bundle.models[arch]));
}
process.stdout.write(JSON.stringify({names,results:result}));
