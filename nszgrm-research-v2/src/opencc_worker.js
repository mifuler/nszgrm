'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const moduleRoot = process.env.NSZGRM_OPENCC_MODULE_ROOT ||
  path.join(__dirname, '..', 'vendor', 'opencc', 'node_modules', 'opencc-js');
const packagePath = path.join(moduleRoot, 'package.json');
const packageInfo = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
if (packageInfo.version !== '1.4.2') {
  throw new Error(`Expected opencc-js 1.4.2, found ${packageInfo.version || 'unknown'}`);
}
const loaded = require('opencc-js');
const OpenCC = loaded && typeof loaded.Converter === 'function' ? loaded : loaded.default;
if (!OpenCC || typeof OpenCC.Converter !== 'function') {
  throw new Error('opencc-js does not expose OpenCC.Converter');
}
const convertJapanese = OpenCC.Converter({ from: 'jp', to: 'cn' });
const convertTraditional = OpenCC.Converter({ from: 't', to: 'cn' });
if (typeof convertJapanese !== 'function' || typeof convertTraditional !== 'function') {
  throw new Error('OpenCC.Converter did not return conversion functions');
}
function toSimplified(text) {
  // OpenCC's Japanese dictionary handles shinjitai, then t2s handles any
  // remaining traditional characters. Same sequence for every name source.
  return convertTraditional(convertJapanese(text));
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', (line) => {
  try {
    const values = JSON.parse(line);
    if (!Array.isArray(values) || values.some((value) => typeof value !== 'string')) {
      throw new TypeError('Expected a JSON array of strings');
    }
    const result = values.map(toSimplified);
    process.stdout.write(JSON.stringify(result) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: String(error && error.message ? error.message : error) }) + '\n');
  }
});
