#!/usr/bin/env python3
"""Install and verify the exact OpenCC JS tarball used by NSZGRM."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.4.2'
TARBALL = f'https://registry.npmjs.org/opencc-js/-/opencc-js-{VERSION}.tgz'
DESTINATION = ROOT / 'vendor' / 'opencc'
MODULE = DESTINATION / 'node_modules' / 'opencc-js'


def main() -> int:
    if not shutil.which('node') or not shutil.which('npm'):
        raise SystemExit('Node.js and npm are required to install opencc-js 1.4.2.')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            'npm', 'install', '--prefix', str(DESTINATION), '--no-save',
            '--ignore-scripts', '--no-audit', '--no-fund', TARBALL,
        ],
        check=True,
    )
    package_json = MODULE / 'package.json'
    if not package_json.is_file():
        raise SystemExit(f'OpenCC package metadata is missing: {package_json}')
    metadata = json.loads(package_json.read_text(encoding='utf-8'))
    if metadata.get('version') != VERSION:
        raise SystemExit(f'Expected opencc-js {VERSION}, installed {metadata.get("version")}')
    env = os.environ.copy()
    env['NSZGRM_OPENCC_MODULE_ROOT'] = str(MODULE)
    env['NODE_PATH'] = str(MODULE.parent) + (os.pathsep + env['NODE_PATH'] if env.get('NODE_PATH') else '')
    smoke = r"""
const api = require('opencc-js');
const OpenCC = api && typeof api.Converter === 'function' ? api : api.default;
if (!OpenCC || typeof OpenCC.Converter !== 'function') throw new Error('OpenCC.Converter unavailable');
const fromJapanese = OpenCC.Converter({from:'jp',to:'cn'});
const toSimplified = OpenCC.Converter({from:'t',to:'cn'});
const convert = (value) => toSimplified(fromJapanese(value));
for (const [input, expected] of [['陳小明','陈小明'],['臺灣','台湾'],['國際山田','国际山田'],['澤田','泽田']]) {
  const actual = convert(input);
  if (actual !== expected) throw new Error(`${input}: expected ${expected}, got ${actual}`);
}
console.log('opencc-js 1.4.2: API and t2s smoke checks passed');
"""
    subprocess.run(['node', '-e', smoke], cwd=ROOT, env=env, check=True)
    print(f'Installed and verified opencc-js {VERSION} at {MODULE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
