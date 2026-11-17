"""Persistent, batched bridge to the pinned OpenCC JS implementation."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

OPENCC_JS_VERSION = '1.4.2'
OPENCC_TARBALL_URL = 'https://registry.npmjs.org/opencc-js/-/opencc-js-1.4.2.tgz'
OPENCC_MODES = ({'from': 'jp', 'to': 'cn'}, {'from': 't', 'to': 'cn'})
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODULE = ROOT / 'vendor' / 'opencc' / 'node_modules' / 'opencc-js'
WORKER_SCRIPT = Path(__file__).with_name('opencc_worker.js')
BATCH_SIZE = 4096
CACHE_LIMIT = 200_000


class OpenCCBridge:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        env = os.environ.copy()
        module_root = Path(env.get('NSZGRM_OPENCC_MODULE_ROOT', str(DEFAULT_MODULE)))
        env['NSZGRM_OPENCC_MODULE_ROOT'] = str(module_root)
        node_path = env.get('NODE_PATH', '')
        env['NODE_PATH'] = str(module_root.parent) + (os.pathsep + node_path if node_path else '')
        self._process = subprocess.Popen(
            ['node', str(WORKER_SCRIPT)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding='utf-8',
            bufsize=1,
        )
        return self._process

    def convert_many(self, values: list[str] | tuple[str, ...]) -> list[str]:
        if not values:
            return []
        source = [str(value) for value in values]
        output: list[str | None] = [None] * len(source)
        missing: OrderedDict[str, None] = OrderedDict()
        with self._lock:
            for index, value in enumerate(source):
                cached = self._cache.get(value)
                if cached is not None:
                    self._cache.move_to_end(value)
                    output[index] = cached
                else:
                    missing.setdefault(value, None)
            if missing:
                keys = list(missing)
                process = self._start()
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError('OpenCC worker pipes were not created')
                converted: list[str] = []
                for start in range(0, len(keys), BATCH_SIZE):
                    batch = keys[start:start + BATCH_SIZE]
                    process.stdin.write(json.dumps(batch, ensure_ascii=False) + '\n')
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        code = process.poll()
                        raise RuntimeError(f'OpenCC worker exited unexpectedly (code={code})')
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError('OpenCC worker returned invalid JSON') from exc
                    if isinstance(response, dict) and response.get('error'):
                        raise RuntimeError('OpenCC conversion failed: ' + str(response['error']))
                    if not isinstance(response, list) or len(response) != len(batch):
                        raise RuntimeError('OpenCC worker returned an unexpected batch shape')
                    converted.extend(str(item) for item in response)
                results_by_key = dict(zip(keys, converted))
                for key, value in results_by_key.items():
                    self._cache[key] = value
                    self._cache.move_to_end(key)
                    if len(self._cache) > CACHE_LIMIT:
                        self._cache.popitem(last=False)
            for index, value in enumerate(source):
                if output[index] is None:
                    # Use this request's results, rather than the bounded LRU cache:
                    # long corpora can exceed CACHE_LIMIT within a single call.
                    output[index] = results_by_key[value]
        return [str(value) for value in output]

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()


_BRIDGE = OpenCCBridge()
atexit.register(_BRIDGE.close)


def convert_many(values: list[str] | tuple[str, ...]) -> list[str]:
    """Convert all accepted Han text, including Japanese shinjitai, to simplified."""
    return _BRIDGE.convert_many(values)


def convert_text(value: str) -> str:
    return _BRIDGE.convert_many([value])[0]
