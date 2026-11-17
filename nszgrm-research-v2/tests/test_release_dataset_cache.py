"""Offline tests for persistent SHA256-checked Release cache snapshots."""
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from release_dataset_cache import build_snapshot, restore_snapshot, candidates


def row(name,entity='Q1234'):
    return {'person':{'type':'uri','value':'http://www.wikidata.org/entity/'+entity},
            'name':{'type':'literal','xml:lang':'zh','value':name}}

class ReleaseCacheTests(unittest.TestCase):
    def test_roundtrip_and_keep_valid_local_cache(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);source=d/'source';source.mkdir()
            file=source/'P103_ru_Q7737_5000.json'
            file.write_text(json.dumps({'results':{'bindings':[row('托尔斯泰')]}}),encoding='utf8')
            file2=source/'P103_ru_Q7737_5000_P103-han-alias-v2.json'
            file2.write_text(json.dumps({'results':{'bindings':[row('普希金','Q2222')]}}),encoding='utf8')
            (source/'P103_de_Q188_5000.json.part').write_text('incomplete')
            (source/'bad.txt').write_text('not a cache')
            pack=d/'saved.tar.gz'
            self.assertEqual(build_snapshot(source,pack),2)
            target=d/'target';target.mkdir()
            self.assertEqual(restore_snapshot(pack,target),2)
            self.assertEqual(len(candidates(target)),2)
            self.assertEqual(restore_snapshot(pack,target),0)
            self.assertEqual((target/file.name).read_bytes(),file.read_bytes())

    def test_reject_tampered_payload(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);source=d/'s';source.mkdir()
            file=source/'P103_ko_Q9176_5000.json'
            file.write_text(json.dumps({'results':{'bindings':[row('金正恩')]}}),encoding='utf8')
            pack=d/'snapshot.tar.gz';self.assertEqual(build_snapshot(source,pack),1)
            # Repackage with original manifest but changed data -> SHA mismatch.
            with tarfile.open(pack) as f:
                manifest=f.extractfile('manifest.json').read()
            bad=d/'tampered.tar.gz'
            import io
            with tarfile.open(bad,'w:gz') as f:
                payload=b'x'*file.stat().st_size
                for name,content in [(file.name,payload),('manifest.json',manifest)]:
                    info=tarfile.TarInfo(name);info.size=len(content)
                    f.addfile(info,io.BytesIO(content))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                restore_snapshot(bad,d/'out')

if __name__=='__main__':unittest.main()
