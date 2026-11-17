# OpenCC dependency (opencc-js 1.4.2)

The dataset normalizer uses the published [opencc-js 1.4.2](https://www.npmjs.com/package/opencc-js) JavaScript implementation with `opencc-data 1.4.2` and its dictionaries. It applies `jp -> cn` followed by `t -> cn` to every candidate containing Chinese Han characters, before name filtering, deduplication, collision checking, and deterministic train/validation/test splitting.

Official tarball: `https://registry.npmjs.org/opencc-js/-/opencc-js-1.4.2.tgz`

Install it by running `python tools/install_opencc.py` in the project root. This pins the exact 1.4.2 archive, verifies the installed `package.json` version, and performs basic conversion checks. The training Action calls the installer automatically before the test suite. OpenCC failures abort training; there is no silent fallback to untranslated text.

Upstream licenses: opencc-js MIT; bundled opencc-data Apache-2.0. They are included in the npm distribution itself. The experiment package does not vendor the old 1.4.1 bundle from `iqonli/MusicCard`.
