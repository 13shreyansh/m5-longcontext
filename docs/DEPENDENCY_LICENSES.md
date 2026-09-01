# Dependency and licence inventory

This inventory covers the exact Python 3.9 / macOS arm64 environment used for
the verified public package. `requirements-solution.txt` declares the four
direct dependencies; `requirements-lock.txt` freezes the complete tested
18-package resolution and the exact target-wheel SHA-256 for each package. The
public setup enforces those hashes and refuses source distributions. The public
repository does **not** redistribute Python wheels or their installed
directories. Pip obtains them separately, and each installed distribution
remains governed by its own terms.

Licence labels below are observations from the installed wheel metadata and
classifiers, not new licence grants or legal conclusions. Where metadata does
not provide one unambiguous identifier, that limitation is preserved.

| Package | Version | Relationship | Installed metadata observation |
|---|---:|---|---|
| NumPy | 2.0.2 | Direct runtime/test dependency | BSD classifier; installed `LICENSE.txt` contains the primary BSD terms and bundled-component notices |
| PyTorch | 2.8.0 | Direct runtime dependency | `BSD-3-Clause`; installed wheel includes `LICENSE` and `NOTICE` |
| Ninja | 1.13.0 | Direct native-build dependency | No single `License` field; classifiers list Apache Software License and BSD License; wheel includes `LICENSE_Apache_20` and `AUTHORS.rst` |
| pytest | 8.4.2 | Direct test dependency | MIT; wheel includes `LICENSE` and `AUTHORS` |
| exceptiongroup | 1.3.1 | pytest transitive dependency | MIT classifier |
| filelock | 3.19.1 | PyTorch transitive dependency | Unlicense |
| fsspec | 2025.10.0 | PyTorch transitive dependency | BSD-3-Clause |
| iniconfig | 2.1.0 | pytest transitive dependency | MIT |
| Jinja2 | 3.1.6 | PyTorch transitive dependency | BSD classifier |
| MarkupSafe | 3.0.3 | Jinja2 transitive dependency | BSD-3-Clause |
| mpmath | 1.3.0 | SymPy transitive dependency | BSD metadata/classifier |
| NetworkX | 3.2.1 | PyTorch transitive dependency | BSD classifier |
| packaging | 26.3 | pytest transitive dependency | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.6.0 | pytest transitive dependency | MIT |
| Pygments | 2.21.0 | pytest transitive dependency | BSD-2-Clause |
| SymPy | 1.14.0 | PyTorch transitive dependency | BSD metadata/classifier |
| tomli | 2.4.1 | pytest dependency on Python 3.9 | MIT |
| typing_extensions | 4.16.0 | PyTorch/pytest transitive dependency | PSF-2.0 |

## Exact target-wheel evidence

The following files were selected by pip 26.0.1 for CPython 3.9 on macOS arm64
using `--no-deps --only-binary=:all:`. They were downloaded into ignored audit
storage, hashed, installed with `--require-hashes`, and are not redistributed.

| Wheel filename | SHA-256 |
|---|---|
| `exceptiongroup-1.3.1-py3-none-any.whl` | `a7a39a3bd276781e98394987d3a5701d0c4edffb633bb7a5144577f82c773598` |
| `filelock-3.19.1-py3-none-any.whl` | `d38e30481def20772f5baf097c122c3babc4fcdb7e14e57049eb9d88c6dc017d` |
| `fsspec-2025.10.0-py3-none-any.whl` | `7c7712353ae7d875407f97715f0e1ffcc21e33d5b24556cb1e090ae9409ec61d` |
| `iniconfig-2.1.0-py3-none-any.whl` | `9deba5723312380e77435581c6bf4935c94cbfab9b1ed33ef8d238ea168eb760` |
| `jinja2-3.1.6-py3-none-any.whl` | `85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67` |
| `markupsafe-3.0.3-cp39-cp39-macosx_11_0_arm64.whl` | `f71a396b3bf33ecaa1626c255855702aca4d3d9fea5e051b41ac59a9c1c41edc` |
| `mpmath-1.3.0-py3-none-any.whl` | `a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c` |
| `networkx-3.2.1-py3-none-any.whl` | `f18c69adc97877c42332c170849c96cefa91881c99a7cb3e95b7c659ebdc1ec2` |
| `ninja-1.13.0-py3-none-macosx_10_9_universal2.whl` | `fa2a8bfc62e31b08f83127d1613d10821775a0eb334197154c4d6067b7068ff1` |
| `numpy-2.0.2-cp39-cp39-macosx_14_0_arm64.whl` | `2b2955fa6f11907cf7a70dab0d0755159bca87755e831e47932367fc8f2f2d0b` |
| `packaging-26.3-py3-none-any.whl` | `d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c` |
| `pluggy-1.6.0-py3-none-any.whl` | `e920276dd6813095e9377c0bc5566d94c932c33b27a3e3945d8389c374dd4746` |
| `pygments-2.21.0-py3-none-any.whl` | `2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9` |
| `pytest-8.4.2-py3-none-any.whl` | `872f880de3fc3a5bdc88a11b39c9710c3497a547cfa9320bc3c5e62fbf272e79` |
| `sympy-1.14.0-py3-none-any.whl` | `e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5` |
| `tomli-2.4.1-py3-none-any.whl` | `0d85819802132122da43cb86656f8d1f8c6587d54ae7dcaf30e90533028b49fe` |
| `torch-2.8.0-cp39-none-macosx_11_0_arm64.whl` | `e9f071f5b52a9f6970dc8a919694b27a91ae9dc08898b2b988abbef5eddfd1ae` |
| `typing_extensions-4.16.0-py3-none-any.whl` | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` |

## Direct upstream identities

These source URLs came from the installed distribution metadata:

- NumPy: <https://github.com/numpy/numpy>
- PyTorch: <https://github.com/pytorch/pytorch>
- Ninja Python distribution: <https://github.com/scikit-build/ninja-python-distributions>
  and upstream Ninja: <https://github.com/ninja-build/ninja>
- pytest: <https://github.com/pytest-dev/pytest>

## Installed direct-package licence evidence

The following SHA-256 values were observed from the verified environment. They
identify the licence/notice files shipped inside those installed wheels; the
files themselves are not copied into this repository.

```text
NumPy LICENSE.txt             279d8a786f5c0011fa62f57d09c9b724bc2391e0d907836a316309c7a3eb6cc4
PyTorch LICENSE               c0ae8cfa6af6fa041232ce34e3b7f7bf79dae23ab611189f857e015568452263
PyTorch NOTICE                c2cc7bf0caec7652c2b460a8a470bea1677f241e4ab8e431df34cf17f5a9fec0
Ninja LICENSE_Apache_20       73ba74dfaa520b49a401b5d21459a8523a146f3b7518a833eea5efa85130bf68
Ninja AUTHORS.rst             6c6135b7f2e19b68abf12ee7fe36cb0e884ff387e9279b0d0aec6f0d5b0a3508
pytest LICENSE                ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b
pytest AUTHORS                c2017d6288262cf9c39aecc8e51df3422a9f2bbca1e4b55d28491022070c7577
```

This environment inventory is separate from copied/adapted solution source.
The latter is limited to the pinned MIT-licensed MLX and triton-msl material
whose full notices are retained under `solution/third_party/` and whose bytes
and transformations are checked by `scripts/verify_solution_provenance.py`.
