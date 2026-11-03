# Third-party notices

## Scope and project licence

The release’s Apache-2.0 terms apply to original project code only. Third-party packages and their components retain their own terms; the project licence does not override them.

This inventory records the direct scientific Python dependencies used by the reproduction code. It copies their installed licence and attribution texts without modification. Dependencies are installed separately; their implementations are not bundled with this source release. This is a direct-dependency inventory, not a complete audit of every transitive dependency or every possible binary build.

The reproduction code generates mathematical regression targets. This release does not bundle external dataset files or conference style files. Citations and upstream package notices remain applicable.

## Direct dependencies

| Package | Inspected version | Licence identification | Retained notices |
|:---|:---|:---|:---|
| [torch](https://pytorch.org) | 2.11.0 | BSD-3-Clause; bundled notices retained | [Licence and attribution files](third_party_licenses/torch/licenses/LICENSE) |
| [numpy](https://numpy.org) | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [Licence and attribution files](third_party_licenses/numpy/licenses/LICENSE.txt) |
| [scipy](https://scipy.org) | 1.17.1 | BSD-3-Clause; bundled notices retained | [Licence and attribution files](third_party_licenses/scipy/LICENSE.txt) |
| [matplotlib](https://matplotlib.org) | 3.10.8 | Matplotlib licence agreement | [Licence and attribution files](third_party_licenses/matplotlib/LICENSE) |
| [pykan](https://github.com/KindXiaoming/pykan) | 0.2.8 | MIT | [Licence and attribution files](third_party_licenses/pykan/LICENSE) |
| [pandas](https://pandas.pydata.org) | 3.0.2 | BSD-3-Clause; bundled notices retained | [Licence and attribution files](third_party_licenses/pandas/LICENSE) |
| [scikit-learn](https://scikit-learn.org) | 1.8.0 | BSD-3-Clause | [Licence and attribution files](third_party_licenses/scikit-learn/licenses/COPYING) |
| [sympy](https://sympy.org) | 1.14.0 | BSD-3-Clause; additional notices retained | [Licence and attribution files](third_party_licenses/sympy/licenses/AUTHORS) |
| [PyYAML](https://pyyaml.org) | 6.0.3 | MIT | [Licence and attribution files](third_party_licenses/PyYAML/licenses/LICENSE) |
| [tqdm](https://tqdm.github.io) | 4.67.3 | MPL-2.0 AND MIT | [Licence and attribution files](third_party_licenses/tqdm/licenses/LICENCE) |

## Binary-distribution notices

NumPy, SciPy, PyTorch and other distributions can include components with additional terms. Read the complete copied files, including bundled-software notices. The NumPy expression and tqdm’s combined terms are reproduced from installed metadata rather than simplified to the main project’s licence.

The listed versions describe the inspected environment. A different platform, package version or wheel can carry different component notices; regenerate the inventory for that environment.

## Reproduction

From the development repository root:

```bash
source /home/skarri/miniconda3/etc/profile.d/conda.sh
conda activate banach_kan
python experiments/export_neurips_public_licenses.py
```

The exporter reads installed distribution metadata and notice files; it does not import model packages, fit models, install dependencies or publish an archive. Its retained manifest is `experiments/results/neurips_public_release/licenses.json`.
