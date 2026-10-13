# ModSimPy Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

This is the public repo for *Modeling and Simulation in Python* (No Starch, 2023): student notebooks without solutions (`chapters/`, `examples/`), the `modsim.py` library, and the Jupyter Book site at <https://allendowney.github.io/ModSimPy/> (built from `jb/`, published on `gh-pages`). Development happens in the private repo **ModSimPySolutions**, cloned *inside* this working tree as `ModSimPySolutions/` (untracked here). Figures and most data come from a third repo, **ModSim** (`AllenDowney/ModSim`, branch `main`).

### Current focus (2026-10-13)

- **Task 1:** Commit the February 2026 build-system work that is still only local (here and in ModSimPySolutions)
- **Task 2:** One canonical `modsim.py`, one written workflow
- **Task 3:** Clear the untracked junk out of the working tree (~230 MB)
- **Task 4:** Remove tracked files nothing uses (incl. 26 MB of journal PDFs)

Tasks 1 → 2 → 3 → 4 is the suggested order: commit what's real before deleting anything.

---

## Task 1: Commit the February 2026 build-system work

**Status:** Open (added 2026-10-13)

**Background:** On 2026-02-01 the build was reworked for Python 3.13 / pandas 3 (`plan.md`): mamba environments, testing split from building, MkDocs API docs, Jupyter Book pinned below 2.0. The notebooks and the root `modsim.py` were pushed that day, but much of the tooling never was. Anyone working from GitHub (including CI) gets the old setup.

Uncommitted **here** (as of 2026-10-13):

- Modified: `Makefile` (mamba targets, `build` target that runs the whole pipeline), `README.md` (+95 lines: installation with mamba, development workflow), `modsim/modsim.py` (the git copy is stale; the local file equals the committed root `modsim.py`), `requirements.txt` (adds `nbextensions`), `requirements-dev.txt` (drops `pandoc`, `pypandoc`), `chapters/build.sh`, `examples/build.sh` (testing removed)
- Deleted: `chapters/chap18.py`
- Untracked but needed: `environment-dev.yml` (the README and Makefile use it), `build.sh` (builds the API docs), `mkdocs.yml` and `docs/` (MkDocs source)

Untracked **in ModSimPySolutions** (38 files): the build scripts themselves — `build.sh`, `chapters/build.sh`, `chapters/export_funcs.py`, `chapters/add_header.py`, `chapters/cap_titles.py`, `.gitattributes` — plus generated files (`chapters/chapNN.py`, CSV/HTML downloads, `ModSimPySolutions.zip`) that should be gitignored rather than committed.

### Checklist

- [x] `jb/` sources that built the live site (`_toc.yml`, `index.md`, `preface.md`, `build.sh`) and a working `jb/_config.yml` — committed 2026-10-13 (`c264192`). `myst_extended_syntax` (broken in Jupyter Book 1.0.4) replaced by an explicit `myst_enable_extensions` list; test builds with and without it were identical
- [ ] Review the local diffs above (`git diff --no-ext-diff`; plain `git diff` fails on notebooks here because `nbdime` isn't installed) and commit them
- [ ] Commit `environment-dev.yml`, `build.sh`, `mkdocs.yml`, `docs/`
- [ ] Decide `nbextensions` in `requirements.txt`: it's the Jupyter classic-notebook extension package, not needed for JupyterLab / Notebook 7 and not in `environment.yml`; probably drop it
- [ ] ModSimPySolutions: commit the build scripts; gitignore the generated files
- [ ] Check the CI run triggered by `c264192` (it was still running when this board was written); `gh run list -R AllenDowney/ModSimPy -L 3`

## Task 2: One canonical `modsim.py`, one written workflow

**Status:** Open (added 2026-10-13)

**Background:** There are 12 copies of `modsim.py` in the working tree. The six current ones are identical (root, `modsim/`, `chapters/`, `examples/`, `ModSimPySolutions/chapters/`, `ModSimPySolutions/examples/`); six older ones sit in `ModSimPyBook/` and `old_code/`. But the notes disagree about which copy is the source:

| Where it's written | Says the canonical copy is | Copies it to |
|---|---|---|
| `notes.txt` (2023) | root `modsim.py` | — |
| `plan.md` (2026) and `modsim/build.sh` | `modsim/modsim.py` | root, `chapters/`, `examples/`, ModSimPySolutions |
| root `build.sh` (2026) | `ModSimPySolutions/chapters/modsim.py` | `modsim/`, root |

So `make build` copies from `modsim/` outward and then, in the API-docs step, back from ModSimPySolutions. Today they match, so no harm; an edit in the "wrong" place will be silently overwritten.

**Must stay where it is:** the root `modsim.py` and `chap03.py`, `chap06.py`, `chap11.py`, `chap12.py`, `chap13.py`, `chap15.py`, `chap22.py` at the top level, and `data/World_population_estimates.html` and `figs/baseball_drag.png`. The notebooks download them from `github.com/AllenDowney/ModSimPy/raw/master/…` (and `raw.githubusercontent.com/…/master/…`), including copies of the notebooks already out in the world.

### Checklist

- [ ] Pick the canonical copy (suggest `modsim/modsim.py`, next to `test_modsim.py`, as `plan.md` says) and make every script copy *from* it; drop the reverse copy in root `build.sh`
- [ ] Merge `notes.txt` (the three-repo overview) and `plan.md` (the update workflow) into one "Development" section of `README.md`, then delete both; record here
- [ ] `Makefile`: the `lint` and `format` targets point at `pacs` (copied from another project) and a `pyproject.toml` that doesn't exist; fix or remove them

## Task 3: Clear the untracked junk out of the working tree

**Status:** Open (added 2026-10-13)

**Background:** None of this is in git; it only clutters the local copy (and makes `git status` unreadable: 74 untracked entries). Sizes as of 2026-10-13.

| Path | Size | Newest file | What it looks like |
|---|---:|---|---|
| `ModSimPyBook/` | 51 MB | 2025-05 | the old book build (LaTeX/HTML), with its own `modsim.py` |
| `old_code/` | 44 MB | 2023-09 | pre-2020 code, tests, solutions |
| `fd/` | 42 MB | 2022-09 | legacy build |
| `soln/` | 18 MB | 2021-02 | legacy solutions (superseded by ModSimPySolutions) |
| `api/` | 9.7 MB | 2025-05 | old Sphinx API docs output (now MkDocs → `site/`) |
| `colab/`, `tex/` | <1 MB | 2021 | legacy |
| `dist/`, `MANIFEST` | — | 2019 | PyPI build output for modsimpy 1.1.1 |
| `.idea/`, `.vscode/`, `.pytest_cache/` | — | — | editor/test caches (`.vscode` points at an `anaconda3` path that no longer exists) |
| `du`, `setup.py~`, `modsim/__init__.py~` | — | — | stray files |
| `picaso_da_vinci.pdf`, `stock_and_flow_diagrams.pdf` | — | 2018 | stray PDFs |
| `data/C:\nppdf32Log\debuglog.txt` | — | — | a Windows PDF-printer log, written with a literal backslash name |
| `data/USASAC*`, `data/*.hdf`, `data/baseball_drag.png`, `figs/*.eps`, `figs/secant.fig.bak` | — | — | working files from writing the book |
| `chapters/`, `examples/`: downloaded `*.csv`, `*.html`, `modsim.py`, `chapNN.py`; `chapters/chap06.md` | — | — | outputs of running notebooks locally |

`chapters/push.sh`, `chapters/test_run.sh`, `chapters/remove_soln.py`, `examples/remove_soln.py` are build helpers: decide whether they belong in git (Task 1) before deleting.

### Checklist

- [ ] Move the legacy folders and stray files to `~/ModSimPy-attic/` (not deleted yet), check that `make build` still works, then delete the attic
- [ ] `.gitignore`: add `ModSimPySolutions/`, `jb/_build/`, `jb/*.ipynb`, `jupyter_execute/`, `site/`, `.pytest_cache/`, `.vscode/`, and the notebook download outputs in `chapters/` and `examples/` (`*.csv`, `*.html`, `chap[0-9][0-9].py`, `modsim.py`), so `git status` shows only real changes
- [ ] Git housekeeping: delete the local branches `main` (2023) and `version3` (2019; also on GitHub) — both fully merged into `master`; `git gc` (145 MB of loose objects vs a 34 MB pack)

## Task 4: Remove tracked files nothing uses

**Status:** Open (added 2026-10-13)

**Background:** These are in git and on GitHub but nothing in the build or the notebooks uses them (checked against every `raw/master/…` URL in the ModSimPySolutions notebooks, 2026-10-13). Removing them from `master` doesn't shrink history; that would need a rewrite, which isn't worth it at 34 MB.

- **`papers/` (26 MB):** seven journal articles used as references for the glucose/insulin chapter, including a 19.7 MB scan of *Scientific American*, November 1977. They are other publishers' copyrighted PDFs in a public repo; **remove** and, if useful, cite them in the chapter instead
- **Packaging:** `setup.py`, `README.rst`, `modsim/__init__.py` for the PyPI package `modsimpy`, last released as 1.1.3 on 2020-01-03, far behind the current `modsim.py`. The notebooks download `modsim.py` directly, so the package isn't needed. Decide: **retire** (remove the files; say in the README that `pip install modsimpy` is outdated) or **revive** (`pyproject.toml`, publish a current version)
- **`_config.yml`** at the top (`theme: jekyll-theme-minimal`): leftover from when Pages served `master`; Pages now serves `gh-pages`
- **`apt.txt`:** Binder configuration (2019); check whether any README badge or notebook still points to Binder
- **`chap18.py`** at the top: no notebook imports it (`chapters/chap18.py` is already deleted locally); keep it if old copies of the notebooks may still download it
- **`data/`, `figs/`:** keep `data/World_population_estimates.html` and `figs/baseball_drag.png` (downloaded by notebooks); the rest (`World_population_estimates2/3.csv`, `DataOWall.csv`, `figs/*.fig`, `golden*`, `secant*`) look like book-writing leftovers — the notebooks get data and figures from the ModSim repo

### Checklist

- [ ] Remove `papers/`
- [ ] Packaging decision (retire or revive)
- [ ] Remove `_config.yml`; check and probably remove `apt.txt`
- [ ] Check the remaining `data/` and `figs/` files against old notebook versions, then remove

## Task 5: Page-view counter and Jupyter Book config

**Status:** Done (2026-10-13)

- Every page of the site loads `https://greenteapress.com/gh.gif?u=PAGE&r=REFERRER` (no cookies, no third party), counted from the greenteapress.com access logs; see GreenTeaPress `PROJECT_BOARD.md`, Task 15. Added to the 34 published pages directly (gh-pages `714ab58`) and to `jb/_config.yml` as `html: extra_footer` (`ceb1bda`), so rebuilds keep it.
- `jb/` sources and config committed (`c264192`; Task 1).
- Building the site: `make build`, or `cd jb && bash build.sh` (copies the chapters from `../ModSimPySolutions`, the API docs from `../site`, builds and pushes `gh-pages`). Jupyter Book 1.0.4.post1 is the last 1.x release; the environment pins `jupyter-book<2.0`.

---

## Notes

- The three repos: **ModSimPy** (this one, public, `master`), **ModSimPySolutions** (private, `master`; CI runs the notebooks monthly and on push — all passing through 2026-10-01), **ModSim** (shared figures and data, `main`).
- Green Tea Press page: <https://greenteapress.com/wp/modsimpy/> (WordPress page 134) links the site and this repo only, so nothing here is linked from it directly.
