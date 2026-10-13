# ModSimPy Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

This is the public repo for *Modeling and Simulation in Python* (No Starch, 2023): student notebooks without solutions (`chapters/`, `examples/`), the `modsim.py` library, and the Jupyter Book site at <https://allendowney.github.io/ModSimPy/> (built from `jb/`, published on `gh-pages`). Development happens in the private repo **ModSimPySolutions**, cloned *inside* this working tree as `ModSimPySolutions/` (untracked here). Figures and most data come from a third repo, **ModSim** (`AllenDowney/ModSim`, branch `main`).

### Current focus (2026-10-13)

Tasks 1–4 (cleanup) and Task 6 (CI and build fixes) done 2026-10-13. Still open:

- **Task 3:** delete `~/ModSimPy-attic/` once nothing has been missed
- **Task 7:** proofreading pass through the book (next)

---

## Task 1: Commit the February 2026 build-system work

**Status:** Done (2026-10-13)

**Background:** On 2026-02-01 the build was reworked for Python 3.13 / pandas 3 (`plan.md`): mamba environments, testing split from building, MkDocs API docs, Jupyter Book pinned below 2.0. The notebooks and the root `modsim.py` were pushed that day, but much of the tooling never was. Anyone working from GitHub (including CI) gets the old setup.

Uncommitted **here** (as of 2026-10-13):

- Modified: `Makefile` (mamba targets, `build` target that runs the whole pipeline), `README.md` (+95 lines: installation with mamba, development workflow), `modsim/modsim.py` (the git copy is stale; the local file equals the committed root `modsim.py`), `requirements.txt` (adds `nbextensions`), `requirements-dev.txt` (drops `pandoc`, `pypandoc`), `chapters/build.sh`, `examples/build.sh` (testing removed)
- Deleted: `chapters/chap18.py`
- Untracked but needed: `environment-dev.yml` (the README and Makefile use it), `build.sh` (builds the API docs), `mkdocs.yml` and `docs/` (MkDocs source)

Untracked **in ModSimPySolutions** (38 files): the build scripts themselves — `build.sh`, `chapters/build.sh`, `chapters/export_funcs.py`, `chapters/add_header.py`, `chapters/cap_titles.py`, `.gitattributes` — plus generated files (`chapters/chapNN.py`, CSV/HTML downloads, `ModSimPySolutions.zip`) that should be gitignored rather than committed.

### Checklist

- [x] `jb/` sources that built the live site (`_toc.yml`, `index.md`, `preface.md`, `build.sh`) and a working `jb/_config.yml` — committed 2026-10-13 (`c264192`). `myst_extended_syntax` (broken in Jupyter Book 1.0.4) replaced by an explicit `myst_enable_extensions` list; test builds with and without it were identical
- [x] Review the local diffs above and commit them, with `environment-dev.yml`, `build.sh`, `mkdocs.yml`, `docs/index.md`, `chapters/test_run.sh` and the two `remove_soln.py` (`d0a3e79`). `docs/mkdocs.yml` (empty duplicate) and `chapters/push.sh` (repeats the end of `chapters/build.sh`) went to the attic
- [x] `nbextensions`: dropped (local change reverted)
- [x] ModSimPySolutions: committed `build.sh`, `chapters/build.sh`, `chapters/export_funcs.py`, the two `test.sh`, `.gitattributes`, `pyproject.toml`; generated files gitignored (`736489a`). The 2021 one-off notebook-editing scripts (`add_header.py`, `cap_titles.py`, `resub_all.py`, `remove_savefig.py`, `rename.py`), `glucose.md/.tex` and a stray `chapter` file went to the attic; `chapters/jntools.py` (2024 notebook utility, needs `astor` and `typer`) left untracked
- [x] CI for `c264192` and `3d82e97` passed

## Task 2: One canonical `modsim.py`, one written workflow

**Status:** Done (2026-10-13, `6143178`)

**Background:** There are 12 copies of `modsim.py` in the working tree. The six current ones are identical (root, `modsim/`, `chapters/`, `examples/`, `ModSimPySolutions/chapters/`, `ModSimPySolutions/examples/`); six older ones sit in `ModSimPyBook/` and `old_code/`. But the notes disagree about which copy is the source:

| Where it's written | Says the canonical copy is | Copies it to |
|---|---|---|
| `notes.txt` (2023) | root `modsim.py` | — |
| `plan.md` (2026) and `modsim/build.sh` | `modsim/modsim.py` | root, `chapters/`, `examples/`, ModSimPySolutions |
| root `build.sh` (2026) | `ModSimPySolutions/chapters/modsim.py` | `modsim/`, root |

So `make build` copies from `modsim/` outward and then, in the API-docs step, back from ModSimPySolutions. Today they match, so no harm; an edit in the "wrong" place will be silently overwritten.

**Must stay where it is:** the root `modsim.py` and `chap03.py`, `chap06.py`, `chap11.py`, `chap12.py`, `chap13.py`, `chap15.py`, `chap22.py` at the top level, and `data/World_population_estimates.html` and `figs/baseball_drag.png`. The notebooks download them from `github.com/AllenDowney/ModSimPy/raw/master/…` (and `raw.githubusercontent.com/…/master/…`), including copies of the notebooks already out in the world.

### Checklist

- [x] Canonical copy is `modsim/modsim.py`; the reverse copy in root `build.sh` is gone
- [x] `notes.txt` and `plan.md` merged into the README's "Development" section (three repos, canonical `modsim.py`, files that must stay put, update steps, testing); `notes.txt` removed, `plan.md` in the attic
- [x] `Makefile`: `lint` and `format` removed

Note: `make build` pushes ModSimPySolutions once, ModSimPy once (at the end of the notebook steps) and `gh-pages`; only the `master` pushes run CI (Task 6).

## Task 3: Clear the untracked junk out of the working tree

**Status:** Done except deleting the attic (2026-10-13)

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

- [x] Legacy folders and stray files moved to `~/ModSimPy-attic/` (also `modsim/solvers/` and `modsim/docs/` from 2019). Checked without pushing: MkDocs API docs build, `modsim` tests pass (40), Jupyter Book builds
- [ ] Delete `~/ModSimPy-attic/` once nothing has been missed
- [x] `.gitignore` extended; `git status` is clean. `chapters/chapNN.py` stay tracked (`chapters/build.sh` commits them)
- [x] Local branches `main` and `version3` deleted (`version3`, merged as PR #42 in 2019, also deleted on GitHub); `git gc` run (pack now 114 MB, mostly recent unreachable objects that later gc runs will prune)

Keep `modsim/__init__.py`: `mkdocs.yml` documents `::: modsim`, which imports the `modsim/` package.

## Task 4: Remove tracked files nothing uses

**Status:** Done (2026-10-13)

**Background:** These are in git and on GitHub but nothing in the build or the notebooks uses them (checked against every `raw/master/…` URL in the ModSimPySolutions notebooks, 2026-10-13). Removing them from `master` doesn't shrink history; that would need a rewrite, which isn't worth it at 34 MB.

- **`papers/` (26 MB):** seven journal articles used as references for the glucose/insulin chapter, including a 19.7 MB scan of *Scientific American*, November 1977. They are other publishers' copyrighted PDFs in a public repo; **remove** and, if useful, cite them in the chapter instead
- **Packaging:** `setup.py`, `README.rst`, `modsim/__init__.py` for the PyPI package `modsimpy`, last released as 1.1.3 on 2020-01-03, far behind the current `modsim.py`. The notebooks download `modsim.py` directly, so the package isn't needed. Decide: **retire** (remove the files; say in the README that `pip install modsimpy` is outdated) or **revive** (`pyproject.toml`, publish a current version)
- **`_config.yml`** at the top (`theme: jekyll-theme-minimal`): leftover from when Pages served `master`; Pages now serves `gh-pages`
- **`apt.txt`:** Binder configuration (2019); check whether any README badge or notebook still points to Binder
- **`chap18.py`** at the top: no notebook imports it (`chapters/chap18.py` is already deleted locally); keep it if old copies of the notebooks may still download it
- **`data/`, `figs/`:** keep `data/World_population_estimates.html` and `figs/baseball_drag.png` (downloaded by notebooks); the rest (`World_population_estimates2/3.csv`, `DataOWall.csv`, `figs/*.fig`, `golden*`, `secant*`) look like book-writing leftovers — the notebooks get data and figures from the ModSim repo

### Checklist

- [x] Remove `papers/`
- [x] Packaging: **retired**. Removed `setup.py` and `README.rst`; the README says the PyPI `modsimpy` package is outdated. `modsim/__init__.py` kept (see Task 3)
- [x] Removed `_config.yml` and `apt.txt` (nothing links to Binder)
- [x] `data/`: kept. Old notebook versions in history download `baseball_drag.csv`, `glucose_insulin.csv`, `World_population_estimates2/3.csv` and `.html` from it. `figs/`: removed `golden*` and `secant*`; only `baseball_drag.png` was ever downloaded
- [x] Root `chap18.py` kept

## Task 5: Page-view counter and Jupyter Book config

**Status:** Done (2026-10-13)

- Every page of the site loads `https://greenteapress.com/gh.gif?u=PAGE&r=REFERRER` (no cookies, no third party), counted from the greenteapress.com access logs; see GreenTeaPress `PROJECT_BOARD.md`, Task 15. Added to the 34 published pages directly (gh-pages `714ab58`) and to `jb/_config.yml` as `html: extra_footer` (`ceb1bda`), so rebuilds keep it.
- `jb/` sources and config committed (`c264192`; Task 1).
- Building the site: `make build`, or `cd jb && bash build.sh` (copies the chapters from `../ModSimPySolutions`, the API docs from `../site`, builds and pushes `gh-pages`). Jupyter Book 1.0.4.post1 is the last 1.x release; the environment pins `jupyter-book<2.0`.

---

## Task 6: CI and build fixes

**Status:** Done (2026-10-13)

- The first CI run with the `modsim` unit tests in `make tests` failed: `vector_cross` on 2D vectors raised `ValueError` under NumPy 2.x (`np.cross` dropped 2D support), and `Vector` is label-indexed under pandas 3. Fixed in `modsim/modsim.py` and copied everywhere. No notebook calls `vector_cross`
- Build scripts only commit; `make build` pushes ModSimPy once, so CI runs once on the finished state instead of three times on partial ones
- `make build` runs `examples` before `chapters`: `chapters/build.sh` builds `ModSimPyNotebooks.zip`, which includes the examples, and used to zip the previous versions
- CI in both repos: one Python version, the current Colab one (3.13; check `googlecolab/backend-info`, `os-info.txt`), on Ubuntu and Windows; `actions/checkout@v5`, `actions/setup-python@v6`

## Task 7: Proofreading pass through the book

**Status:** Planned (added 2026-10-13)

**Goal:** Read every notebook in the book for typos and for content issues, and fix them in the source notebooks in ModSimPySolutions.

**Scope:** the source notebooks in `ModSimPySolutions/`, never the student copies in `chapters/` and `examples/` here (the build regenerates those).

- `chapters/chap01`–`chap27` (about 49,000 words)
- `examples/`: 16 notebooks (about 13,000 words); `queue` is in the Jupyter Book, the rest are linked from the chapters
- Not `header.ipynb` and `link_header.ipynb` (boilerplate), except where a fix belongs there

### Decisions (2026-10-13)

- **Fixes:** typos, grammar and dead links are fixed directly. Content issues (prose vs output, dated claims, exercise problems) are listed under "Content issues for the author" with a proposed fix, and changed only after approval
- **Pace:** one notebook at a time: commit, then a summary of changes and open questions for review before the next
- **Order:** chap01–chap27, then the examples
- **Errata:** every change, typos included, goes in the errata list below; the printed edition (No Starch, 2023) doesn't have them

### What to check

- **Typos and grammar:** spelling, repeated words, punctuation, inconsistent terms or capitalization
- **Text vs output:** numbers and descriptions in the prose that no longer match what the code produces (pandas 3, NumPy 2 and SciPy changes since 2023)
- **Code that still runs but is dated:** deprecated calls and warnings, things the text says about library behavior that are no longer true
- **Links:** dead or redirected URLs, and data and figure URLs (ModSim repo, `raw/master/...` here)
- **Exercises:** clear instructions; the solution matches what the exercise asks
- **Cross-references:** chapter and section numbers, "in the previous chapter" and similar

### Workflow (per notebook)

The `.ipynb` is the source; there is no jupytext pairing and no `.md` copies in the repo (same pattern as ElementsOfDataScience):

```bash
S=<scratchpad>
jupytext --to md -o $S/chap05.md ModSimPySolutions/chapters/chap05.ipynb
# edit $S/chap05.md
jupytext --to ipynb --update -o ModSimPySolutions/chapters/chap05.ipynb $S/chap05.md
```

- `--update` keeps the outputs and metadata (cell tags, which hide solutions in the Jupyter Book). **Never run plain `jupytext --to ipynb`**: it discards every output
- Markdown-only changes: no re-execution, so outputs and figures stay as they are
- If a code cell changes, re-execute (`jupyter nbconvert --to notebook --execute --inplace ...`), strip the `execution` timestamps nbconvert adds, and check that no output recorded a local path
- Check each round trip: only the edited cells differ (compare cell sources before and after), and the number of outputs is unchanged
- Run `pytest --nbmake` on the notebook before committing

`jupytext` is in `environment-dev.yml` (added 2026-10-13; 1.19.6 installed locally).

### Commits and pushes

- One commit per notebook in ModSimPySolutions, listing what changed and any content issues left for the author
- Push rarely (each push runs CI); rebuild the student notebooks and the site with `make build` at the end, or at milestones

### Progress

| Notebooks | Status |
|---|---|
| chap01 | done (ModSimPySolutions `32a51be`, content fixes in the next commit) |
| chap02 | done (ModSimPySolutions `8303751`); content fix in the next commit |
| chap03 | done (ModSimPySolutions `4a487e4`); no content issues |
| chap04 | done (ModSimPySolutions `09930a8` and the next commit) |
| chap05 | done (ModSimPySolutions `775a602`); no content issues |
| chap06 | done (ModSimPySolutions `42c70f0`, `792001b`) |
| chap07 | done (ModSimPySolutions `1385073` and the next commit) |
| chap08 | done (ModSimPySolutions `3ea71b0` and the next commit) |
| chap09 | done (ModSimPySolutions `7cf4af4` and the next commit) |
| chap10–chap27 | not started |
| examples (16) | not started |

### Content issues for the author

Items that need a decision rather than an obvious fix. Mark each **open**, **approved** or **declined**.

**chap01**

1. **approved: describe Veritasium** · *The Falling Penny Myth*: the text says Mythbusters tested the myth and gives <https://modsimpy.com/myth/> for "the results", but that short link redirects to a Veritasium video ("How Dangerous is a Penny Dropped From a Skyscraper?"). Either point the link at the Mythbusters segment or change the text to describe the Veritasium video. The short link is on modsimpy.com, so it can be fixed without changing the notebook. (The 29 m/s terminal velocity matches the 64.4 mph Mythbusters estimate.)
2. **approved: CC BY-NC-SA 4.0** (README fixed; it now also says the code is MIT, as `LICENSE` says) · License: the notebook header says CC BY-NC-SA 4.0; the README says CC BY-NC 4.0. Probably shared by every notebook's header, so decide once for the whole book.
3. **approved: rewritten for Colab and JupyterLab** · *More Jupyter* (hidden in the book, shown to students): the instructions are for the classic Notebook interface ("Insert cell below" in the Insert menu, "Close and Halt", "Restart & Run All"), which Notebook 7, JupyterLab and Colab don't have. Proposed: rewrite briefly so it works for Colab and JupyterLab.
4. **declined** · Exercise 8 solution uses `round` to split the pace into minutes and seconds. That works for 7.22 min/mile but would give a negative remainder for, say, 7.6. Proposed: `floor` instead of `round` (and import `floor` from NumPy rather than shadowing the built-in `round`).

**chap02**

1. **approved: delete the second** · *Modeling a Bike Share System*: the same point is made twice in a row. "...in a `State` object, which gets assigned to a new variable named `bikeshare`." and, two sentences later, "The `State` object is assigned to a new variable named `bikeshare`." Proposed: delete the second sentence.

**chap04**

1. **approved: reword** · Exercise 6 asked for "a range of values for `p1` and `p2`" but only sweeps `p1`.
2. **approved: short answer** · Exercise 6's solution didn't answer "What value of `p1` minimizes...?"

**chap06**

1. **approved: fix both** · The text and the Exercise 1 solution docstring said the growth function computes next year's population; it returns the net growth.

**chap07**

1. **approved: drop the sentence** · The introduction promised an exercise on a time-varying growth rate that isn't in this chapter.
2. **approved: change the text** · Exercise 2 said 1 to 20 billion; the solution uses 1 to 25.

**chap08**

1. **approved: move it** · The notebook notice was in the Exercises section instead of after the introduction.

**chap09**

1. **approved: "previous chapters"** · "The population models in the previous chapter and this one" (this chapter has no new model).
2. Notebook notice moved, applying the chap08 decision.

### Errata (changes relative to the published text)

One entry per change: notebook, section, old text → new text. Content fixes are marked as such.

**chap01**

- *False Precision*: "a 1 meter railing" → "a 1-meter railing"
- *Summary*: "Empire State building" → "Empire State Building"
- Exercise 1: "side-by-side" → "side by side"
- Exercise 2: $sin^2 (\pi/4) + cos^2 (\pi/4)$ → $\sin^2 (\pi/4) + \cos^2 (\pi/4)$ (upright function names)
- Exercise 3: "a 10 foot pole" → "a 10-foot pole"
- Exercise 6: heading "Exercise 6:" → "Exercise 6"
- Exercise 6, step 1: "...with constant acceleration `a`." → "...with constant acceleration `a`?"
- *The Falling Penny Myth* (content): "The television show *Mythbusters* has tested the myth ... and a physical model of a human head." → "The science video channel *Veritasium* tested the myth of the falling penny more carefully, with help from Adam Savage of *Mythbusters* ... measurements of a penny falling at terminal velocity in a vertical wind tunnel, and research on how much force it takes to fracture a skull." (the link always went to the Veritasium video)
- *More Jupyter* (content; not in the printed book): classic-Notebook menu steps replaced with Colab and JupyterLab steps

**chap02**

- Exercise 3, steps 4–5: "it should plot the results and / 5. Decorate the axes." → "it should plot the results. / 5. Then it should decorate the axes."
- Exercise 3, testing step 2: "Call `run_simulation` with parameters" → "with arguments" (the chapter defines the difference)
- *Under the Hood*: Matplotlib docs link from version 3.2.2 to the current docs (<https://matplotlib.org/stable/api/axes_api.html>)
- *Modeling a Bike Share System* (content): removed the repeated sentence "The `State` object is assigned to a new variable named `bikeshare`."

**chap03**

- *Metrics*: "system-owner's point of view" → "system owner's point of view"
- *Summary*: "We used a conditional operator, `==`" → "We used a comparison operator, `==`" (the chapter calls them comparison operators)
- Exercise 2: "run the simulation with parameters `p1=0.3`..." → "with arguments"

**chap04**

- Introduction: "bike sharing system" → "bike share system"
- *Loops and Arrays*: "equally-spaced" → "equally spaced"
- *Sweeping Parameters* and *Under the Hood*: Matplotlib colors links (one moved, one to version 3.3.2) → <https://matplotlib.org/stable/users/explain/colors/colors.html>
- Exercise 6 (content): "with a range of values for `p1` and `p2`." → "with a range of values for `p1` and these parameters:" (only `p1` is swept)
- Exercise 6 solution (content; solutions aren't in the printed book): added the answer, "The minimum is near p1 = 0.35, close to p2 = 0.3."

**chap05**

- Introduction: "Paul Erlich" → "Paul Ehrlich"
- *Absolute and Relative Errors*: "counting everyone is the world is hard" → "counting everyone in the world is hard"
- *Modeling Population Growth*: "the second to last element" → "the second-to-last element"

**chap06**

- *System Objects*: "`t0` and `t_end` are the first and last years" → "`t_0` and `t_end`..."
- *Under the Hood*: "defined in the ModSim library, is based on" → "defined in the ModSim library is based on"; Python docs link from 3.7 to the current version
- *Factoring Out the Update Function* (content): "the second parameter is a function, like `growth_func1`, that computes the population for the next year" → "the second argument is a function, like `growth_func1`, that computes the net growth during the current year"
- Exercise 1 solution docstring (content; not in the printed book): "Compute the population next year ... returns: population next year" → "Compute the net growth during the current year ... returns: net growth"

**chap07**

- *Quadratic Growth*: "the parameters `alpha` and `beta` ." → "the parameters `alpha` and `beta`."
- *Net Growth*: "To plot growth rate versus population" → "To plot net growth versus population"
- Introduction (content): removed "In the exercises, you will have a chance to try the second." (this chapter's exercises don't try a time-varying growth rate)
- Exercise 2 (content): "initial populations between 1 and 20 billion" → "between 1 and 25 billion" (matches the solution)

**chap08**

- *Summary*: "In this chapter we use the quadratic growth model" → "we used"
- Exercise 1: "time stamp" → "timestamp" (twice)
- Layout (content): the "This chapter is available as a Jupyter notebook" note moved from the start of Exercises to after the introduction, with the same wording as the other chapters

**chap09**

- *Difference Equations* (content): "The population models in the previous chapter and this one" → "The population models in the previous chapters"
- *Analysis and Simulation*: "see <http://modsimpy.com/plato>.)." → "see <http://modsimpy.com/plato>)."
- *Solving the Quadratic Growth Model*: added the missing period after the Khan Academy link
- *Summary*: $f(t) = p_0 \exp{\alpha t}$ → $f(t) = p_0 \exp(\alpha t)$ (the braces rendered without parentheses)
- Layout (content): notebook notice moved from Exercises to after the introduction (same as chap08)

## Notes

- The three repos: **ModSimPy** (this one, public, `master`), **ModSimPySolutions** (private, `master`; CI runs the notebooks monthly and on push — all passing through 2026-10-01), **ModSim** (shared figures and data, `main`).
- Green Tea Press page: <https://greenteapress.com/wp/modsimpy/> (WordPress page 134) links the site and this repo only, so nothing here is linked from it directly.
