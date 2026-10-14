# ModSimPy

Public repo for *Modeling and Simulation in Python* (No Starch, 2023): the student notebooks, the `modsim.py` library, and the Jupyter Book site at <https://allendowney.github.io/ModSimPy/>. Work in progress and history are in [`PROJECT_BOARD.md`](PROJECT_BOARD.md); the update workflow is in the README's "Development" section.

## Three repos

- **ModSimPy** (this one, public, `master`): student notebooks in `chapters/` and `examples/` (generated; no solutions, no outputs), `modsim/`, `jb/` (book config), `gh-pages` (the published site).
- **ModSimPySolutions** (private, `master`), cloned inside this tree as `ModSimPySolutions/` (gitignored here). **The source notebooks live there**; edit those, never the student copies.
- **ModSim** (`main`): shared figures and data.

`modsim/modsim.py` is the one canonical copy of the library; `modsim/build.sh` copies it everywhere else. Keep the top-level `modsim.py`, `chapNN.py`, `data/` and `figs/baseball_drag.png`: notebooks in the wild download them from `raw/master`.

## Editing a notebook

The `.ipynb` is the source; there is no jupytext pairing and no `.md` copies in the repo. Go through Markdown and back:

```bash
jupytext --to md -o /tmp/chap05.md ModSimPySolutions/chapters/chap05.ipynb
# edit /tmp/chap05.md
jupytext --to ipynb --update -o ModSimPySolutions/chapters/chap05.ipynb /tmp/chap05.md
```

- **Always `--update`.** Plain `jupytext --to ipynb` discards every output and the cell tags that hide solutions in the book.
- Check the round trip: only the cells you meant to change should differ (compare cell sources and output counts before and after). Removing a code cell between two Markdown cells makes jupytext **merge** them; an empty trailing Markdown cell gets dropped. Most `examples/` notebooks are nbformat 4.2 with no cell ids, so compare cells by position there.
- Markdown-only changes need no re-execution. If code changes, re-execute (`jupyter nbconvert --to notebook --execute --inplace`), then strip the per-cell `execution` timestamps nbconvert adds, and check no output recorded a local path. Some notebooks have no random seed, so re-executing changes every random output; check that the prose still holds.
- Run `pytest --nbmake` on the notebook before committing.
- The text uses non-breaking spaces before units (`2 min`, `90 °C`, `0.5 m`); match on them when editing.

## Conventions

- **Display math:** `$$` on its own line above and below the formula, no trailing spaces, no `\label` (KaTeX in VS Code and Colab doesn't support it). This form works in VS Code, Colab, Jupyter and Jupyter Book.
- **`modsim.py` calls `np.set_printoptions(legacy="1.25")`** so NumPy 2 scalars print as `72.3`, not `np.float64(72.3)`, as in the book.
- Spelling as settled in the proofreading pass (Task 7): "timestamp", "time step", "newtons", "bike share", "counterclockwise"; masses in kg are "mass", not "weight".
- Chapters have one "This chapter is available as a Jupyter notebook" notice, after the introduction.

## Short links

`modsimpy.com/...` links are redirects in `~/GreenTeaPress/domains/modsimpy.com/.htaccess`, with a copy for `greenteapress.com/modsimpy/...` in `~/GreenTeaPress/greenteapress/modsimpy/.htaccess`. Keep the two in sync. Deploy with `scripts/deploy_gtp.sh` (`--site modsimpy.com` for the domain), dry run first, and only after asking. Avoid percent-encoded targets; the server's handling of them is untested.

## Testing and publishing

- `make tests` (here): `modsim` unit tests and the student chapters. `cd ModSimPySolutions && make tests`: all solution notebooks. CI runs on every push to `master` and monthly, on the current Colab Python only.
- **Every push to `master` runs CI, so push rarely**: batch commits and push once at the end of a piece of work.
- `make build` runs the whole pipeline: it pushes ModSimPySolutions, regenerates the student notebooks and the notebook zip, pushes ModSimPy once, builds the API docs, then builds the Jupyter Book and pushes `gh-pages` (which doesn't run CI). Run the solution tests locally first. The build scripts commit only when something is staged.
- Chapter numbers in the HTML come from `numbered: True` in `jb/_toc.yml`.
