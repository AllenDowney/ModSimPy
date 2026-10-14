# Proofreading pass (Task 7)

Details for Task 7 on the [project board](PROJECT_BOARD.md): decisions, method, progress, and the content issues raised for the author with their outcomes. The list of changes is in [errata.md](errata.md); the editing workflow is in [CLAUDE.md](CLAUDE.md).


**Goal:** Read every notebook in the book for typos and for content issues, and fix them in the source notebooks in ModSimPySolutions.

**Scope:** the source notebooks in `ModSimPySolutions/`, never the student copies in `chapters/` and `examples/` here (the build regenerates those).

- `chapters/chap01`–`chap27` (about 49,000 words)
- `examples/`: 16 notebooks (about 13,000 words); `queue` is in the Jupyter Book, the rest are linked from the chapters
- Not `header.ipynb` and `link_header.ipynb` (boilerplate), except where a fix belongs there

## Decisions (2026-10-13)

- **Short links** (`modsimpy.com/...`) live in `~/GreenTeaPress`: `domains/modsimpy.com/.htaccess`, plus a copy for `greenteapress.com/modsimpy/...` in `greenteapress/modsimpy/.htaccess`. Keep the two in sync; deploy with `scripts/deploy_gtp.sh` (dry run first) after asking

- **Fixes:** typos, grammar and dead links are fixed directly. Content issues (prose vs output, dated claims, exercise problems) are listed under "Content issues for the author" with a proposed fix, and changed only after approval
- **Pace:** one notebook at a time: commit, then a summary of changes and open questions for review before the next
- **Order:** chap01–chap27, then the examples
- **Errata:** every change, typos included, goes in [errata.md](errata.md); the printed edition (No Starch, 2023) doesn't have them

## What to check

- **Typos and grammar:** spelling, repeated words, punctuation, inconsistent terms or capitalization
- **Text vs output:** numbers and descriptions in the prose that no longer match what the code produces (pandas 3, NumPy 2 and SciPy changes since 2023)
- **Code that still runs but is dated:** deprecated calls and warnings, things the text says about library behavior that are no longer true
- **Links:** dead or redirected URLs, and data and figure URLs (ModSim repo, `raw/master/...` here)
- **Exercises:** clear instructions; the solution matches what the exercise asks
- **Cross-references:** chapter and section numbers, "in the previous chapter" and similar

## Workflow (per notebook)

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

## Commits and pushes

- One commit per notebook in ModSimPySolutions, listing what changed and any content issues left for the author
- Push rarely (each push runs CI); rebuild the student notebooks and the site with `make build` at the end, or at milestones

## Progress

Pushed after chap10 (2026-10-13): both repos, one push each; student notebooks and the site not rebuilt yet.

**Released after chap27 (2026-10-13):** ModSimPySolutions and ModSimPy pushed (CI passed on both); student chapters and examples, the notebook zip and `modsim.py` regenerated; site rebuilt and published (chapter numbers, all fixes; Pages deploy checked live); short links `/minmod`, `/cdc`, `/minimize` deployed (GreenTeaPress `2f9dd77`). All 44 solution notebooks passed nbmake locally first.

**Released after the examples (2026-10-13):** `make build` end to end (with the build-script fix, it skipped the empty commits instead of stopping): ModSimPySolutions and ModSimPy pushed once each, CI passed on both; student examples regenerated (no solutions, no outputs); site rebuilt and checked live (e.g. `queue`). All 44 solution notebooks passed nbmake locally first.

**Build snag:** `make build` stops when a build script has nothing new to commit (the scripts run `git commit` unconditionally and exit with its status), e.g. when the exported `chapNN.py` are unchanged. This time the steps were run by hand in the Makefile's order. Fix: make those commits conditional (`git diff --cached --quiet || git commit ...`). **Fixed 2026-10-13** in `chapters/build.sh`, `examples/build.sh` and `ModSimPySolutions/chapters/build.sh`.

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
| chap10 | done (ModSimPySolutions `199fc93` and the next commit; re-executed) |
| chap11 | done (ModSimPySolutions `149cee0`, `6490eb4`) |
| chap12 | done (ModSimPySolutions `d06ddb3`, `690f145`) |
| chap13 | done (ModSimPySolutions `dee3e7f`); no content issues |
| chap14 | done (ModSimPySolutions `6bfcf9f`, `4a65094`, and the next commit) |
| chap15 | done (ModSimPySolutions `c4c9907`, `335dcdb`; re-executed) |
| chap16 | done (ModSimPySolutions `94a41db`, `8321b6d`; re-executed) |
| chap17 | done (ModSimPySolutions `97c752b`, `adeda99`); short links deployed |
| chap18 | done (ModSimPySolutions `866e2ce`, `22bdca1`; re-executed) |
| chap19 | done (ModSimPySolutions `4ce0858`); no content issues |
| chap20 | done (ModSimPySolutions `fa03e2c`, `8a36d9e`; re-executed) |
| chap21 | done (ModSimPySolutions `170aa03`, `9220418`; re-executed) |
| chap22 | done (ModSimPySolutions `2200872`, `5fe402d`; re-executed) |
| chap23 | done (ModSimPySolutions `6fa5b26`, `a4d94c3`); short link deployed |
| chap24 | done (ModSimPySolutions `11c6cc2`, `4c8c42c`) |
| chap25 | done (ModSimPySolutions `2d5c3d6`, `1e3b20d`) |
| chap26 | done (ModSimPySolutions `6520729`); no content questions |
| chap27 | done (ModSimPySolutions `24d8087`) |
| examples/bungee1 | done (ModSimPySolutions `cd4a948`, `4fdb76f`) |
| examples/bungee2 | done (ModSimPySolutions `b1217d5`, `2463673`; re-executed) |
| examples/filter | done (ModSimPySolutions `bbb9eb9`, `4c71f98`) |
| examples/glucose | done (ModSimPySolutions `1bcc911`, `64698f9`; re-executed) |
| examples/hiv_model | done (ModSimPySolutions `a47420c`); no content questions |
| examples/insulin | done (ModSimPySolutions `f8e8983`) |
| examples/kitten | done (ModSimPySolutions `eac904a`); no content questions |
| examples/orbit | done (ModSimPySolutions `3f8cf4e`); no content questions |
| examples/plague | done (ModSimPySolutions `5fb8d84`); no content questions |
| examples/queue | done (ModSimPySolutions `aea2509`, `2182ff9`; re-executed) |
| examples/salmon | done (ModSimPySolutions `4e6abd1`; re-executed); no content questions |
| examples/spiderman | done (ModSimPySolutions `a5c25ba`, `64364b5`; re-executed) |
| examples/throwingaxe | done (ModSimPySolutions `cd5a64a`, `959405d`; re-executed) |
| examples/trees | done (ModSimPySolutions `086c633`); no content questions |
| examples/wall | done (ModSimPySolutions `b205eeb`, `082cf55`) |
| examples/yoyo | done (ModSimPySolutions `479920f`); no content questions |
| **all chapters and examples** | **done** (2026-10-13); `header.ipynb` skipped (boilerplate) |

## Content issues for the author

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

**chap10**

1. **approved: fix and re-execute** · Wrong title on the 1940-present solution plot.

**chap11**

1. **approved** · Peak timing: 35 days → 30 days.
2. **approved: fix all three** · `run_simulation1` named as `run_simulation`, and "values of `0` and `t_end`".
3. **approved: add the count** · Exercise 1 asks how many students; the solution gave a fraction.
4. **done** · <http://modsimpy.com/sir> redirected to maa.org, which returns 403 Forbidden (at least to scripts). Now points to <https://sites.math.duke.edu/education/postcalc/sir/index.html> (same article). Deployed 2026-10-13, GreenTeaPress `241076f`.

**chap12**

1. **approved: increase in gamma, with the mechanism explained** · The quarantine solution said "reduction in gamma" but the code raises gamma. Discussed lowering beta instead: that models fewer contacts for every infected student (distancing), not isolating identified cases; for the total-infected metric both give the same result when beta/gamma is the same.

**chap14**

1. **approved after discussion** · The $c < 1$ discrepancy is due to the size of the initial infection, not discrete time; reworded as above.
2. **approved after discussion** · The steep-curve example: 25%–60% infected → $c$ near 1.3. The bullet shows that in the steep part, a wide (noisy) range of observations still gives about the same $c$; the flat part (second bullet) is where $c$ is hard to estimate.

**chap15**

1. **approved** · Temperature relates to average kinetic energy, not average velocity.
2. **approved: fix and re-execute** · Exercise 2 solution didn't assign the `root_scalar` result.
3. **approved: drop the Exercises copy** · Chapters 15–27 have the notebook notice twice; keep only the one after the introduction (applies to every remaining chapter).
4. **approved** · NumPy 2 scalar display: `np.set_printoptions(legacy="1.25")` in `modsim.py`.

**chap16**

1. **approved** · Mix last now simulates the milk for 30 minutes: 62.9 → 63.2 °C.
2. **declined** · "identical ... with a small difference due to rounding" kept as is.

**chap17**

1. **approved** · $X(0) = I_b$ → $X(0) = 0$.
2. **approved: drop "PDF at"** · Dead PDF link; `/minmod` redirected to the abstract.
3. **approved: PubMed** · The MINMOD citation linked a different paper.
4. **approved: update both** · Diabetes prevalence 10% → 12%; `/cdc` redirected to the current report.

**chap18**

1. **approved** · `X` declines with blood insulin, not glucose.
2. **approved: drop it** · The hidden G-minus-X comparison cell (a relative comparison of `X` is meaningless because `X` starts at 0).

**chap20**

1. **approved** · Exercise 1 solution computed the answer in days but didn't display it.

**chap21**

1. **approved: fix and re-execute** · `params` overwritten in Exercise 1, leaking `v_init=-30` into Exercise 2.
2. **approved** · `error_func` docstring.
3. **approved: "near the low end"** · 40 mph is not the low end of 35–65 mph.

**chap22**

1. **approved: fix all and re-execute** · Three slips in the solution code (`details` for `details2`; old functions tested in Exercise 3).

**chap23**

1. **approved** · The arguments of `maximize_scalar` were described wrongly.
2. **approved: Brent's method** · `/minimize` → <https://en.wikipedia.org/wiki/Brent_method> (redirects to "Brent's method"; ASCII-only, to keep `%27` out of the `.htaccess`).

**chap24**

1. **approved: "earlier in the chapter"** · "Section xxx" placeholder.
2. **approved: renumber** · "Dividing Equations 1 and 2" referred to the wrong Equation 1.

**chap25**

1. **approved** · 0.333 is a little more than the guess of 0.3.
2. **approved: say "mass"** · Masses in kg were called weights.

**chap27**

1. **declined** · "evaluates the slope function seven times" per step (six new evaluations, since the last is reused); kept as a fair simplification.

**examples/bungee1**

1. **approved: match the code** · Parameter list had L = 40 m, k = 20 N/m.
2. **approved: replace** · Obsolete note about `zero_force` and units.

**examples/bungee2**

1. **approved: test at 54** · Spring-force test didn't match the text.
2. **approved** · Stale note about an event function in Phase 2.

**examples/filter**

1. **approved: simpler** · Unfinished `tau` bullet.
2. **approved** · Stale `t_eval=ts` suggestion.
3. **approved** · Offset approaches -90 degrees.

**examples/glucose**

1. **approved** · Fit plot showed the starting parameters.
2. **declined** · Minus sign in $S \equiv -\delta E / \delta I$ kept as is.

**examples/insulin**

1. **declined** · The normal-range citation ("Pacini and Bergman") links a ResearchGate paper with a different title; kept as is.

**examples/queue**

1. **approved: fix and re-execute** · Test runs reused stale `results`.

**examples/spiderman**

1. **approved: code to 76** · Mass 75 kg in the code vs 76 kg in the text.
2. **approved** · The 1-D spring force is in the bungee case studies, not Chapter 21.
3. **approved: drop it** · Orphan sentence.

**examples/throwingaxe**

1. **approved** · The "final state" figure drew the initial state.
2. **approved: reword** · No `make_system` in this notebook.
3. **declined** · Target "typically 6 m away" vs 8 m in the simulation; left as is (open-ended exercise).

**examples/wall**

1. **approved: remove** · Obsolete paragraph about units.

**Book-wide: display math format** (ModSimPySolutions `6ffa856`)

- VS Code showed a KaTeX ParseError in chapter 18 (*The Update Function*). Rather than depend on each reader's setup, every `$$...$$` block now has the delimiters on their own lines, with no trailing spaces. Markup only (57 cells in 19 notebooks). The two `\label{}` in chap24 (unreferenced; unsupported by KaTeX) are removed
- Checked by rendering every markdown cell with VS Code's notebook renderer settings (`@vscode/markdown-it-katex`, KaTeX 0.16.21, `throwOnError`, bare and fenced blocks): 0 errors (was 2, in chap24). Jupyter Book (MathJax) builds with all math rendered. Confirmed working in VS Code
- **Jupyter Book chapter numbers** (ModSimPy `6d58752`): `numbered: True` on the Chapters part of `jb/_toc.yml`, preface in its own Front Matter part (as in ThinkBayes2). Test build shows "18. Glucose and Insulin", sections 18.1, ...; live on the next site build
