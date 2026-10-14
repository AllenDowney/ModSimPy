# Errata

Changes made in the proofreading pass of October 2026 (Task 7 on the [project board](PROJECT_BOARD.md)), relative to the published text (No Starch, 2023) and the notebooks as they were before.

One entry per change: notebook, section, old text → new text. Content fixes are marked "(content)". Entries marked "not in the printed book" (solutions, hidden cells, layout, and all of `examples/`) don't affect the print edition; the rest do. Decisions behind the content fixes are in [proofreading.md](proofreading.md).

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

**chap10**

- *Historical World Population*: broken Wikipedia link (`.../World_population_estimates.html`, 404) → <https://en.wikipedia.org/wiki/World_population_estimates>
- Solution plot (content; not in the printed book): the 1940-present plot was titled "Prehistoric population estimates" → "Model vs. estimates since 1940". Notebook re-executed; the other three figures were re-rendered by current Matplotlib with no visible change

**chap11**

- *The Kermack-McKendrick Model*: "or the variables name `beta`" → "or the variable name `beta`"
- Heading "Implementing the KM model" → "Implementing the KM Model"
- *Implementing the KM Model*: "Now that we have object to represent" → "Now that we have objects to represent"
- *Now With a TimeFrame*: "a kind of a `DataFrame`" → "a kind of `DataFrame`"
- *Running the Simulation* (content): "The parameters of `run_simulation` are..." → "`run_simulation1`"; "initial conditions, and values of `0` and `t_end`" → "initial conditions, and `t_end`"; "We can call `run_simulation` like this" → "`run_simulation1`"
- *Collecting the Results* (content): "about five weeks (35 days) to peak" → "about four weeks (30 days) to peak" (the infected fraction peaks on day 30)
- Exercise 1 solution (content; not in the printed book): added "About 38% of the class, or 34 of the 90 students, get infected."
- Layout (content): notebook notice moved from Exercises to after the introduction (same as chap08)

**chap12**

- *Sweeping Immunization*: "This following function" → "The following function"
- Exercise 1 solution (content; not in the printed book): "model quarantine as an effective reduction in gamma" → "an effective increase in gamma", with an explanation: R means "removed", quarantined students can't infect anyone, so a shorter time in the infectious compartment is a shorter recovery time and a higher `gamma = 1/tr`
- Layout (content): notebook notice moved from Exercises to after the introduction (same as chap08)

**chap13**

- *Sweeping Beta*: "also called an *f-string* to assemble a label" → "also called an *f-string*, to assemble a label"
- Layout (content): notebook notice moved from Exercises to after the introduction (same as chap08)

**chap14**

- Introduction: "Let's think what that relationship might be" → "Let's think about what that relationship might be"
- *Contact Number*: "we divided the contact rate by the infection rate to get $c$" → "by the recovery rate"
- *Exploring the Results*: `--` → `---` around the aside about `metric`
- *Analysis and Simulation* (content): the explanation of why simulation and analysis disagree when $c < 1$ was "the simulation divides time into a discrete series of days, whereas the analysis treats time as a continuous quantity." Now: the analysis assumes the initial number of infections is negligible (like a very large population); the simulation starts with one student, about 1% of the class, who infects a few others before recovering; when $c > 1$ these few are negligible compared to the outbreak. (A continuous-time model with the same start gives the same small totals, so discrete time is not the cause.)
- *Estimating Contact Number* (content): "anywhere from 20% to 80%, we would conclude that $c$ is near 2" → "anywhere from 25% to 60%, we would conclude that $c$ is near 1.3" (by the chapter's formula, 20%–80% is $c$ from 1.1 to 2.0; 25%–60% is 1.15 to 1.5, in the steep part of the curve)
- Layout (content): notebook notice moved from Exercises to after the introduction (same as chap08)

**chap15**

- *Temperature and Heat* (content): "related to the average velocity of the particles" → "related to the average kinetic energy of the particles"
- *Newton's Law of Cooling*: `"law "` → `"law"`
- *Implementing Newtonian Cooling*: "time stamp" → "timestamp" (three places); the `loc` example said the world population labels start in 1960 → 1950
- *Estimating r*: "about -11°C" → "about -11 °C"
- Exercise 2 solution (content; not in the printed book): `root_scalar(...)` → `res = root_scalar(...)`; the displayed `res` had been the coffee result. Re-executed: `res.root` is 0.133 for the milk
- Layout: the second notebook notice (in Exercises) removed; chapters 15–27 have one after the introduction (tagged `remove-print`) and one in Exercises

**chap16**

- Introduction: "root finding algorithm" → "root-finding algorithm"
- *Mix First or Last?*: "61.5 °C which is still" → "61.5 °C, which is still"
- *Mix First or Last?* (content): the mix-last case simulated the coffee for 30 minutes but the milk for only 15 (its `t_end` from the measurement). Now both run 30 minutes (`milk.t_end = 30`), and "the temperature is 62.9 °C" → "63.2 °C", which matches the end of the sweep. Re-executed
- *Analytic Solution*: "If we observe the that the temperature" → "If we observe that the temperature"; "the value of `r` we computed in the previous chapter, `0.115`" → "`0.0115`"; "time stamps" → "timestamps"
- Layout: duplicate notebook notice removed from Exercises

**chap17**

- Introduction (content): dropped the dead "PDF at <http://modsimpy.com/minmod>" (the redirect went to an unrelated homepage; no free PDF found); `/minmod` now goes to the abstract, like `/bergman`
- *The Minimal Model*: "neither too high or too low" → "neither too high nor too low"; "Long term, severe hyperglycemia" → "Long-term, severe hyperglycemia"; "2--10 minutes" → "2–10 minutes"
- *The Minimal Model* (content): "affects almost 10% of the population in the U.S." → "about 12%" (CDC, 2023 data); `/cdc` now goes to the CDC's current report page
- *The Glucose Minimal Model* (content): "The initial state of the model is $X(0) = I_b$" → "$X(0) = 0$" (as in chapter 18's code and Bergman's model)
- *Getting the Data* (content): the Pacini and Bergman (1986) citation linked a different paper on ResearchGate → <https://pubmed.ncbi.nlm.nih.gov/3640682/>
- *Interpolation*: "equally-spaced" → "equally spaced"
- Layout: duplicate notebook notice removed from Exercises

**Book-wide: NumPy 2 scalar display** (ModSimPy `c91e8e1`)

- NumPy 2 shows scalars as `np.float64(72.3)` where the book shows `72.3`, for every reader with current NumPy (Colab included). `modsim.py` now calls `np.set_printoptions(legacy="1.25")`. Readers get it once ModSimPy is pushed (the notebooks download `modsim.py` from `master`)
- Re-executed notebooks now match the book; SciPy's `root_scalar` result also prints slightly differently now (`flag: converged`, plus `method: brentq`)

**chap18**

- Introduction: "to solve the differential equation using a better algorithm" → "differential equations"
- "time stamp" → "timestamp" (three places)
- *Running the Simulation* (content): "`X` ... then declines as the concentration of glucose declines" → "as the concentration of insulin in the blood declines" (in the model, `X` is driven by `I`, not `G`)
- Hidden cell (content; not in the printed book): removed a comparison that computed `results.G - results2.X` (mean `inf`); re-executed
- Layout: duplicate notebook notice removed from Exercises

**chap19**

- *The Glucose Minimal Model*: `"least squares"; so-named because` → `"least squares", so named because`
- *The Insulin Minimal Model*: "Berman et al." → "Bergman et al."
- *HIV*: the link to Phillips (1996) was dead (404) → <https://doi.org/10.1126/science.271.5248.497> (the journal's page; the old link was a free copy); also removed a stray ")" and added "*Science*, 1996"

**chap20**

- *Newton's Second Law of Motion*: $H(x, y, \frac{dy}{dt})$ → $H(x, y, \frac{dy}{dx})$
- *Dropping Pennies*: "time stamp" → "timestamp"
- Exercise 1: "meters, kilograms, and Newtons" → "newtons"; the *Ask an Astronomer* link (dead, 404) → its Wayback Machine copy from 2018 (as `greenteapress.com/matlab/astro` already does)
- Exercise 1 solution (content; not in the printed book): now displays the answer in days, 64.8; re-executed
- Layout: duplicate notebook notice removed from Exercises

**chap21**

- *Drag Force* (content): "Using the low end of their range, 40 mph" → "Using a value near the low end of their range, 40 mph" (the range is 35–65 mph)
- *The Params Object*: "inital state" → "initial state"; "the density of air, and acceleration due to gravity, and the observed terminal velocity" → "the density of air, acceleration due to gravity, and the observed terminal velocity"
- *Simulating the Penny Drop*: "time stamp" → "timestamp"
- Exercise 1 solution (content; not in the printed book): `params = params.set(v_init=-30)` overwrote `params`, so in Exercise 2 the quarter was thrown down at 30 m/s. Now `params2 = ...`. Exercise 2's answer changes from v_term = 19.5 m/s, C_d = 0.54 to v_term = 21.7 m/s, C_d = 0.44 (the same as the penny). Re-executed
- Exercise 2 solution (content; not in the printed book): `error_func` docstring said "Final height as a function of C_d ... returns: height in m" → "Error in flight time as a function of v_term ... returns: difference between simulated and measured flight time in s"
- Layout: duplicate notebook notice removed from Exercises

**chap22**

- *Vectors*: "the same angle and magnitude of `A`" → "as `A`"
- *Simulating Baseball Flight*: a colon after the *Physics of Baseball* link → period
- *Drag Force*: "in Newtons" → "in newtons"; "time stamp" → "timestamp" (twice)
- *Visualizing Trajectories*: "about 57° degrees" → "about 57°"
- *Animating the Baseball*: `animate` "takes ... a `TimeSeries`" → "a `TimeFrame`"
- Solutions (content; not in the printed book): Exercise 1 displayed `details.message` → `details2.message`; Exercise 3's test cells called `drag_force` and `slope_func` → `drag_force2` and `slope_func2`. Re-executed; the stored outputs had been made under NumPy 2 before the `modsim.py` fix and showed `np.float64(...)`
- Layout: duplicate notebook notice removed from Exercises

**chap23**

- Introduction: "the angle you should hit a baseball" → "the angle at which you should hit a baseball"; "called `maximize_scalar` that searches" → "called `maximize_scalar`, that searches"
- *The Manny Ramirez Problem*: "the angle the ball should leave home plate" → "the angle at which the ball should leave home plate"
- *Finding the Range* (content): "The first parameter is the function we want to maximize. The second is the range of values we want to search" → "The first argument is the function we want to maximize. The second is `params`, which `maximize_scalar` passes along to `range_func`. The keyword argument `bounds` is the range of values we want to search"
- Exercise 1: the error function "takes a speed and a `System` object" → "a `Params` object"
- *Under the Hood*: stray ")" after the link; `/minimize` (Golden-section search) now goes to Brent's method, which the text describes
- Layout: duplicate notebook notice removed from Exercises

**chap24**

- *Simulating the System*: "time stamp" → "timestamp"; "The final value of `r` is 0.55 m" → "0.055 m"
- *Analytic Solution* (content): the label "(1)" moved from $d\theta/dt = \omega$ to $dr/dt = k \omega$, so "Dividing Equations 1 and 2 yields $dr/dy = k/r$" is correct (dividing $d\theta/dt$ by $dy/dt$ gives $d\theta/dy = 1/r$)
- *Analytic Solution* (content): "the same as the "estimate" we computed in Section xxx" → "earlier in the chapter"
- Display math: the two unused `\label{}` removed (book-wide math change)
- Layout: duplicate notebook notice removed from Exercises

**chap25**

- *Angular Acceleration*: "newton meters (Nm)" → "newton meters (N m)"
- *Teapots and Turntables* (content): "its weight is 7 kg" → "its mass is 7 kg"; "The teapot weights 0.3 kg" → "The mass of the teapot is 0.3 kg"
- Heading "Two Phase Simulation" → "Two-Phase Simulation"
- *Estimating Friction* (content): "0.333 N m, a little less than the initial guess" → "a little more" (the guess was 0.3)
- Layout: duplicate notebook notice removed from Exercises

**chap26**

- *Bungee Dunk Revisited*: "In the previous case study, we assume" → "we assumed"; a stray "." before ")" in the Heck et al. citation
- Heading "Spider-man" → "Spider-Man"; "Spider-Man weighs 76 kg" → "has a mass of 76 kg" (as in chap25)
- *Simulating a Yo-yo*: "rolled up string" → "rolled-up string"; "counter-clockwise" → "counterclockwise" (twice); an empty last row (`\\` before `\end{aligned}`) removed from the results

**chap27**

- Introduction: "Most of the time you don't need to know, you can use" → "you don't need to know; you can use"
- *How root_scalar Works*: missing period after the SciPy link; a stray backtick before "You just invented a numerical method called *bisection*"
- *How maximize_scalar Works*: alt text for the second golden-section figure (was empty)

**examples/bungee1** (not in the printed book)

- "in the opposite of their direction of motion" → "opposite to their direction of motion"; "a `Param` object" → "a `Params` object"; "springs back almost to almost 70 m" → "almost to 70 m"; "more oscillation that we expect" → "than we expect"; "the numerical derivative of `ys`" → "of `results.v`"; docstring "from system|" → "from system:"
- `event_func.direction = 10` → `1`, matching the text ("The value 1 (or any positive value)")
- (content) The parameter list said `L = 40 m` and `k = 20 N / m`; the code uses 25 m and 40 N/m → list matches the code; "With a terminal velocity of 60 m/s." → "Let's assume the jumper's terminal velocity is 60 m/s."
- (content) Removed "returns `zero_force`, which is either 0 Newtons or 0, depending on whether the `System` object has units..." → "If the cord is not extended, it returns 0."

**examples/bungee2** (not in the printed book)

- Same slips as bungee1: "in the opposite of their direction" → "opposite to"; "`Param`" → "`Params`"; missing period; also "The jumpers's" → "The jumper's"
- `cord_acc` docstring: "Computes the force of the bungee cord ... velocity of the jumpter ... acceleration in m/s" → "Computes the acceleration of the jumper due to the cord ... velocity of the jumper ... acceleration in m/s**2"
- (content) The spring-force test used `spring_force(56)` (1 m above full extension, giving -40) under text that says 1 m of extension gives 40 N → `spring_force(54)`. Re-executed
- (content) "Here's how we run Phase 2, setting the direction of the event function so it doesn't stop the simulation immediately." → "Here's how we run Phase 2." (Phase 2 has no event function)

**examples/filter** (not in the printed book)

- "a circuit takes a signal" → "a circuit that takes a signal"; "audio signal" → "audio signals"; "pass the `Params` object `make_system` which" → "to `make_system`, which"; missing period; "is call" → "is called"; "functions that ... returns" → "return"; `estimate_offset` docstring "returns: amplitude ratio" → "phase offset in degrees"
- (content) The `tau` bullet stopped mid-sentence → "which determines how long the initial startup phase lasts."
- (content) Dropped "I suggest using `t_eval=ts`..." (no `ts`; the notebook uses `num=401`)
- (content) "For high frequencies, it approaches 90 degrees" → "-90 degrees"
- The two electronics-tutorials.ws links couldn't be checked (the site hangs for scripts)

**examples/glucose** (not in the printed book)

- "In the previous chapter" → "In Chapter 18"; "Levenberg-Marquart" → "Levenberg–Marquardt"; "at the some time stamps" → "at the same timestamps"; pandas `iloc` link → <https://pandas.pydata.org/docs/user_guide/indexing.html#selection-by-position> (the old anchor is gone)
- (content) The plot after fitting showed `results.G` (the starting parameters) → `results2.G` (the best fit). Re-executed; `leastsq` under current SciPy gives the same fit to about five significant digits (SSE 318.15 vs 318.13)

**examples/hiv_model** (not in the printed book)

- Phillips (1996) link (dead, as in chap19) → <https://doi.org/10.1126/science.271.5248.497>; the response letter's old sciencemag.org PDF link → <https://doi.org/10.1126/science.272.5270.1960>
- "the initial conditional" → "the initial conditions"; "Exericise" → "Exercise"; "What do you think of the arguments on both sides." → "?"

**examples/insulin** (not in the printed book)

- "In Chapter 17 I present" → "I presented"; "time stamp" → "timestamp" (twice; "evaluated as the same" → "at the same"); `error_func` docstring "actual: array of values" → "data: DataFrame of values"; "For $I_{max}$ is the maximum" → "$I_{max}$ is the maximum"

**examples/kitten** (not in the printed book)

- "`Mcore` is the weight of the core ... `Mroll` is the total weight" → "mass" (as in chap25)
- (content) "$r_2$ is the outer diameter, and $r_1$ is the inner diameter" → "radius" (the formula uses $r^4$ with radii); "Since the outer diameter changes" → "outer radius"

**examples/orbit** (not in the printed book)

- `universal_gravitation` docstring "State object with distance r" → "with x, y, vx, and vy"; "the earth" → "the Earth"; 'DOP853' → `DOP853`

**examples/plague** (not in the printed book)

- `logistic` docstring "generalize" → "generalized"; "`fraction` and the remaining" → "`fraction`, and the remaining"; "$50" → "\$50" (escaped, like the other amounts)

**examples/queue** (in the Jupyter Book, not in the printed book)

- "the bikeshare model ... any timestep" → "the bike share model ... any time step"; alt text for the scenarios figure (was empty); "f you are not sure" → "If you are not sure"; `update_func3` comments "the first servers is busy, check it it's done" → "the first server is busy, check if it's done", "the second queue is busy" → "the second server"; a missing "?"
- (content) The two-server and two-queue test cells called `run_simulation(...)` without assigning `results`, so they plotted and measured the single-server run → `results = run_simulation(...)`. Re-executed (no random seed, so all random outputs changed; the conclusion holds: two queues 6.53 min vs one queue 6.34 min, "slightly higher")

**examples/salmon** (not in the printed book)

- Plot title 'Narraguacus River' → 'Narraguagus River'; "from year-to-year" → "from year to year"; "functon" → "function"; missing period; "to run generate" → "to generate"; "substract" → "subtract"; "The plot option `alpha=0.1`" → "`alpha=0.3`" (as in the code); `run_simulation` docstring "Simulate a queueing system" → "Simulate the population over time"
- The USASAC report link (dead; redirected to a generic NOAA page) → NOAA's Atlantic Salmon Assessments page, <https://www.fisheries.noaa.gov/resource/document/atlantic-salmon-assessments> (the 2017 report itself is probably in the NOAA repository, which blocks scripts; a copy is in `~/ModSimPy-attic/data/`); NumPy `random.choice` docs link (404) → current location
- Re-executed for the title fix (no random seed; the claims hold: median net change negative, decline vs growth 0.64 vs 0.36, "about 2:1")

**examples/spiderman** (not in the printed book)

- Spider-Man wiki link wikia.com → fandom.com; missing periods; `spring_force` docstring "Compute drag force" → "Compute spring force"; "a function that takes `t_release` and `V_0`" → "and `params`"
- (content) "Spider-Man weighs 76 kg" but `mass = 75` in the code → "has a mass of 76 kg" and `mass = 76` (as in chapter 26). Re-executed; the best range changes from 149.5 m to 149.4 m
- (content) "We saw the 1-D version in Chapter 21" → "in the bungee jumping case studies"
- (content) Removed an orphan sentence, "We can extract the x and y components as `Series` objects."

**examples/throwingaxe** (not in the printed book)

- State variables listed in code order (x, y, vx, vy, theta, omega); a doubled slash in a figure URL
- (content) "Here's a version of `make_system` that sets the initial conditions" (there is none) → "Here are the parameters and initial conditions."
- (content) "Here's the final state of the axe at the point of impact" drew `system.init` → draws `results.iloc[-1]` at `t_end`. Re-executed

**examples/trees** (not in the printed book)

- Dead links: the Flewelling et al. working paper (UW, 404) → Wayback copy from 2015 (the 2022 snapshot is a captured 404); Garcia (1983), citeseerx (404) → <https://doi.org/10.2307/2531339>; EasySDE site (forestgrowth.unbc.ca, no longer resolves) → Wayback copy from 2021
- `run_simulation` docstring "computes the population next year" → "the height next year"; "logisitic" → "logistic"; a missing colon; "**Open Exercise**" → "**Open Exercise:**"
- Checked: fitted dimension 3.11 (Model 2) and 2.64 (Model 3, "about 2.6") match the text; the analysis derivations check out

**examples/wall** (not in the printed book)

- "[which I downloaded here]" → "[available here]"; "timesteps" → "time steps"; "packs the parameters into `Series` objects" → "into tuples"; docstrings copied from other examples ("state: position, velocity", "returns: derivatives of y and v", "params: Params object") → describe this code; missing periods
- (content) Removed an obsolete paragraph about units and `require_units` (nothing in the notebook uses units)

**examples/yoyo** (not in the printed book)

- "rolled up string" → "rolled-up string"; "counter-clockwise" → "counterclockwise" (twice), as in chapter 26
- Checked: inline math written with spaces inside the delimiters (`$ a = -r \alpha $`) renders in VS Code's KaTeX, so it was left as is
