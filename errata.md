# Errata

Changes made in the proofreading pass of October 2026 (Task 7 on the [project board](PROJECT_BOARD.md)), that affect the published text (No Starch, 2023).

Only changes that affect the printed text are listed: one entry per change, by chapter and section, old text → new text. Content fixes are marked "(content)". Changes that appear only in the notebooks (solutions, hidden cells, layout, the examples, `modsim.py`) are listed in [proofreading.md](proofreading.md), along with the decisions behind the content fixes.

**chap01**

- *False Precision*: "a 1 meter railing" → "a 1-meter railing"
- *Summary*: "Empire State building" → "Empire State Building"
- Exercise 1: "side-by-side" → "side by side"
- Exercise 2: $sin^2 (\pi/4) + cos^2 (\pi/4)$ → $\sin^2 (\pi/4) + \cos^2 (\pi/4)$ (upright function names)
- Exercise 3: "a 10 foot pole" → "a 10-foot pole"
- Exercise 6: heading "Exercise 6:" → "Exercise 6"
- Exercise 6, step 1: "...with constant acceleration `a`." → "...with constant acceleration `a`?"
- *The Falling Penny Myth* (content): "The television show *Mythbusters* has tested the myth ... and a physical model of a human head." → "The science video channel *Veritasium* tested the myth of the falling penny more carefully, with help from Adam Savage of *Mythbusters* ... measurements of a penny falling at terminal velocity in a vertical wind tunnel, and research on how much force it takes to fracture a skull." (the link always went to the Veritasium video)

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

**chap05**

- Introduction: "Paul Erlich" → "Paul Ehrlich"
- *Absolute and Relative Errors*: "counting everyone is the world is hard" → "counting everyone in the world is hard"
- *Modeling Population Growth*: "the second to last element" → "the second-to-last element"

**chap06**

- *System Objects*: "`t0` and `t_end` are the first and last years" → "`t_0` and `t_end`..."
- *Under the Hood*: "defined in the ModSim library, is based on" → "defined in the ModSim library is based on"; Python docs link from 3.7 to the current version
- *Factoring Out the Update Function* (content): "the second parameter is a function, like `growth_func1`, that computes the population for the next year" → "the second argument is a function, like `growth_func1`, that computes the net growth during the current year"

**chap07**

- *Quadratic Growth*: "the parameters `alpha` and `beta` ." → "the parameters `alpha` and `beta`."
- *Net Growth*: "To plot growth rate versus population" → "To plot net growth versus population"
- Introduction (content): removed "In the exercises, you will have a chance to try the second." (this chapter's exercises don't try a time-varying growth rate)
- Exercise 2 (content): "initial populations between 1 and 20 billion" → "between 1 and 25 billion" (matches the solution)

**chap08**

- *Summary*: "In this chapter we use the quadratic growth model" → "we used"
- Exercise 1: "time stamp" → "timestamp" (twice)

**chap09**

- *Difference Equations* (content): "The population models in the previous chapter and this one" → "The population models in the previous chapters"
- *Analysis and Simulation*: "see <http://modsimpy.com/plato>.)." → "see <http://modsimpy.com/plato>)."
- *Solving the Quadratic Growth Model*: added the missing period after the Khan Academy link
- *Summary*: $f(t) = p_0 \exp{\alpha t}$ → $f(t) = p_0 \exp(\alpha t)$ (the braces rendered without parentheses)

**chap10**

- *Historical World Population*: broken Wikipedia link (`.../World_population_estimates.html`, 404) → <https://en.wikipedia.org/wiki/World_population_estimates>

**chap11**

- *The Kermack-McKendrick Model*: "or the variables name `beta`" → "or the variable name `beta`"
- Heading "Implementing the KM model" → "Implementing the KM Model"
- *Implementing the KM Model*: "Now that we have object to represent" → "Now that we have objects to represent"
- *Now With a TimeFrame*: "a kind of a `DataFrame`" → "a kind of `DataFrame`"
- *Running the Simulation* (content): "The parameters of `run_simulation` are..." → "`run_simulation1`"; "initial conditions, and values of `0` and `t_end`" → "initial conditions, and `t_end`"; "We can call `run_simulation` like this" → "`run_simulation1`"
- *Collecting the Results* (content): "about five weeks (35 days) to peak" → "about four weeks (30 days) to peak" (the infected fraction peaks on day 30)

**chap12**

- *Sweeping Immunization*: "This following function" → "The following function"

**chap13**

- *Sweeping Beta*: "also called an *f-string* to assemble a label" → "also called an *f-string*, to assemble a label"

**chap14**

- Introduction: "Let's think what that relationship might be" → "Let's think about what that relationship might be"
- *Contact Number*: "we divided the contact rate by the infection rate to get $c$" → "by the recovery rate"
- *Exploring the Results*: `--` → `---` around the aside about `metric`
- *Analysis and Simulation* (content): the explanation of why simulation and analysis disagree when $c < 1$ was "the simulation divides time into a discrete series of days, whereas the analysis treats time as a continuous quantity." Now: the analysis assumes the initial number of infections is negligible (like a very large population); the simulation starts with one student, about 1% of the class, who infects a few others before recovering; when $c > 1$ these few are negligible compared to the outbreak. (A continuous-time model with the same start gives the same small totals, so discrete time is not the cause.)
- *Estimating Contact Number* (content): "anywhere from 20% to 80%, we would conclude that $c$ is near 2" → "anywhere from 25% to 60%, we would conclude that $c$ is near 1.3" (by the chapter's formula, 20%–80% is $c$ from 1.1 to 2.0; 25%–60% is 1.15 to 1.5, in the steep part of the curve)

**chap15**

- *Temperature and Heat* (content): "related to the average velocity of the particles" → "related to the average kinetic energy of the particles"
- *Newton's Law of Cooling*: `"law "` → `"law"`
- *Implementing Newtonian Cooling*: "time stamp" → "timestamp" (three places); the `loc` example said the world population labels start in 1960 → 1950
- *Estimating r*: "about -11°C" → "about -11 °C"

**chap16**

- Introduction: "root finding algorithm" → "root-finding algorithm"
- *Mix First or Last?*: "61.5 °C which is still" → "61.5 °C, which is still"
- *Mix First or Last?* (content): the mix-last case simulated the coffee for 30 minutes but the milk for only 15 (its `t_end` from the measurement). Now both run 30 minutes (`milk.t_end = 30`), and "the temperature is 62.9 °C" → "63.2 °C", which matches the end of the sweep. Re-executed
- *Analytic Solution*: "If we observe the that the temperature" → "If we observe that the temperature"; "the value of `r` we computed in the previous chapter, `0.115`" → "`0.0115`"; "time stamps" → "timestamps"

**chap17**

- Introduction (content): dropped the dead "PDF at <http://modsimpy.com/minmod>" (the redirect went to an unrelated homepage; no free PDF found); `/minmod` now goes to the abstract, like `/bergman`
- *The Minimal Model*: "neither too high or too low" → "neither too high nor too low"; "Long term, severe hyperglycemia" → "Long-term, severe hyperglycemia"; "2--10 minutes" → "2–10 minutes"
- *The Minimal Model* (content): "affects almost 10% of the population in the U.S." → "about 12%" (CDC, 2023 data); `/cdc` now goes to the CDC's current report page
- *The Glucose Minimal Model* (content): "The initial state of the model is $X(0) = I_b$" → "$X(0) = 0$" (as in chapter 18's code and Bergman's model)
- *Getting the Data* (content): the Pacini and Bergman (1986) citation linked a different paper on ResearchGate → <https://pubmed.ncbi.nlm.nih.gov/3640682/>
- *Interpolation*: "equally-spaced" → "equally spaced"

**chap18**

- Introduction: "to solve the differential equation using a better algorithm" → "differential equations"
- "time stamp" → "timestamp" (three places)
- *Running the Simulation* (content): "`X` ... then declines as the concentration of glucose declines" → "as the concentration of insulin in the blood declines" (in the model, `X` is driven by `I`, not `G`)

**chap19**

- *The Glucose Minimal Model*: `"least squares"; so-named because` → `"least squares", so named because`
- *The Insulin Minimal Model*: "Berman et al." → "Bergman et al."
- *HIV*: the link to Phillips (1996) was dead (404) → <https://doi.org/10.1126/science.271.5248.497> (the journal's page; the old link was a free copy); also removed a stray ")" and added "*Science*, 1996"

**chap20**

- *Newton's Second Law of Motion*: $H(x, y, \frac{dy}{dt})$ → $H(x, y, \frac{dy}{dx})$
- *Dropping Pennies*: "time stamp" → "timestamp"
- Exercise 1: "meters, kilograms, and Newtons" → "newtons"; the *Ask an Astronomer* link (dead, 404) → its Wayback Machine copy from 2018 (as `greenteapress.com/matlab/astro` already does)

**chap21**

- *Drag Force* (content): "Using the low end of their range, 40 mph" → "Using a value near the low end of their range, 40 mph" (the range is 35–65 mph)
- *The Params Object*: "inital state" → "initial state"; "the density of air, and acceleration due to gravity, and the observed terminal velocity" → "the density of air, acceleration due to gravity, and the observed terminal velocity"
- *Simulating the Penny Drop*: "time stamp" → "timestamp"

**chap22**

- *Vectors*: "the same angle and magnitude of `A`" → "as `A`"
- *Simulating Baseball Flight*: a colon after the *Physics of Baseball* link → period
- *Drag Force*: "in Newtons" → "in newtons"; "time stamp" → "timestamp" (twice)
- *Visualizing Trajectories*: "about 57° degrees" → "about 57°"
- *Animating the Baseball*: `animate` "takes ... a `TimeSeries`" → "a `TimeFrame`"

**chap23**

- Introduction: "the angle you should hit a baseball" → "the angle at which you should hit a baseball"; "called `maximize_scalar` that searches" → "called `maximize_scalar`, that searches"
- *The Manny Ramirez Problem*: "the angle the ball should leave home plate" → "the angle at which the ball should leave home plate"
- *Finding the Range* (content): "The first parameter is the function we want to maximize. The second is the range of values we want to search" → "The first argument is the function we want to maximize. The second is `params`, which `maximize_scalar` passes along to `range_func`. The keyword argument `bounds` is the range of values we want to search"
- Exercise 1: the error function "takes a speed and a `System` object" → "a `Params` object"
- *Under the Hood*: stray ")" after the link; `/minimize` (Golden-section search) now goes to Brent's method, which the text describes

**chap24**

- *Simulating the System*: "time stamp" → "timestamp"; "The final value of `r` is 0.55 m" → "0.055 m"
- *Analytic Solution* (content): the label "(1)" moved from $d\theta/dt = \omega$ to $dr/dt = k \omega$, so "Dividing Equations 1 and 2 yields $dr/dy = k/r$" is correct (dividing $d\theta/dt$ by $dy/dt$ gives $d\theta/dy = 1/r$)
- *Analytic Solution* (content): "the same as the "estimate" we computed in Section xxx" → "earlier in the chapter"

**chap25**

- *Angular Acceleration*: "newton meters (Nm)" → "newton meters (N m)"
- *Teapots and Turntables* (content): "its weight is 7 kg" → "its mass is 7 kg"; "The teapot weights 0.3 kg" → "The mass of the teapot is 0.3 kg"
- Heading "Two Phase Simulation" → "Two-Phase Simulation"
- *Estimating Friction* (content): "0.333 N m, a little less than the initial guess" → "a little more" (the guess was 0.3)

**chap26**

- *Bungee Dunk Revisited*: "In the previous case study, we assume" → "we assumed"; a stray "." before ")" in the Heck et al. citation
- Heading "Spider-man" → "Spider-Man"; "Spider-Man weighs 76 kg" → "has a mass of 76 kg" (as in chap25)
- *Simulating a Yo-yo*: "rolled up string" → "rolled-up string"; "counter-clockwise" → "counterclockwise" (twice); an empty last row (`\\` before `\end{aligned}`) removed from the results

**chap27**

- Introduction: "Most of the time you don't need to know, you can use" → "you don't need to know; you can use"
- *How root_scalar Works*: missing period after the SciPy link; a stray backtick before "You just invented a numerical method called *bisection*"
- *How maximize_scalar Works*: alt text for the second golden-section figure (was empty)
