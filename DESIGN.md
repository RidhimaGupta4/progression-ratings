# Study design (written before any model was fitted)

This file fixes the analysis choices before results were seen, so they cannot drift towards a preferred
answer. Any later deviation is listed at the bottom with its reason.

## Question
How much value does each player add by moving the ball (passes, carries and dribbles in open play), and
how many minutes of data are needed before that number can be trusted?

## Data
- StatsBomb Open Data, 2015/16 season, commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb` (pinned).
- Premier League (380 matches), La Liga (380), Serie A (380), Ligue 1 (377; StatsBomb could not source
  video for 3). Total 1,517 matches, all event and line-up files verified against the git blob hashes.
- Excluded: Bundesliga 2015/16. The public release contains only Bayer Leverkusen's 34 matches, a
  single-team selection, not a league season.

## Time split (strictly temporal)
For each league, the season midpoint is the date of the first round-20 fixture. Matches played before it
form the first half (H1); matches played on or after it form the second half (H2). Splitting by date, not
round number, keeps postponed early-round fixtures that were played late in H2.

## Actions
Built from StatsBomb events in their original order: pass (open-play or set-piece subtype, cross flag),
carry, dribble, shot (penalty / free-kick subtypes), interception, tackle, clearance, ball recovery, block,
goalkeeper action, foul, miscontrol/dispossession (bad touch). Ball receipts, pressures and administrative
events are not actions. Coordinates are StatsBomb's (120 x 80, acting team attacking left to right).

## Action-value model (VAEP; Decroos et al., 2019)
- Labels: does the acting team score (concede) within the next 10 actions, including the current one, in
  the same period? Own goals count for the benefiting team.
- Features: the current and the two previous actions only: type, result, body part, start and end
  location, distance and angle to goal, movement, time between actions, whether the previous actions were
  by the same team, goal difference before the action, period and elapsed time. Nothing after the current
  action is used. The current action's result and end location are inputs by design: the value is
  assigned after the action has happened.
- Model: gradient-boosted trees with fixed hyperparameters chosen in advance (300 iterations, learning
  rate 0.05, 31 leaves, min 200 samples per leaf, L2 1.0, no early stopping).
- Leakage control: the model applied to league L is trained only on the first halves of the other three
  leagues. Every action value in league L, in both halves, therefore comes from a model that never saw
  league L and never saw anything played after any league's midpoint.
- Value of an action: [P_score(after) - P_score(before)] - [P_concede(after) - P_concede(before)], with the
  "before" state taken from the previous action (team perspective swapped if the previous action was by the
  opponent) and set to zero at the start of each period and after a goal.
- Validation: on each held-out league, log loss, Brier score and AUC against the base rate and a
  location-only model; calibration curve.

## Player measure
Open-play ball-progression value per 90 minutes: the sum of action values of a player's open-play passes,
carries and dribbles, divided by minutes played / 90. Shots, set pieces and defensive actions are excluded
(finishing and set-piece duty are separate, noisier skills). Minutes come from the line-ups' on-pitch
position intervals (changed: see Deviations 1). Goalkeepers are excluded. Position group: the group in which the player spent most
minutes in that half (CB, FB/WB, DM/CM, AM/W, CF).

## Rating model (Bayesian, fitted on H1 only)
Match-level likelihood: progression value of player p in match m ~ Normal(lambda_p * t_pm, sigma_g^2 * t_pm),
where t_pm is minutes/90 (variance proportional to time on pitch, as for a sum of independent increments).
Player rates are partially pooled: lambda_p ~ Normal(mu[league, group], tau_g). Weakly informative priors.
Fitted with NUTS (NumPyro); convergence checked with R-hat and effective sample size.

## Validation of ratings (H1 -> H2)
For players with at least 450 H1 minutes and at least 450 H2 minutes:
1. Point prediction of each player's H2 rate from (a) the league-position mean, (b) the raw H1 rate,
   (c) the Bayesian posterior mean. Minutes-weighted mean squared error; uncertainty by bootstrap over teams
   (players in a team are dependent).
2. Calibration of uncertainty: coverage of the 50% and 80% posterior predictive intervals for each
   player's H2 rate (should be about 50% and 80%).
3. Reliability against minutes: reliability(m) = tau^2 / (tau^2 + sigma^2 / (m/90)); minutes needed for
   reliability 0.5, 0.7 and 0.8; compared with the empirical H1-H2 correlation in bands of minutes.
4. Positive control: the same pipeline must detect something known to be stable.

## Known limitation stated in advance
A player's value depends on his team (better teams create more valuable situations). Within one season
players rarely change teams, so player and team effects cannot be separated cleanly. Ratings describe
value added in the player's actual team context; a team-adjusted variant is reported as a sensitivity.

## Deviations from this design
Each change below was made before the H1 -> H2 validation numbers it could affect were final, and each is
applied identically to every player, league and analysis.

1. **Minutes are computed from events, not from line-up intervals.** The line-up files contain errors: in
   at least one match a player who left for treatment and was then substituted is listed as back on the
   pitch until the final whistle, which puts 12 players on the pitch for 38 team-matches. Minutes now come
   from the 'Starting XI', 'Substitution' and red-card events; a test checks that no team ever has more
   than 11 players on the pitch. Position group per player-match is the most common position in his events.
2. **Players who changed club during the season.** 145 players appear for two or more teams (38 in two
   leagues). A player's league, team and position group in a fitted period are those where he played the
   most minutes in that period. The H1 -> H2 point-prediction comparison is also reported for single-team
   players only, as a sensitivity.
3. **Added to the validation:** coverage of 95% predictive intervals (in addition to 50% and 80%); a
   team-adjusted model as a sensitivity (the limitation below anticipated this); bootstrap over teams for
   the H1-H2 correlation intervals as well as for the error comparison; 95% intervals (bootstrap over teams)
   for 80% coverage overall and by rating level; posterior 95% intervals for the minutes needed to reach a
   given reliability (the point values are plug-in posterior means).
4. **Positive control fixed as open-play passes per 90** (the design above named no specific measure):
   pass volume is a well-known stable style trait, so the pipeline should find high reliability for it.
5. **Player-specific noise (added after a first-half diagnostic, before its validation was run).** A check
   on first-half data only showed that a player's match-to-match noise grows with his rate (median noise SD
   in the top third of rates is about 1.3-1.9x that of the bottom third, in every position group;
   first written as 1.5-1.9x, corrected when the figures were re-checked). One noise
   level per group would then give intervals that are too narrow for high-rate players and too wide for
   low-rate players, even if overall coverage looks right. Added variant: sigma_p = sigma_g * exp(omega_g * z_p),
   z_p ~ Normal(0, 1), i.e. each player's noise level is partially pooled towards his group's. Both models
   are validated H1 -> H2 and both are reported. Coverage is also reported by thirds of the first-half rating
   within each position group, and for the top 10%.
   Decision rule, fixed before the variant's H1 -> H2 results were seen: the published ratings use the
   player-specific-noise model if (a) its overall 80% coverage is within 3 percentage points of 80% and (b) its
   80% coverage in the top third is closer to 80% than the original model's; otherwise they use the original.
   Outcome: the variant failed both conditions (overall 71.4%, top third 68.2%, against 71.8% for the original),
   so the published ratings use the original model (results/model_choice.csv). No further variants were tried
   against the second half.
6. **Published ratings** are fitted on the full season with the same model and settings. The procedure is
   what was validated forward in time; the full-season numbers themselves have no out-of-sample check.

## Corrections made during the analysis
- Own goals: team names differ between StatsBomb match files and event files (e.g. "Olympique de
  Marseille" vs "Marseille"). The benefiting team was first resolved by name, which mislabelled two own
  goals by Marseille. It is now resolved by team id, and the pipeline asserts that the goals credited to
  each team equal that team's score in all 1,517 matches. The action-value model and everything downstream
  were re-run after the fix.
- Team-adjusted variant: its model-implied half-to-half correlation covers player skill only, but it was first
  compared with correlations of rates that still contained team effects. The observed correlation for that variant
  is now computed after removing the first-half team effects (like-for-like). The main model is unaffected.
- Model-implied half-to-half correlation: first computed as the average of each player's implied correlation,
  while the observed value is one Pearson correlation pooled over all position groups. These are different
  quantities when groups differ in skill spread. The implied value is now pooled in the same way:
  sum(tau^2) / sqrt(sum(tau^2 + sigma^2/t1) * sum(tau^2 + sigma^2/t2)). Changes for the main model were at
  most 0.014; conclusions unchanged.
