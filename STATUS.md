# Project status

Last updated: Oct 12. Mid-semester report: Wed Oct 14 (5 minutes, 5 slides).

## Done

- [x] Topic chosen: Sephora skincare, ingredients vs price (backups: airline on-time bunching, F1 car vs driver)
- [x] Plan doc and rubric map written
- [x] Dataset downloaded and column names checked
- [x] Repo created, scaffold in place (.gitignore, requirements.txt, README skeleton, AI log)
- [x] Partner approved AI use for permitted tasks (Oct 12)
- [x] Data facts known: 2,420 skincare products; 2,112 after dropping gift sets and minis; 1,951 after dropping missing price, ingredients or rating; 1,884 after also dropping 67 products with "mini" in the name (found Oct 12 while checking pilot pairs)

## Before Wednesday (Oct 12 to 14)

- [x] src/01_load_clean.py runs and prints counts that match the numbers above
- [x] src/02_parse_ingredients.py: first-list rule decided, 20 products checked by eye, feature matrix built
- [x] src/03_describe.py: price and rating histograms, log price vs rating scatter, mean rating by price tier with CI
- [x] src/04_baseline_ols.py: OLS by normal equations matches a library, first R-squared ladder numbers
- [x] src/05_pilot_pairs.py: count of matched pairs at Jaccard 0.8 and 0.7
- [ ] Repo pushed, partner can clone and run
- [ ] 5 slides: title, question and data, course topics, done and left, problems
- [ ] Final paper deadline confirmed from D2L

## After Wednesday

| Dates | Work | Course topic (lecture) |
| --- | --- | --- |
| Oct 15 to 25 | Tier t-test, matched-pair test on real pairs | Confidence intervals, hypothesis tests (Oct 19) |
| Oct 26 to Nov 8 | Monte Carlo null on pairs, cluster bootstrap | Monte Carlo (Nov 2), bootstrapping (Nov 4) |
| Nov 9 to 25 | Ridge with CV, gradient boosting, premium regression | Regression (Nov 9 to 25) |
| Nov 30 to Dec 7 | PCA, final figures, rerun from a clean clone | Feature engineering, PCA |
| Dec 7 onward | Finish paper, README, AI log, rubric check | |

Start writing paper sections as results land. Do not leave the whole paper for the end.

## Open problem: placebo pairs

After dropping minis and comparing price per ounce, only 4 pairs survive at Jaccard >= 0.8 (0 across brands). Sweep: Jaccard >= 0.4 and ratio >= 1.25 gives 103 pairs but only 13 across brands. Options: nearest-neighbour matching instead of a hard cutoff, relax to ~0.5, or allow same-brand pairs only when sizes differ little.
