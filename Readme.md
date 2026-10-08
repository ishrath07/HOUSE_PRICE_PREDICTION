# House Price Prediction

End-to-end regression project on the Ames Housing dataset, built in an industry-style structure
(pipelines, tests, reproducible config).

> **Status:** Phase 0 complete (scaffold and decisions). Next: Phase 1 (data and EDA).

## Quick start

```bash
make setup                 # create venv, install deps, install package
# put train.csv in data/raw/  (see data/README.md)
make data                  # create the fixed train/test split
make test                  # run tests
```

## Project layout

```
data/raw, data/processed   input data (git-ignored)
notebooks/                 exploration only
src/house_price/           reusable code (config, data, features, pipeline, train, evaluate, predict)
models/                    saved model artifacts
reports/figures/           plots for the README
tests/                     pytest
docs/decisions.md          problem, metric, evaluation protocol
```

## Roadmap

- [x] Phase 0: Setup and decisions
- [ ] Phase 1: Data and EDA
- [ ] Phase 2: Baseline
- [ ] Phase 3: Features and preprocessing pipeline
- [ ] Phase 4: Modelling and validation
- [ ] Phase 5: Evaluation and diagnostics
- [ ] Phase 6: Packaging (CLI, tests, docs)

## Results
_Filled in as the project progresses._