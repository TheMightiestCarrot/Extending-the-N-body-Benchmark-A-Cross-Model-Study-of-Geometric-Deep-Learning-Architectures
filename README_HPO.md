hpo quickstart

## objective
- fisher-combined ks p over: collisions, group-collisions, leaving, sharp30, sharp45, stickings, plus energy_total/potential/kinetic p-values.
- per trial we log log(combined p); best config is retrained per regime.

## run
```bash
# from repo root
python hpo/hpo.py --trials 8 --trial_minutes 40 --config config.yaml
python hpo/hpo.py --trials 6 --trial_minutes 30 --config config.yaml
```

## notes
- macros saved each checkpoint; during hpo, set test_macros_every small (256–1024) for faster scoring.
- param budgets enforced by adjusting width-like knobs; tolerance ±7%.
- single gpu expected.


