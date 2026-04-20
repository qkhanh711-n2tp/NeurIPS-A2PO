# Experiments

This folder captures the experiments described in [A2PO.md](../A2PO.md).

Included experiments:

1. `exp01_matrix_game`
   Heterogeneous cooperative matrix game.
2. `exp02_navigation`
   Cooperative navigation with heterogeneous dynamics.
3. `exp03_heterogeneity_ablation`
   Matrix-game heterogeneity sweep.
4. `exp04_mujoco_linear`
   Multi-Agent MuJoCo with linear Gaussian policies.
5. `exp05_halfcheetah_mlp`
   HalfCheetah-6x1 with MLP policies.
6. `exp06_scaling`
   Matrix-game scaling to more agents.

What is in this folder:

- `registry.py`
  Canonical experiment registry derived from the paper markdown.
- `specs.py`
  Dataclasses for experiment/task/runtime metadata.
- `run.py`
  CLI to list experiments or export a resolved config into `generated/`.
- `configs/*.json`
  Frozen experiment configs for direct inspection or external tooling.

Current scope:

- The folder records the exact experiment structure and hyperparameters from the paper draft.
- Experiments 1, 2, 3, and 6 are marked as synthetic tasks that can be implemented inside this repo.
- Experiments 4 and 5 are marked as requiring a Multi-Agent MuJoCo backend that is not yet present in this repo.

Examples:

```bash
cd /home/khanh/Khanh_stuff/Inprogress/A2PO
python experiments/run.py --list
python experiments/run.py --name exp01_matrix_game
python experiments/run.py --name exp04_mujoco_linear --write
```
