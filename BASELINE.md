# Phase 2 Baseline

Canonical benchmark preset:

- Config file: `configs/phase2_baseline_n5_plus_ready.json`
- Executable scope: 1D repeater-chain DQN
- `n=5`
- `pgen=0.4`
- `pswap=0.7`
- `obs_mode=counter_exposed_plus_ready`
- `seed=12345`
- `reward_mode=base`
- `max_actions=100`
- `eval_episodes=500`
- `eval_epsilon=0.0`
- `model_tag=phase2_baseline_n5`

## Train

```bash
python scripts/train_qamel.py \
  --n 5 \
  --pgen 0.4 \
  --pswap 0.7 \
  --obs_mode counter_exposed_plus_ready \
  --seed 12345 \
  --reward_mode base \
  --model_tag phase2_baseline_n5 \
  --train_episodes 10000 \
  --checkpoint_every 1000 \
  --log_every 100
```

Training artifacts are written under:

- `qamel/outputs/runs/dqn_n5_pgen0.4_pswap0.7_phase2_baseline_n5/`
- `qamel/outputs/models/dqn_n5_pgen0.4_pswap0.7_phase2_baseline_n5.pt`

## Evaluate

```bash
python scripts/evaluate_qamel.py \
  --n 5 \
  --pgen 0.4 \
  --pswap 0.7 \
  --obs_mode counter_exposed_plus_ready \
  --seed 12345 \
  --reward_mode base \
  --model_tag phase2_baseline_n5 \
  --eval_episodes 500 \
  --max_actions 100 \
  --eval_epsilon 0.0
```

Evaluation now defaults to the run-local model:

- `qamel/outputs/runs/dqn_n5_pgen0.4_pswap0.7_phase2_baseline_n5/model.pt`

Evaluation outputs are written under:

- `qamel/outputs/runs/dqn_n5_pgen0.4_pswap0.7_phase2_baseline_n5/evaluations/`

Key files:

- `eval_episodes500_max100_eps0.0.csv`
- `eval_episodes500_max100_eps0.0_seed12345.csv`
- `eval_episodes500_max100_eps0.0_seed12345_summary.json`
- `eval_episodes500_max100_eps0.0_seed12345_ent_counts.txt`
- `eval_episodes500_max100_eps0.0_seed12345_swap_counts.txt`

## Smoke Check

Short smoke versions of the same commands:

```bash
python scripts/train_qamel.py \
  --n 5 \
  --pgen 0.4 \
  --pswap 0.7 \
  --obs_mode counter_exposed_plus_ready \
  --seed 12345 \
  --reward_mode base \
  --model_tag phase2_baseline_n5 \
  --train_episodes 2 \
  --checkpoint_every 1 \
  --log_every 1 \
  --force_train
```

```bash
python scripts/evaluate_qamel.py \
  --n 5 \
  --pgen 0.4 \
  --pswap 0.7 \
  --obs_mode counter_exposed_plus_ready \
  --seed 12345 \
  --reward_mode base \
  --model_tag phase2_baseline_n5 \
  --eval_episodes 2 \
  --max_actions 100 \
  --eval_epsilon 0.0
```
