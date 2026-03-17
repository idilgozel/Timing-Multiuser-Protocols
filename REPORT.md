# Technical Report: `Timing-Multiuser-Protocols`

## Title
Technical Report on the `Timing-Multiuser-Protocols` Codebase

## Executive summary

The executable reinforcement learning code in this repository implements a stochastic **1D quantum repeater-chain control problem**, not a full multi-user grid multipartite-entanglement environment. The main environment is `RepeaterChain`, which models binary entangled links on a line of `n` nodes and lets the agent choose which adjacent links to attempt generating and which interior repeater nodes to swap in each step ([`qamel/environment.py#L3`](qamel/environment.py#L3), [`qamel/utils.py#L4`](qamel/utils.py#L4)).

At the same time, the broader repository framing is larger than the RL implementation:

- the README claims a model-free RL agent for "`n` node grid topology" and multipartite entangled distribution ([`README.md#L7`](README.md#L7)),
- the analytical code and notebooks focus on **grid-based multipartite entanglement, swapping, fusion, and latency** ([`analytical_solution/analytical_equations.py#L60`](analytical_solution/analytical_equations.py#L60), [`analytical_solution/monte_carlo.py#L4`](analytical_solution/monte_carlo.py#L4)),
- the paper PDF frames the work as latency of multipartite entanglement distribution in a quantum SDN architecture.

So the repository is currently split into two partially disconnected models:

| Layer | Actual modeled problem |
| --- | --- |
| RL code | Bipartite end-to-end entanglement over a repeater chain |
| Analytical code / notebooks / paper framing | Multipartite grid distribution with fusion and latency |

This mismatch is the single most important issue in the codebase.

Beyond that, the repository contains:

- a tabular Q-learning baseline over enumerated adjacency states ([`qamel/agent.py#L8`](qamel/agent.py#L8)),
- a DQN over tensor observations ([`qamel/dqn.py#L18`](qamel/dqn.py#L18)),
- analytical Markov and Monte Carlo utilities for a different problem definition,
- evaluation scripts and notebooks that export operation counts and derive latency estimates.

There are also several technical reliability issues:

- the baseline Q-learning exploration rule is wrong,
- the baseline terminal Q update is wrong,
- baseline saved models ignore `pgen` and `pswap`,
- CLI support for `counter_exposed_plus_ready` is broken,
- DQN target backups do not validity-mask next actions,
- DQN training reward and evaluation reward are not the same,
- evaluation success and terminal reward are not aligned,
- notebook workflows depend on missing files and hard-coded external paths.

## Repository overview

### High-level purpose of the project

Confirmed from the executable RL code, the project tries to learn a policy that minimizes the cost of producing an end-to-end entangled link across a stochastic repeater chain.

The chain consists of:

- nearest-neighbor elementary links that can be generated probabilistically,
- interior repeater nodes where entanglement swapping can be attempted probabilistically,
- a terminal goal of producing a direct entangled link between the two end nodes.

The analytical side of the repository broadens that to a grid-based multipartite setting with fusion and latency estimation. That broader goal is visible in:

- [`README.md#L7`](README.md#L7),
- [`analytical_solution/analytical_equations.py`](analytical_solution/analytical_equations.py),
- [`analytical_solution/monte_carlo.py`](analytical_solution/monte_carlo.py),
- [`notebooks/verify_simulations.ipynb`](notebooks/verify_simulations.ipynb),
- [`notebooks/calculate_ratio_and_latencies.ipynb`](notebooks/calculate_ratio_and_latencies.ipynb),
- [`OFC_paper_2025.pdf`](OFC_paper_2025.pdf).

### Scientific / computational problem

In the RL environment, the computational problem is:

1. Given a chain of `n` nodes,
2. where elementary entanglement generation succeeds with probability `pgen`,
3. and swapping at an interior repeater succeeds with probability `pswap`,
4. choose actions over time to obtain an end-to-end link efficiently.

The environment is stochastic because each generation and swap attempt is a Bernoulli trial ([`qamel/environment.py#L33`](qamel/environment.py#L33), [`qamel/environment.py#L50`](qamel/environment.py#L50)).

### Role of reinforcement learning

Reinforcement learning is used as a control policy over the operation schedule:

- when to request elementary link generation,
- when to trigger swaps,
- and which subsets of those operations to do in parallel at each step.

The tabular baseline uses enumerated adjacency states and a Q-table ([`qamel/agent.py#L45`](qamel/agent.py#L45)).

The DQN uses tensor observations derived from the environment state, either:

- adjacency only,
- or adjacency plus normalized operation counters,
- or adjacency plus counters plus a swap-readiness channel ([`qamel/dqn.py#L4`](qamel/dqn.py#L4)).

### What task the agent is trying to optimize

The task objective in the current code is not simply "minimum number of operations".

The implemented reward is:

- `-1` for each nonterminal step,
- `-100` for a bad/truncated state,
- `1000 / torch.amax(chain_state)` on terminal success ([`qamel/utils.py#L109`](qamel/utils.py#L109)).

Because `chain_state` includes operation-counter channels, terminal reward depends on the **maximum value anywhere in the state tensor**, which in practice is dominated by the largest generation/swap counter. So the agent is effectively optimizing a mix of:

- fewer elapsed steps,
- fewer peak local attempts,
- avoidance of bad states.

This differs from the README wording and also differs from the DQN training reward whenever swap bonuses are enabled.

## Full architecture of the codebase

### Main files and modules

| File | Role |
| --- | --- |
| [`README.md`](README.md) | Project framing and basic run instructions |
| [`qamel/environment.py`](qamel/environment.py) | Core repeater-chain dynamics |
| [`qamel/utils.py`](qamel/utils.py) | Action/state generation, validity checks, reward, terminal checks |
| [`qamel/agent.py`](qamel/agent.py) | Tabular Q-learning agent |
| [`qamel/dqn.py`](qamel/dqn.py) | DQN preprocessing and network |
| [`scripts/train_qamel.py`](scripts/train_qamel.py) | Training entry point |
| [`scripts/evaluate_qamel.py`](scripts/evaluate_qamel.py) | Evaluation entry point |
| [`analytical_solution/analytical_equations.py`](analytical_solution/analytical_equations.py) | Markov / analytical waiting-time formulas |
| [`analytical_solution/monte_carlo.py`](analytical_solution/monte_carlo.py) | Recursive Monte Carlo baseline |
| [`analytical_solution/utils.py`](analytical_solution/utils.py) | Grid-distance and utility helpers |
| [`notebooks/verify_simulations.ipynb`](notebooks/verify_simulations.ipynb) | Verification of analytical / simulation results |
| [`notebooks/calculate_ratio_and_latencies.ipynb`](notebooks/calculate_ratio_and_latencies.ipynb) | Latency post-processing from operation counts |
| [`notebooks/agent_latencies.py`](notebooks/agent_latencies.py) | Intended helper script for Qamel latency counts; currently broken |

### Execution flow from entry point to outputs

#### Training flow

1. Training starts in [`scripts/train_qamel.py#L286`](scripts/train_qamel.py#L286).
2. The script parses `--n`, `--pgen`, `--pswap`, `--model_tag`, and optional DQN-specific flags ([`scripts/train_qamel.py#L291`](scripts/train_qamel.py#L291)).
3. It chooses between:
   - tabular baseline via `train_q_agent()` ([`scripts/train_qamel.py#L24`](scripts/train_qamel.py#L24)),
   - DQN via `train_dqn_agent()` ([`scripts/train_qamel.py#L140`](scripts/train_qamel.py#L140)).
4. Both instantiate `RepeaterChain` with `n`, `pgen`, `pswap` ([`scripts/train_qamel.py#L40`](scripts/train_qamel.py#L40), [`scripts/train_qamel.py#L154`](scripts/train_qamel.py#L154)).
5. Each episode starts from `env.reset()`, which returns a zero tensor of shape `(3, n, n)` ([`qamel/environment.py#L10`](qamel/environment.py#L10)).
6. Each step:
   - chooses an action,
   - calls `env.step(state, action)`,
   - computes `bad_state`, `final_state`, and reward,
   - updates the learner,
   - stops on terminality or step limit.
7. Outputs are saved as:
   - baseline Q-table text files in `qamel/q_table_storage/`,
   - DQN checkpoints in `qamel/outputs/models/`.

#### Evaluation flow

1. Evaluation starts in [`scripts/evaluate_qamel.py#L279`](scripts/evaluate_qamel.py#L279).
2. Depending on `obs_mode`, it loads either:
   - a Q-table and enumerated states,
   - or a DQN checkpoint and action basis.
3. It rolls out `eval_episodes` episodes in the same environment.
4. It records:
   - `final_state`,
   - `bad_state`,
   - `steps`,
   - `total_return`,
   - `ent_attempt_max`,
   - `swap_attempt_max`.
5. It writes:
   - CSV summaries in `qamel/outputs/results/`,
   - `ent_counts/{n}_nodes.txt`,
   - `swap_counts/{n}_nodes.txt`.

### How data, states, actions, rewards, and metrics move through the system

| Item | Representation | Produced by | Consumed by |
| --- | --- | --- | --- |
| State | `(3, n, n)` tensor | `RepeaterChain.reset/step` | agent, reward, evaluation |
| Action | `(n, n)` matrix | tabular policy or DQN | `RepeaterChain.step` |
| Reward | scalar | `reward_shape()` plus optional DQN bonus | training loops |
| Metrics | arrays of returns, steps, counters, success flags | evaluation loop | CSV export, notebooks |
| Analytical counts | expected wait-time / attempt arrays | analytical code | notebooks only |

## Physics / domain model

### Physical model implemented in the RL code

The RL environment models:

- a **linear repeater chain**,
- nearest-neighbor elementary entanglement generation,
- probabilistic entanglement swapping at interior nodes,
- persistent links until consumed,
- no explicit noise, no decoherence, no memory lifetime, and no fidelity.

The physical state of the chain is represented only as **binary link presence/absence** in `state[0]` ([`qamel/environment.py#L30`](qamel/environment.py#L30), [`qamel/environment.py#L34`](qamel/environment.py#L34), [`qamel/environment.py#L51`](qamel/environment.py#L51)).

### Repeater-chain and swapping logic

The implemented logic in [`qamel/environment.py#L14`](qamel/environment.py#L14) is:

1. For each acted-on nearest-neighbor edge:
   - increment generation counters on both endpoint nodes,
   - clear the current edge,
   - re-sample its presence with probability `pgen`.
2. For each acted-on interior repeater node:
   - inspect the currently connected neighboring nodes,
   - if exactly two links are present, increment that node's swap counter,
   - with probability `pswap`, create a direct edge between those two neighbors,
   - remove the two links incident to the swapping node.

This produces the expected repeater-chain contraction pattern. In a deterministic check with `n=5`, acting on all four elementary segments followed by swaps at nodes `1`, `3`, then `2` produces the final end-to-end edge `0 <-> 4`.

### Meaning of generation probabilities and swap probabilities

| Parameter | Code meaning |
| --- | --- |
| `pgen` | success probability of a generation attempt on an acted-on elementary segment ([`qamel/environment.py#L33`](qamel/environment.py#L33)) |
| `pswap` | success probability of a swap attempt on an acted-on repeater node ([`qamel/environment.py#L50`](qamel/environment.py#L50)) |

Both are treated as independent Bernoulli events.

### Fidelities, memory, lifetime, latency

These are **not modeled** in the RL environment:

- fidelity,
- decoherence,
- memory expiration,
- storage lifetime,
- classical signaling delay,
- teleportation latency.

The only time-like quantity in the RL environment is discrete step count.

### Stochastic processes

Stochasticity in the RL environment is limited to:

- generation success/failure,
- swap success/failure.

The analytical code separately uses:

- geometric waiting times for generation,
- recursive swap retry logic,
- optional fusion success sampling ([`analytical_solution/monte_carlo.py#L41`](analytical_solution/monte_carlo.py#L41)).

### Physical assumptions vs RL/control assumptions

#### Physical modeling assumptions

- chain topology,
- nearest-neighbor elementary links,
- probabilistic swap,
- link persistence until consumed,
- binary links instead of continuous fidelity.

#### RL/control assumptions

- an action can include multiple parallel generation requests and multiple swaps,
- rewards penalize steps and peak local operation count rather than a direct physical latency,
- the DQN can observe counters while the baseline cannot,
- optional swap-readiness shaping reward is introduced only during DQN training.

## Environment and MDP design

### State representation in detail

`RepeaterChain.reset()` returns `torch.zeros(size=(3, n, n))` ([`qamel/environment.py#L10`](qamel/environment.py#L10)).

| Channel | Meaning |
| --- | --- |
| `state[0]` | Current binary adjacency matrix of available entanglement links |
| `state[1]` | Per-node generation-attempt counters stored on the diagonal |
| `state[2]` | Per-node swap-attempt counters stored on the diagonal |

Evaluation later reduces channels `1` and `2` using `torch.amax`, so exported counts are maxima, not totals ([`scripts/evaluate_qamel.py#L160`](scripts/evaluate_qamel.py#L160), [`scripts/evaluate_qamel.py#L161`](scripts/evaluate_qamel.py#L161)).

### Action space in detail

The action generator in [`qamel/utils.py#L4`](qamel/utils.py#L4) constructs matrices with:

- free interior diagonal bits for swap decisions,
- free one-offset diagonals for nearest-neighbor generation decisions,
- forced zeros on the first and last diagonal entries.

Then `generate_all_valid_actions()` removes structurally invalid combinations where a node marked for swap also has edge actions in its row/column ([`qamel/utils.py#L27`](qamel/utils.py#L27)).

Measured valid-action counts from direct execution:

| `n` | Valid actions |
| --- | ---: |
| 3 | 5 |
| 4 | 13 |
| 5 | 34 |
| 6 | 89 |
| 7 | 233 |

### Transition rules step by step

Equivalent pseudocode for `step()`:

```text
clone current state

for each acted-on upper-triangular edge:
    increment generation counters at both endpoints
    clear the edge
    with probability pgen:
        set the edge to 1

for each acted-on interior swap node:
    connected_nodes = current neighbors in state[0]
    if degree > 2:
        return (-100, state_copy)   # inconsistent return type
    if degree == 2:
        increment swap counter at node
        with probability pswap:
            create direct edge between the two neighbors
        remove both edges incident to the swap node

return updated state
```

### Full episode progression

Common training/evaluation episode logic:

```text
state = env.reset()
done = False
while not done:
    action = policy(state)
    next_state = env.step(state, action)
    bad = check_if_bad_state(next_state)
    final = check_if_final_state(next_state)
    reward = reward_shape(next_state, final, bad)
    learner_update(...)
    state = next_state
    stop if final or bad or timeout
```

### Terminal conditions

| Condition | Code | Meaning |
| --- | --- | --- |
| Final | [`qamel/utils.py#L89`](qamel/utils.py#L89) | direct link between node `0` and node `n-1` |
| Bad | [`qamel/utils.py#L96`](qamel/utils.py#L96) | endpoint degree exceeds `1` or interior degree exceeds `2` |
| Timeout | [`scripts/train_qamel.py#L65`](scripts/train_qamel.py#L65), [`scripts/evaluate_qamel.py#L153`](scripts/evaluate_qamel.py#L153) | `steps >= max_actions` |

### What “success” means

There are two definitions in play:

1. Environment success: `final_state == True`.
2. Reported evaluation success: `final_state and not bad_state` ([`scripts/evaluate_qamel.py#L266`](scripts/evaluate_qamel.py#L266)).

Those are not equivalent. In the provided saved baseline CSV, 26 episodes are simultaneously `final_state=True` and `bad_state=True`.

### Rewards

`reward_shape()` is:

- terminal: `1000 / torch.amax(chain_state).item()`,
- bad/truncated: `-100`,
- otherwise: `-1` ([`qamel/utils.py#L109`](qamel/utils.py#L109)).

Important implication: because `torch.amax` is taken over the full 3-channel tensor, terminal reward is effectively based on the maximum local attempt counter anywhere in the state, not total operations.

### Observation / reward mismatch

The baseline tabular learner chooses actions from `current_state[0]` only ([`scripts/train_qamel.py#L50`](scripts/train_qamel.py#L50)).

But reward depends on channels `1` and `2`.

Therefore the baseline observation is **not Markov** with respect to the reward actually used.

The DQN with `counter_exposed` is closer to the true state because it sees the counter channels.

## RL methods implemented

### Tabular Q-learning implementation

The tabular learner is `Agent` in [`qamel/agent.py`](qamel/agent.py).

#### Data structures

- `self.all_states`: enumerated valid adjacency states ([`qamel/agent.py#L31`](qamel/agent.py#L31)),
- `self.all_actions`: enumerated valid action matrices ([`qamel/agent.py#L43`](qamel/agent.py#L43)),
- `self.q_table`: zero-initialized Q-values of shape `(num_states, num_actions)` ([`qamel/agent.py#L45`](qamel/agent.py#L45)).

For the checked-in `n=5` artifacts:

- states shape: `(88, 5, 5)`,
- actions shape: `(34, 5, 5)`,
- Q-table shape: `(88, 34)`.

#### Action selection

`predict_action()` is intended to be epsilon-greedy ([`qamel/agent.py#L50`](qamel/agent.py#L50)) but currently uses:

```python
explore = True if torch.randn(1) < epsilon else False
```

That is incorrect. It should use uniform random sampling, not a standard normal draw.

It also does not mask state-invalid actions.

#### Q update

`update_q_table()` at [`qamel/agent.py#L60`](qamel/agent.py#L60) uses:

- nonterminal:
  - `Q <- Q + alpha * (reward + gamma * max Q(next) - Q)`
- terminal/bad:
  - `Q <- Q + alpha * (reward + Q)`

The terminal branch is mathematically wrong. Standard terminal handling should subtract the current Q-value, not add it.

### DQN implementation

The DQN training path is `train_dqn_agent()` in [`scripts/train_qamel.py#L140`](scripts/train_qamel.py#L140).

#### Observation preprocessing

Implemented in [`qamel/dqn.py#L4`](qamel/dqn.py#L4).

Modes:

| Mode | Input shape |
| --- | --- |
| `baseline` | `(n, n)` adjacency only |
| `counter_exposed` | `(3, n, n)` full state with normalized counters |
| `counter_exposed_plus_ready` | `(4, n, n)` full state plus readiness channel |

The readiness channel is a constant plane filled with the count of interior nodes whose degree is `2` ([`qamel/dqn.py#L10`](qamel/dqn.py#L10)).

#### Network architecture

`DQNNet` in [`qamel/dqn.py#L18`](qamel/dqn.py#L18):

- Flatten
- Linear(input, 512)
- ReLU
- Linear(512, 512)
- ReLU
- Linear(512, num_actions)

Measured for `n=5`:

| Mode | Input dim | Output dim | Parameters |
| --- | ---: | ---: | ---: |
| `counter_exposed` | 75 | 34 | 319,010 |
| `counter_exposed_plus_ready` | 100 | 34 | 331,810 |

#### Replay buffer

Implemented as a `deque` in [`scripts/train_qamel.py#L88`](scripts/train_qamel.py#L88).

Each entry stores:

- observation,
- action index,
- reward,
- next observation,
- done flag.

#### Target network logic

A target network is created as a copy of the policy network and refreshed every `target_update_steps = 1000` environment steps ([`scripts/train_qamel.py#L172`](scripts/train_qamel.py#L172), [`scripts/train_qamel.py#L273`](scripts/train_qamel.py#L273)).

#### Exploration strategy

DQN correctly uses a linear epsilon schedule via `linear_schedule()` ([`qamel/utils.py#L105`](qamel/utils.py#L105), [`scripts/train_qamel.py#L214`](scripts/train_qamel.py#L214)).

Hyperparameters in `dqn_hyperparameters`:

- `gamma = 0.99`,
- `lr = 1e-3`,
- `batch_size = 64`,
- `buffer_size = 50000`,
- `target_update_steps = 1000`,
- `eps_start = 1.0`,
- `eps_end = 0.05`,
- `eps_decay_steps = 10000`,
- `counter_norm = 20.0`.

#### Action masking

At acting time, DQN computes valid action indices with `_get_valid_action_indices()` and sets invalid Q-values to `-1e9` before `argmax` ([`scripts/train_qamel.py#L221`](scripts/train_qamel.py#L221), [`scripts/train_qamel.py#L228`](scripts/train_qamel.py#L228)).

However, the target backup uses:

```python
next_q = target_net(next_obs_batch).max(1, keepdim=True)[0]
```

without validity masking ([`scripts/train_qamel.py#L264`](scripts/train_qamel.py#L264)). This can overestimate impossible next actions.

#### Loss function

The code uses MSE loss ([`scripts/train_qamel.py#L268`](scripts/train_qamel.py#L268)) and Adam optimization ([`scripts/train_qamel.py#L177`](scripts/train_qamel.py#L177)).

There is no:

- Huber loss,
- Double DQN,
- gradient clipping,
- prioritized replay.

#### Additional reward shaping

DQN training optionally adds:

```text
swap_ready_bonus * ready_nodes
```

if the chosen action is swap-ready ([`scripts/train_qamel.py#L240`](scripts/train_qamel.py#L240)).

Evaluation does **not** add this bonus, so training and evaluation optimize/report different returns.

### What changed and what stayed the same between tabular and DQN

#### Same

- same environment,
- same action basis,
- same terminal logic,
- same base reward function.

#### Different

| Aspect | Tabular | DQN |
| --- | --- | --- |
| Observation | adjacency only | tensor observation, optionally includes counters |
| Function approximator | Q-table | MLP |
| Exploration | intended epsilon-greedy, but bugged | linear epsilon-greedy |
| Action validity | not masked | masked during acting only |
| Learning | direct Q-table update | replay + target network |

## Evaluation and metrics

### Evaluation scripts and notebooks

| File | Purpose |
| --- | --- |
| [`scripts/evaluate_qamel.py`](scripts/evaluate_qamel.py) | Main policy evaluation and CSV export |
| [`notebooks/verify_simulations.ipynb`](notebooks/verify_simulations.ipynb) | Analytical / Monte Carlo verification |
| [`notebooks/calculate_ratio_and_latencies.ipynb`](notebooks/calculate_ratio_and_latencies.ipynb) | Count-to-latency conversion and percent-improvement plots |
| [`notebooks/agent_latencies.py`](notebooks/agent_latencies.py) | Intended helper for Qamel counts, but currently broken |

### Metrics reported by evaluation script

CSV headers in [`scripts/evaluate_qamel.py#L246`](scripts/evaluate_qamel.py#L246):

- `episode`
- `final_state`
- `bad_state`
- `steps`
- `total_return`
- `ent_attempt_max`
- `swap_attempt_max`

Summary metrics:

- success rate,
- mean and std of steps,
- mean and std of entanglement attempts,
- mean and std of swap attempts,
- mean and std of total return ([`scripts/evaluate_qamel.py#L266`](scripts/evaluate_qamel.py#L266)).

### How those metrics are computed

| Metric | Code path |
| --- | --- |
| `final_state` | `check_if_final_state(current_state)` |
| `bad_state` | `check_if_bad_state(current_state)` |
| `steps` | incremented each environment step |
| `total_return` | cumulative sum of per-step rewards |
| `ent_attempt_max` | `torch.amax(current_state[1]).item()` |
| `swap_attempt_max` | `torch.amax(current_state[2]).item()` |
| `success_rate` | `mean(final_state && !bad_state)` |

### Meaning of “total return” in this codebase

`total_return` in evaluation is the accumulated sum of:

- `-1` for each nonterminal step,
- plus a terminal reward `1000 / max(counter)`,
- or `-100` if the episode is bad/timeout.

For DQN models trained with `swap_ready_bonus`, evaluation return is not the same objective the policy trained against.

### Saved evaluation results present in the repository

Using the checked-in CSV files in `qamel/outputs/results/`:

| Result file | Episodes | Success rate | Mean steps | Mean return |
| --- | ---: | ---: | ---: | ---: |
| `eval_n5_pgen0.4_pswap0.7_baseline.csv` | 100 | 0.73 | 13.15 | 89.60 |
| `eval_n5_pgen0.4_pswap0.7_counter_exposed.csv` | 50 | 0.10 | 97.42 | -95.25 |
| `eval_n5_pgen0.4_pswap0.7_m1_b0p2.csv` | 200 | 0.00 | 100.00 | -100.00 |
| `eval_n5_pgen0.4_pswap0.7_m1_b0p5.csv` | 100 | 0.00 | 100.00 | -100.00 |
| `eval_n5_pgen0.4_pswap0.7_m2_ready.csv` | 200 | 0.725 | 13.66 | 99.25 |
| `eval_n5_pgen0.4_pswap0.7_m3_on.csv` | 200 | 0.745 | 14.46 | 86.90 |

These confirm that the unmodified `counter_exposed` DQN underperforms the baseline badly, while modified variants recover.

### Benchmarking against previous work

The benchmarking story in the notebooks is mostly analytical:

- [`notebooks/verify_simulations.ipynb`](notebooks/verify_simulations.ipynb) compares Monte Carlo histograms to `bernardes_eq`, `markov_approach`, and `including_fusion`.
- [`notebooks/calculate_ratio_and_latencies.ipynb`](notebooks/calculate_ratio_and_latencies.ipynb) converts operation-count statistics into classical and quantum latency contributions using fixed operation times and fiber-propagation assumptions.

This benchmark pipeline is not directly coupled to the RL environment definition.

## How to run the code

### Dependencies

From [`requirements.txt`](requirements.txt):

- `numpy`
- `torch`
- `matplotlib`
- `rich`
- `scipy`

Notebook dependencies not listed there but used by notebooks:

- `networkx`
- `seaborn`

### Suggested setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install networkx seaborn notebook
```

### Baseline training

```bash
python scripts/train_qamel.py --n 5 --pgen 0.4 --pswap 0.7 --model_tag baseline
```

### DQN training

As currently written, the CLI only reliably reaches the DQN branch for `counter_exposed`:

```bash
python scripts/train_qamel.py \
  --n 5 --pgen 0.4 --pswap 0.7 \
  --obs_mode counter_exposed \
  --model_tag counter_exposed \
  --train_episodes 10000
```

### Baseline evaluation

```bash
python scripts/evaluate_qamel.py \
  --n 5 --pgen 0.4 --pswap 0.7 \
  --eval_episodes 100 \
  --model_tag baseline \
  --obs_mode baseline
```

### DQN evaluation

```bash
python scripts/evaluate_qamel.py \
  --n 5 --pgen 0.4 --pswap 0.7 \
  --eval_episodes 100 \
  --model_tag counter_exposed \
  --obs_mode counter_exposed
```

### Hidden assumptions / missing pieces

- baseline Q-tables are keyed only by `n`, not by `pgen`/`pswap`,
- some saved evaluation outputs do not have corresponding saved checkpoints,
- `counter_exposed_plus_ready` is listed but not properly wired through CLI logic,
- notebook workflows depend on missing result files and absolute external paths,
- no version pinning,
- no seed control,
- no tests.

## Problems and limitations

### Scientific / modeling inconsistencies

1. **RL code is a chain, not a grid.**
   - README and analytical notebooks describe a different problem from the executable RL environment.

2. **No fidelity or decoherence modeling.**
   - This is a major omission if the scientific claim is physical entanglement-distribution realism.

3. **Latency is not part of the RL environment.**
   - Notebook latency models are post hoc and external.

### RL / implementation issues

4. **Baseline exploration is wrong.**
   - [`qamel/agent.py#L52`](qamel/agent.py#L52) uses `torch.randn` instead of uniform `rand`.

5. **Baseline terminal Q update is wrong.**
   - [`qamel/agent.py#L65`](qamel/agent.py#L65) adds `reward + Q` instead of `reward - Q`.

6. **Baseline file naming ignores environment parameters.**
   - [`scripts/train_qamel.py#L365`](scripts/train_qamel.py#L365) stores by `n` only.

7. **Step timeout is not consistently stored as terminal in DQN replay.**
   - Replay `done` is written before timeout check ([`scripts/train_qamel.py#L246`](scripts/train_qamel.py#L246), [`scripts/train_qamel.py#L276`](scripts/train_qamel.py#L276)).

8. **Bellman targets are not validity-masked.**
   - [`scripts/train_qamel.py#L264`](scripts/train_qamel.py#L264)

9. **Training reward and evaluation reward differ for DQN.**
   - `swap_ready_bonus` appears only in training.

10. **Success metric and reward are inconsistent.**
   - `final_state=True, bad_state=True` can still receive terminal reward.

11. **Tabular observation is non-Markov for the implemented reward.**
   - adjacency-only state IDs do not include counters.

12. **Existing links can be re-attempted.**
   - validity logic only counts absent new edges, but `step()` clears and resamples acted-on links.

13. **`step()` has an inconsistent return type on one branch.**
   - [`qamel/environment.py#L44`](qamel/environment.py#L44) returns `(-100, state_copy)` instead of a state tensor.

14. **CLI handling for `counter_exposed_plus_ready` is broken.**
   - training branch only checks for exact string `"counter_exposed"` ([`scripts/train_qamel.py#L316`](scripts/train_qamel.py#L316)),
   - evaluation remaps unsupported modes back to baseline ([`scripts/evaluate_qamel.py#L299`](scripts/evaluate_qamel.py#L299)).

### Reproducibility issues

15. **Missing notebook dependencies in `requirements.txt`.**
16. **Hard-coded Windows paths in notebooks.**
17. **Broken helper script `notebooks/agent_latencies.py`.**
18. **No tests.**
19. **No seed control.**
20. **No pinned versions.**

### Scaling issues

21. **State enumeration does not scale.**
   - `generate_all_states()` enumerates `2^(n choose 2)` candidate graphs in memory ([`qamel/utils.py#L51`](qamel/utils.py#L51)).

22. **Action space grows quickly.**
   - measured valid-action growth: `5, 13, 34, 89, 233` for `n=3..7`.

23. **Environment stepping is scalar and Python-loop based.**
   - not suitable for large-scale RL rollout throughput.

## Scaling and next steps

### What would be required to run reliably on GPU / HPC

The scripts already select CUDA if available ([`scripts/train_qamel.py#L287`](scripts/train_qamel.py#L287), [`scripts/evaluate_qamel.py#L204`](scripts/evaluate_qamel.py#L204)), but the environment is still mostly CPU-bound.

To scale:

1. Vectorize environment stepping across many episodes.
2. Replace Python loops in `step()` and validity checking with batched tensor logic.
3. Move replay sampling and preprocessing fully onto device.
4. Add distributed rollout workers if using HPC.
5. Save full run configs and seeds for cluster reproducibility.

### CPU-bound parts

- action-validity enumeration each step,
- environment stepping loops,
- exhaustive state/action generation,
- notebook post-processing.

### Parts that could benefit from batching/vectorization

- parallel environment rollouts,
- validity masking over action batches,
- next-state target computation,
- counter preprocessing for DQN input.

### What likely breaks with more repeater nodes

- tabular state enumeration first,
- baseline artifact naming collisions second,
- DQN output head size third,
- acting-time action masking cost fourth.

### Concrete next steps

1. Decide on one scientific problem definition and align README, code, and notebooks.
2. Fix baseline exploration and Q updates.
3. Make reward, success, and evaluation consistent.
4. Mask invalid next actions in DQN targets.
5. Fix CLI handling for observation modes.
6. Parameterize output filenames by environment and model config.
7. Add tests and deterministic seeds.
8. Only after that, start scaling or extending the physics.

## Minimal changes needed to get this code working reliably

Priority order:

1. **Fix the project scope mismatch.**
   - Either relabel the RL code as repeater-chain Bell-pair scheduling,
   - or implement the missing grid/fusion environment.

2. **Fix baseline correctness.**
   - uniform epsilon-greedy,
   - correct terminal Q update,
   - state-valid action masking.

3. **Fix reward / metric consistency.**
   - unify training and evaluation reward,
   - decide whether `final && bad` is success or failure,
   - decide whether the objective is steps, total operations, max local load, or latency.

4. **Fix DQN backup logic.**
   - validity-mask next-state actions when computing targets.

5. **Fix artifact naming.**
   - include `n`, `pgen`, `pswap`, `obs_mode`, and `model_tag` in saved outputs.

6. **Fix CLI mode support.**
   - make `counter_exposed_plus_ready` usable end-to-end.

7. **Repair notebook pipeline.**
   - remove absolute paths,
   - regenerate missing files from scripts,
   - fix `notebooks/agent_latencies.py`.

8. **Add reproducibility basics.**
   - seeds,
   - tests,
   - pinned package versions,
   - saved config metadata.

## Appendix: file-by-file breakdown

### [`README.md`](README.md)

- States the project is a model-free RL agent for `n`-node grid topology and multipartite entangled distribution.
- This is broader than the executable RL code.

### [`requirements.txt`](requirements.txt)

- Lists only `numpy`, `torch`, `matplotlib`, `rich`, `scipy`.
- Missing notebook dependencies.

### [`qamel/environment.py`](qamel/environment.py)

- Defines `RepeaterChain`.
- Holds all executable RL environment dynamics.
- Implements generation and swap state transitions.
- Contains one inconsistent-return bug path.

### [`qamel/utils.py`](qamel/utils.py)

- Generates action and state spaces.
- Defines validity checks and terminal checks.
- Defines the reward.
- Encodes the real optimization target.

### [`qamel/agent.py`](qamel/agent.py)

- Baseline tabular learner.
- Loads/saves enumerated states and actions.
- Contains exploration and terminal-update bugs.
- Uses adjacency-only state IDs.

### [`qamel/dqn.py`](qamel/dqn.py)

- Converts environment state into DQN observation tensors.
- Defines a 2-layer MLP DQN.

### [`scripts/train_qamel.py`](scripts/train_qamel.py)

- Main training entry point.
- Contains both tabular and DQN training loops.
- Includes replay buffer, epsilon schedule, target network, and optional curriculum.

### [`scripts/evaluate_qamel.py`](scripts/evaluate_qamel.py)

- Main evaluation driver.
- Loads either tabular or DQN models.
- Exports CSV metrics and per-episode max counter files.

### [`analytical_solution/utils.py`](analytical_solution/utils.py)

- Helper math and grid-distance routines for the analytical side.

### [`analytical_solution/analytical_equations.py`](analytical_solution/analytical_equations.py)

- Transition-matrix and closed-form waiting-time models.
- Uses a grid / central-node / fusion framing rather than the chain RL framing.

### [`analytical_solution/monte_carlo.py`](analytical_solution/monte_carlo.py)

- Recursive Monte Carlo baseline for generation, swap, and fusion counts.
- Not directly integrated into the RL environment.

### [`notebooks/verify_simulations.ipynb`](notebooks/verify_simulations.ipynb)

- Reproduces comparisons between Monte Carlo and analytical formulas.
- Supports the paper-facing analytical story.

### [`notebooks/calculate_ratio_and_latencies.ipynb`](notebooks/calculate_ratio_and_latencies.ipynb)

- Converts operation-count estimates into classical and quantum latency contributions.
- Depends on missing precomputed Qamel result files and hard-coded absolute paths.

### [`notebooks/agent_latencies.py`](notebooks/agent_latencies.py)

- Intended to produce Qamel count summaries for notebook use.
- Currently broken due to API mismatch and indentation error.

### [`OFC_paper_2025.pdf`](OFC_paper_2025.pdf)

- Frames the overall research as multipartite entanglement distribution latency in quantum SDN.
- Scientific scope is broader than the RL environment currently checked in.
