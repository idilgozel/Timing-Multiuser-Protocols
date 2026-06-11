import numpy as np
import torch
from itertools import product

def generate_all_actions(n):

    num_free_diagonal = n - 2        
    num_one_offset = n - 1  

    combinations = list(product([0, 1], repeat=num_free_diagonal + num_one_offset))

    matrices = torch.zeros((len(combinations), n, n), dtype=torch.float32)

    for idx, comb in enumerate(combinations):
        diag_elements = [0] + list(comb[:num_free_diagonal]) + [0]  # Force first and last diagonal to be 0
        one_offset_elements = comb[num_free_diagonal:]  # One-offset diagonal  

        # Convert to tensors
        diag_tensor = torch.tensor(diag_elements, dtype=torch.float32)
        one_offset_tensor = torch.tensor(one_offset_elements, dtype=torch.float32)

        matrices[idx].diagonal().copy_(diag_tensor)  # Set diagonal
        matrices[idx].diagonal(1).copy_(one_offset_tensor)  # Set one-offset diagonal
        matrices[idx].diagonal(-1).copy_(one_offset_tensor)  # Mirror for symmetry

    return matrices

def generate_all_valid_actions(n):

    matrices = generate_all_actions(n)

    valid_matrices = []

    for matrix in matrices:
        valid = True
        n = matrix.shape[0]

        for i in range(n):
            if matrix[i, i] == 1:
                # If diagonal is 1, row i and column i (excluding M[i, i]) must be zero
                if torch.any(matrix[i, :i]) or torch.any(matrix[i, i+1:]) or \
                   torch.any(matrix[:i, i]) or torch.any(matrix[i+1:, i]):
                    valid = False
                    break

        if valid:
            valid_matrices.append(matrix)

    return torch.stack(valid_matrices) if valid_matrices else torch.empty((0, n, n), dtype=torch.float32)


def generate_all_states(n):
    total_edges = int(((n)*(n-1))/2)

    combinations = list(product(range(2), repeat=total_edges))

    matrices = torch.zeros((len(combinations), n, n), dtype=torch.float32)

    for idx, comb in enumerate(combinations):
        this_matrix = torch.zeros(size = (n, n))
        indexes_upper = torch.triu_indices(n, n, 1)
        this_matrix[indexes_upper[0], indexes_upper[1]] = torch.Tensor(comb)
        
        this_matrix += torch.rot90(torch.fliplr(this_matrix))

        matrices[idx] = this_matrix

    return matrices


def generate_all_valid_states(n):
    all_states = generate_all_states(n)

    all_valid_states = []
    for state in all_states:
        good_state = True
        for node in range(n):
            if (node == 0 or node == n-1):
                if len(state[node].nonzero()) > 1:
                    good_state = False
            else:
                if len(state[node].nonzero()) > 2:
                    good_state = False

        if good_state:
            all_valid_states.append(state)

    return torch.stack(all_valid_states)

def check_if_final_state(state) -> bool:
    chain_state = state[0]
    if (chain_state[0, -1] != 0 and chain_state[-1, 0] != 0):
        return True
    else:
        return False
    
def check_if_bad_state(state: torch.Tensor) -> bool:
    chain_state = state[0]
    if (torch.count_nonzero(chain_state[-1]).item() > 1 or 
        torch.count_nonzero(chain_state[0]).item() > 1 or
        (torch.count_nonzero(chain_state[1:-1], dim = 1) > 2).any()):
        return True
    else:
        return False

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def reward_shape(chain_state, terminated, truncated):
    if terminated:
        return 100.0
    elif truncated:
        return -100.0
    else:
        return -1.0

def chain_progress_potential(state: np.ndarray, n: int) -> int:
    """
    Potential function Phi(s) for PBRS on a repeater chain.

    Returns the length of the longest contiguous run of present
    elementary links starting at node 0 in the adjacency channel s[0].

    Concretely: the largest k such that s[0][i, i+1] == 1
    for all i in 0..k-1. If s[0][0,1] == 0 then Phi = 0.
    Range: 0 .. n-1.

    This is a valid potential (bounded, depends only on s, not on a)
    and therefore PBRS with F = gamma * Phi(s') - Phi(s) preserves
    the optimal policy.
    """
    for i in range(n - 1):
        if state[0][i, i + 1] != 1:
            return i
    return max(0, n - 1)

def chain_progress_potential_batch(states: torch.Tensor) -> torch.Tensor:
    """Return Phi(s) for batched states shaped (batch, 3, n, n)."""
    if states.dim() != 4:
        raise ValueError(f"states must have shape (batch, 3, n, n), got {tuple(states.shape)}")
    if states.size(1) < 1 or states.size(2) != states.size(3):
        raise ValueError(f"states must have shape (batch, 3, n, n), got {tuple(states.shape)}")

    batch_size = states.size(0)
    n = states.size(2)
    if n <= 1:
        return torch.zeros(batch_size, dtype=torch.long, device=states.device)

    idx = torch.arange(n - 1, device=states.device)
    elementary_links = states[:, 0, idx, idx + 1].eq(1)
    return elementary_links.to(torch.long).cumprod(dim=1).sum(dim=1)

def test_chain_progress_potential():
    n = 5
    expected = [0, 1, 2, 3, 4]
    states = []

    for phi in expected:
        state = np.zeros((3, n, n), dtype=np.float32)
        for i in range(phi):
            state[0, i, i + 1] = 1
            state[0, i + 1, i] = 1
        if phi + 1 < n - 1:
            state[0, n - 2, n - 1] = 1
            state[0, n - 1, n - 2] = 1
        states.append(state)

    for state, phi in zip(states, expected):
        assert chain_progress_potential(state, n) == phi

    torch_states = torch.from_numpy(np.stack(states, axis=0))
    actual = chain_progress_potential_batch(torch_states)
    expected_tensor = torch.tensor(expected, dtype=torch.long)
    assert torch.equal(actual, expected_tensor), (actual, expected_tensor)

def count_swap_ready_nodes(state0, action_matrix):
    n = state0.size(0)
    degrees = torch.count_nonzero(state0, dim=1)

    new_edges = (action_matrix > 0) & (state0 == 0)
    upper = torch.triu(torch.ones_like(state0), diagonal=1).bool()
    new_edges = new_edges & upper

    add = torch.zeros(n, device=state0.device, dtype=degrees.dtype)
    idx = new_edges.nonzero(as_tuple=False)
    if idx.numel() > 0:
        add.index_add_(0, idx[:, 0], torch.ones(idx.size(0), device=state0.device, dtype=degrees.dtype))
        add.index_add_(0, idx[:, 1], torch.ones(idx.size(0), device=state0.device, dtype=degrees.dtype))

    new_degrees = degrees + add
    swap_nodes = torch.diagonal(action_matrix, 0)
    if swap_nodes.numel() > 2:
        swap_nodes = swap_nodes[1:-1]
    swap_idx = (swap_nodes > 0).nonzero(as_tuple=True)[0]
    if swap_idx.numel() == 0:
        return 0

    return int(torch.sum(new_degrees[swap_idx + 1] == 2).item())

def compute_reward(
    chain_state,
    terminated,
    truncated,
    reward_mode="base",
    swap_ready_bonus=0.0,
    prev_state0=None,
    action_matrix=None,
):
    if reward_mode != "base":
        raise ValueError(
            f"Unsupported reward_mode='{reward_mode}'. Supported reward modes: ['base']"
        )
    return reward_shape(chain_state, terminated, truncated)

def get_episode_status(state, steps, max_actions):
    """Return normalized episode flags after a transition.

    A state only counts as final/successful if it reaches the end-to-end link
    without simultaneously violating the degree constraints.
    """
    bad_state = check_if_bad_state(state)
    final_state = check_if_final_state(state) and not bad_state
    terminated = final_state or bad_state
    truncated = (steps >= max_actions) and not terminated
    done = terminated or truncated
    return {
        "final_state": final_state,
        "bad_state": bad_state,
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
    }

def is_action_valid_given_state(state0, action_matrix):
    n = state0.size(0)
    degrees = torch.count_nonzero(state0, dim=1)

    new_edges = (action_matrix > 0) & (state0 == 0)
    upper = torch.triu(torch.ones_like(state0), diagonal=1).bool()
    new_edges = new_edges & upper

    add = torch.zeros(n, device=state0.device, dtype=degrees.dtype)
    idx = new_edges.nonzero(as_tuple=False)
    if idx.numel() > 0:
        add.index_add_(0, idx[:, 0], torch.ones(idx.size(0), device=state0.device, dtype=degrees.dtype))
        add.index_add_(0, idx[:, 1], torch.ones(idx.size(0), device=state0.device, dtype=degrees.dtype))

    new_degrees = degrees + add
    limits = torch.full((n,), 2, device=state0.device, dtype=degrees.dtype)
    limits[0] = 1
    limits[-1] = 1

    swap_nodes = torch.diagonal(action_matrix, 0)
    if swap_nodes.numel() > 2:
        swap_nodes = swap_nodes[1:-1]
    swap_idx = (swap_nodes > 0).nonzero(as_tuple=True)[0]
    if swap_idx.numel() > 0:
        swap_degrees = new_degrees[swap_idx + 1]
        if not torch.all(swap_degrees == 2):
            return False

    return torch.all(new_degrees <= limits).item()


def get_operation_count(states):
    ent_count = []; swap_count = []
    for state in states:
        ent_count.append(torch.amax(state[1]).item())
        swap_count.append(torch.amax(state[2]).item())

    return ent_count, swap_count

def get_state_id(all_states, state):
    matches = torch.all(torch.isclose(all_states, state), dim=(1, 2))
    matching_indices = torch.nonzero(matches, as_tuple=True)[0]
    return matching_indices
        
def get_action_id(all_actions, action):
    matches = torch.all(torch.isclose(all_actions, action), dim=(1, 2))
    matching_indices = torch.nonzero(matches, as_tuple=True)[0]
    return matching_indices
