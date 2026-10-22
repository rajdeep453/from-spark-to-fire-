"""
parameter fitting and analysis module
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import os


@dataclass
class FitResult:
    """fitting result"""
    topology: str
    N: int
    infection_function: str
    beta: float
    delta: float
    mse: float
    s_observed: np.ndarray
    s_predicted: np.ndarray
    S_observed: np.ndarray
    S_predicted: np.ndarray
    s_std: np.ndarray = None
    S_std: np.ndarray = None
    n_runs: int = 0


def load_log_entries(log_path: str) -> List[dict]:
    entries = []    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def compute_s_i_t(
    entries: List[dict],
    topology: str,
    N: int,
    T: int,
    gamma: Optional[float] = None,
    return_std: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Compute s_i(t) = mean(X_i^(r)(t)) from log entries.
    """
    infection_by_run = defaultdict(lambda: defaultdict(dict))
    
    for e in entries:
        if e.get("topology") != topology or e.get("N") != N:
            continue
        if gamma is not None and e.get("gamma") != gamma:
            continue
        
        run_id = e.get("run_id", "")
        t = e.get("t", 0)
        agent_i = e.get("agent_i", 0)
        infected = e.get("infected", 0)
        infection_by_run[run_id][t][agent_i] = infected
    
    max_t_per_run = {rid: max(rd.keys()) for rid, rd in infection_by_run.items() if rd}
    n_runs = len(infection_by_run)
    
    s_mean = np.zeros((T + 1, N))
    s_std = np.zeros((T + 1, N)) if return_std else None
    
    for t in range(1, T + 1):
        for i in range(N):
            values = []
            for run_id, run_data in infection_by_run.items():
                if t in run_data and i in run_data[t]:
                    values.append(run_data[t][i])
                elif t > max_t_per_run.get(run_id, 0):
                    values.append(1)
            
            if values:
                s_mean[t, i] = np.mean(values)
                if return_std:
                    s_std[t, i] = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    s_mean[0] = s_mean[1]
    if return_std:
        s_std[0] = s_std[1]
    
    return s_mean, s_std, n_runs


def f_prod(
    s: np.ndarray,
    A: np.ndarray,
    beta: float,
    t: int,
    i: int
) -> float:
    """
    f_i^prod(t) = 1 - prod_{j in N(i)} (1 - beta * a_ji * s_j(t))
    """
    N = A.shape[0]
    product = 1.0
    
    for j in range(N):
        if A[j, i] > 0: 
            product *= (1.0 - beta * A[j, i] * s[t, j])
    
    return 1.0 - product


def f_pois(
    s: np.ndarray,
    A: np.ndarray,
    beta: float,
    t: int,
    i: int
) -> float:
    """
    f_i^pois(t) = 1 - exp(- beta * sum_{j in N(i)} a_ji * s_j(t))
    """
    N = A.shape[0]
    total_influence = 0.0
    
    for j in range(N):
        if A[j, i] > 0:
            total_influence += A[j, i] * s[t, j]
    
    return 1.0 - np.exp(-beta * total_influence)


def predict_dynamics(
    s0: np.ndarray,
    A: np.ndarray,
    beta: float,
    delta: float,
    T: int,
    infection_func: str = "prod"
) -> np.ndarray:
    """
    s_i(t+1) = (1-delta) * s_i(t) + (1-s_i(t)) * f_i(t)

    """
    N = A.shape[0]
    s = np.zeros((T + 1, N))
    s[0] = s0
    
    f_func = f_prod if infection_func == "prod" else f_pois
    
    for t in range(T):
        for i in range(N):
            f_i = f_func(s, A, beta, t, i)
            s[t + 1, i] = (1 - delta) * s[t, i] + (1 - s[t, i]) * f_i
            s[t + 1, i] = np.clip(s[t + 1, i], 0.0, 1.0)
    
    return s


def compute_mse(s_observed: np.ndarray, s_predicted: np.ndarray) -> float:
    return np.mean((s_observed - s_predicted) ** 2)


def grid_search_fit(
    s_observed: np.ndarray,
    A: np.ndarray,
    infection_func: str = "prod",
    beta_range: Tuple[float, float] = (0.0, 1.0),
    delta_range: Tuple[float, float] = (0.0, 1.0),
    n_coarse: int = 21,
    n_fine: int = 11
) -> Tuple[float, float, float]:
    """
    grid search fit (beta, delta)
    """
    T = s_observed.shape[0] - 1
    s0 = s_observed[0]
    
    betas_coarse = np.linspace(beta_range[0], beta_range[1], n_coarse)
    deltas_coarse = np.linspace(delta_range[0], delta_range[1], n_coarse)
    
    best_beta = 0.0
    best_delta = 0.0
    best_mse = float('inf')
    
    for beta in betas_coarse:
        for delta in deltas_coarse:
            s_pred = predict_dynamics(s0, A, beta, delta, T, infection_func)
            mse = compute_mse(s_observed, s_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_beta = beta
                best_delta = delta
    
    beta_step = (beta_range[1] - beta_range[0]) / (n_coarse - 1) if n_coarse > 1 else 0.1
    delta_step = (delta_range[1] - delta_range[0]) / (n_coarse - 1) if n_coarse > 1 else 0.1
    
    betas_fine = np.linspace(
        max(0.0, best_beta - beta_step),
        min(1.0, best_beta + beta_step),
        n_fine
    )
    deltas_fine = np.linspace(
        max(0.0, best_delta - delta_step),
        min(1.0, best_delta + delta_step),
        n_fine
    )
    
    for beta in betas_fine:
        for delta in deltas_fine:
            s_pred = predict_dynamics(s0, A, beta, delta, T, infection_func)
            mse = compute_mse(s_observed, s_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_beta = beta
                best_delta = delta
    
    return best_beta, best_delta, best_mse


def fit_from_log(
    log_path: str,
    topology: str,
    N: int,
    T: int,
    A: np.ndarray,
    gamma: Optional[float] = None
) -> Dict[str, FitResult]:
    """
    fit parameters from log file
    """
    entries = load_log_entries(log_path)
    
    s_observed, s_std, n_runs = compute_s_i_t(entries, topology, N, T, gamma, return_std=True)
    
    results = {}
    
    for func_name in ["prod", "pois"]:
        s_fit = s_observed[1:] if s_observed.shape[0] > 1 else s_observed

        beta, delta, mse = grid_search_fit(s_fit, A, func_name)
        
        T_fit = s_fit.shape[0] - 1
        s_pred_fit = predict_dynamics(s_fit[0], A, beta, delta, T_fit, func_name)
        
        if s_observed.shape[0] > 1:
            s_predicted = np.full((T + 1, N), np.nan, dtype=float)
            s_predicted[1:] = s_pred_fit
        else:
            s_predicted = s_pred_fit
        
        S_observed = s_observed.mean(axis=1)
        S_predicted = s_predicted.mean(axis=1)
        
        S_std = s_std.mean(axis=1) if s_std is not None else None
        
        if len(S_observed) > 0:
            S_observed = S_observed.copy()
            S_observed[0] = np.nan
        if S_std is not None and len(S_std) > 0:
            S_std = S_std.copy()
            S_std[0] = np.nan
        
        results[func_name] = FitResult(
            topology=topology,
            N=N,
            infection_function=func_name,
            beta=beta,
            delta=delta,
            mse=mse,
            s_observed=s_observed,
            s_predicted=s_predicted,
            S_observed=S_observed,
            S_predicted=S_predicted,
            s_std=s_std,
            S_std=S_std,
            n_runs=n_runs
        )
    
    return results


def print_fit_results(results: Dict[str, FitResult]):
    """print fit results"""
    print("\n" + "=" * 60)
    print("Fitting Results")
    print("=" * 60)
    
    for func_name, result in results.items():
        print(f"\n{func_name.upper()} Infection Function:")
        print(f"  Topology: {result.topology}, N={result.N}")
        print(f"  Fitted beta: {result.beta:.4f}")
        print(f"  Fitted delta: {result.delta:.4f}")
        print(f"  MSE: {result.mse:.6f}")
        print(f"  S(t) observed: {result.S_observed}")
        print(f"  S(t) predicted: {result.S_predicted}")


def compute_reinfection_probability(entries: List[dict], codeword_filter: Optional[str] = None) -> float:
    """
    compute reinfection probability P(reinfection | recovered)
    """
    agent_trajectories = defaultdict(list)
    
    for e in entries:
        key = (e.get("run_id"), e.get("agent_i"))
        t = e.get("t", 0)
        infected = e.get("infected", 0)
        agent_trajectories[key].append((t, infected))
    
    recovery_events = 0
    reinfection_events = 0
    
    for key, trajectory in agent_trajectories.items():
        trajectory.sort(key=lambda x: x[0])
        
        prev_infected = 0
        recovered = False
        
        for t, infected in trajectory:
            if prev_infected == 1 and infected == 0:
                recovered = True
                recovery_events += 1
            elif recovered and infected == 1:
                reinfection_events += 1
                recovered = False
            prev_infected = infected
    
    if recovery_events == 0:
        return 0.0
    
    return reinfection_events / recovery_events


def analyze_gamma_sweep(
    log_paths: List[str],
    gammas: List[float],
    topology: str,
    N: int,
    T: int,
    A: np.ndarray
) -> Dict[float, Dict[str, FitResult]]:
    """
    analyze gamma sweep results
    """
    all_results = {}
    
    for log_path, gamma in zip(log_paths, gammas):
        results = fit_from_log(log_path, topology, N, T, A, gamma)
        all_results[gamma] = results
    
    return all_results


def save_results_csv(results: List[FitResult], output_path: str):
    """save fit results to CSV"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("topology,N,infection_function,beta,delta,mse\n")
        for r in results:
            f.write(f"{r.topology},{r.N},{r.infection_function},{r.beta:.4f},{r.delta:.4f},{r.mse:.6f}\n")


# if __name__ == "__main__":
#     print("Testing fit module...")
    
#     N = 3 # number of nodes
#     T = 5 # number of rounds
    
#     A = np.ones((N, N)) - np.eye(N)
    
#     s0 = np.array([1.0, 0.0, 0.0])
#     true_beta = 0.3
#     true_delta = 0.1
    
#     s_true = predict_dynamics(s0, A, true_beta, true_delta, T, "prod")
    
#     np.random.seed(42)
#     s_observed = s_true + np.random.normal(0, 0.05, s_true.shape)
#     s_observed = np.clip(s_observed, 0, 1)
    
#     print(f"\nTrue parameters: beta={true_beta}, delta={true_delta}")
#     print(f"True S(t): {s_true.mean(axis=1)}")
    
#     for func_name in ["prod", "pois"]:
#         beta, delta, mse = grid_search_fit(s_observed, A, func_name)
#         print(f"\n{func_name.upper()} fit: beta={beta:.3f}, delta={delta:.3f}, MSE={mse:.6f}")
