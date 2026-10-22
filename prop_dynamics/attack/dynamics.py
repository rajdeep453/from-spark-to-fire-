"""Dynamics Analysis."""

import numpy as np
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prop_dynamics.fit import grid_search_fit, predict_dynamics
from prop_dynamics.theory import compute_spectral_radius


def extract_s_i_t(results: List[Dict]) -> np.ndarray:
    if not results or "s_i_t" not in results[0]:
        return np.array([])
    
    s_i_t_list = [r["s_i_t"] for r in results if "s_i_t" in r and r["s_i_t"]]
    if not s_i_t_list:
        return np.array([])
    
    max_rounds = max(len(s) for s in s_i_t_list)
    n_agents = len(s_i_t_list[0][0]) if s_i_t_list[0] else 0
    
    s_mean = np.zeros((max_rounds, n_agents))
    
    for t in range(max_rounds):
        for i in range(n_agents):
            values = []
            for s_i_t in s_i_t_list:
                if t < len(s_i_t) and i < len(s_i_t[t]):
                    values.append(s_i_t[t][i])
            if values:
                s_mean[t, i] = np.mean(values)
    
    return s_mean


def fit_attack_dynamics(
    results: List[Dict],
    A: np.ndarray,
    infection_func: str = "prod"
) -> Dict:
    s_i_t = extract_s_i_t(results)
    
    if s_i_t.size == 0:
        return {
            "beta": 0.0,
            "delta": 0.0,
            "mse": float('inf'),
            "rho_A": 0.0,
            "R0": 0.0,
            "s_observed": s_i_t,
            "s_predicted": np.array([])
        }
    
    beta, delta, mse = grid_search_fit(s_i_t, A, infection_func)
    
    s_pred = predict_dynamics(s_i_t[0], A, beta, delta, len(s_i_t) - 1, infection_func)
    
    rho = compute_spectral_radius(A)
    R0 = (beta * rho) / delta if delta > 1e-6 else float('inf')
    
    return {
        "beta": beta,
        "delta": delta,
        "mse": mse,
        "rho_A": rho,
        "R0": R0,
        "threshold_exceeded": beta * rho > delta,
        "s_observed": s_i_t,
        "s_predicted": s_pred
    }


def build_framework_adjacency(framework: str, N: int = None) -> np.ndarray:
    from prop_dynamics.attack.frameworks import get_topology
    from prop_dynamics.topology import build_topology
    
    topology_map = {
        "chain": "layered",
        "star": "centralized",
        "mesh": "decentralized"
    }
    
    topo_type = get_topology(framework)
    topo_name = topology_map.get(topo_type, "centralized")
    
    if N is None:
        N = 4 if topo_type == "chain" else 4
    
    A, _, _ = build_topology(topo_name, N, edge_prob=1.0, random_seed=42)
    return A
