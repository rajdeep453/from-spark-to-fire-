"""
Topology Generation Module
"""

import numpy as np
from typing import Tuple, List, Dict, Set


def build_topology(
    name: str, 
    N: int,
    edge_weight_mode: str = "uniform",
    edge_prob: float = 1.0,
    random_seed: int = 42,
    horizontal_prob: float = 0.3,
    skip_prob: float = 0.0,
    reciprocity: float = 0.5
) -> Tuple[np.ndarray, Dict[int, Set[int]], List[str]]:
    """
    Build adjacency matrix and neighbor sets for specified topology.
    """
    if name == "centralized":
        A, agents = _build_centralized(N)
    elif name == "layered":
        A, agents = _build_layered(N)
    elif name == "layered_horizontal":
        A, agents = _build_layered_horizontal(N, horizontal_prob, skip_prob, reciprocity, random_seed)
    elif name == "decentralized":
        A, agents = _build_decentralized(N, edge_prob, random_seed)
    else:
        raise ValueError(f"Unknown topology: {name}")
    
    if edge_weight_mode == "normalized":
        A = _normalize_by_in_degree(A)
    
    neighbors = _build_neighbors(A)
    
    return A, neighbors, agents


def _build_centralized(N: int) -> Tuple[np.ndarray, List[str]]:
    A = np.zeros((N, N), dtype=np.float64)
    center = 0
    
    for leaf in range(1, N):
        A[center, leaf] = 1.0
        A[leaf, center] = 1.0
    
    agents = ["Coordinator"] + [f"Expert_{i}" for i in range(1, N)]
    return A, agents


def _build_layered(N: int) -> Tuple[np.ndarray, List[str]]:
    A = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N - 1):
        A[i, i + 1] = 1.0
    
    if N == 3:
        agents = ["Analyst", "Solver", "Validator"]
    else:
        agents = [f"Layer_{i}" for i in range(N)]
    
    return A, agents


def _build_layered_horizontal(
    N: int,
    horizontal_prob: float = 0.3,
    skip_prob: float = 0.0,
    reciprocity: float = 0.5,
    random_seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(random_seed)
    A = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N - 1):
        A[i, i + 1] = 1.0
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if A[i, j] > 0:
                continue
            
            dist = abs(i - j)
            prob = horizontal_prob if dist == 1 else skip_prob if dist > 1 else 0
            
            if rng.random() < prob:
                A[i, j] = 1.0
                if rng.random() < reciprocity:
                    A[j, i] = 1.0
    
    agents = [f"Layer_{i}" for i in range(N)]
    return A, agents


def _build_decentralized(N: int, edge_prob: float = 1.0, random_seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(random_seed)
    
    if edge_prob >= 1.0:
        A = np.ones((N, N), dtype=np.float64) - np.eye(N)
    else:
        max_attempts = 100
        for attempt in range(max_attempts):
            A = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    if i != j and rng.random() < edge_prob:
                        A[j, i] = 1.0
            
            if _is_weakly_connected(A):
                break
            
            rng = np.random.default_rng(random_seed + attempt + 1)
        else:
            A = _ensure_connectivity(A, rng)
    
    agents = [f"Agent_{i+1}" for i in range(N)]
    return A, agents


def _is_weakly_connected(A: np.ndarray) -> bool:
    N = A.shape[0]
    if N <= 1:
        return True
    
    A_undirected = (A + A.T) > 0
    
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    
    while queue:
        node = queue.pop(0)
        for neighbor in range(N):
            if A_undirected[node, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1
    
    return count == N


def _ensure_connectivity(A: np.ndarray, rng) -> np.ndarray:
    """Ensure graph is connected by adding minimum edges."""
    N = A.shape[0]
    A_undirected = (A + A.T) > 0
    
    visited = [False] * N
    components = []
    
    for start in range(N):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in range(N):
                if A_undirected[node, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    
    for i in range(len(components) - 1):
        u = rng.choice(components[i])
        v = rng.choice(components[i + 1])
        A[u, v] = 1.0
        A[v, u] = 1.0
    
    return A


def _normalize_by_in_degree(A: np.ndarray) -> np.ndarray:
    """Normalize adjacency matrix by in-degree."""
    A_norm = A.copy()
    in_degrees = A.sum(axis=0)
    
    for i in range(A.shape[1]):
        if in_degrees[i] > 0:
            A_norm[:, i] /= in_degrees[i]
    
    return A_norm


def _build_neighbors(A: np.ndarray) -> Dict[int, Set[int]]:
    """Build neighbor sets from adjacency matrix."""
    N = A.shape[0]
    return {i: set(j for j in range(N) if A[j, i] > 0) for i in range(N)}


def get_visible_senders(neighbors: Dict[int, Set[int]], node_i: int) -> List[int]:
    """Get sorted list of visible senders for node i."""
    return sorted(list(neighbors.get(node_i, set())))


def describe_topology(A: np.ndarray, agents: List[str]) -> str:
    """Generate text description of topology."""
    N = A.shape[0]
    lines = [f"Topology with {N} nodes:"]
    
    for i in range(N):
        incoming = [agents[j] for j in range(N) if A[j, i] > 0]
        if incoming:
            lines.append(f"  {agents[i]} receives from: {', '.join(incoming)}")
        else:
            lines.append(f"  {agents[i]} receives from: (none)")
    
    return "\n".join(lines)
