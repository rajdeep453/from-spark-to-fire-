"""
V1 Experiment: Topology comparison
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prop_dynamics.config import ExperimentConfig
from prop_dynamics.topology import build_topology, describe_topology
from prop_dynamics.runner import run_experiment, load_sample_questions
from prop_dynamics.fit import fit_from_log, print_fit_results, save_results_csv


def run_v1_experiments(
    questions: list = None,
    topologies: list = None,
    N: int = 5,
    T_rounds: int = 5,
    R_repeats: int = 30,
    seed: int = 42,
    output_dir: str = "outputs/v1_topology",
    num_seeds: int = 1,
    edge_prob: float = 0.3,
    horizontal_prob: float = 0.3,
    skip_prob: float = 0.0,
    reciprocity: float = 0.5
):
    """Run V1 topology comparison experiment."""
    if questions is None:
        questions = load_sample_questions(5)
    if topologies is None:
        topologies = ["centralized", "layered", "layered_horizontal", "decentralized"]
    
    all_results = []
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "N": N, "T_rounds": T_rounds, "R_repeats": R_repeats,
            "seed": seed, "num_seeds": num_seeds, "edge_prob": edge_prob,
            "horizontal_prob": horizontal_prob, "skip_prob": skip_prob, "reciprocity": reciprocity,
            "n_questions": len(questions),
            "total_trials": len(questions) * R_repeats
        }, f, indent=2)
    
    total_trials = len(questions) * R_repeats
    print("=" * 60)
    print("V1: Topology Comparison Experiment")
    print(f"N={N}, T={T_rounds}, R={R_repeats}, Questions={len(questions)}")
    print(f"Total trials per topology: {total_trials}")
    print("=" * 60)

    for topo in tqdm(topologies, desc="Topologies"):
        print(f"\n--- Running: {topo} ---")
        
        config = ExperimentConfig(
            topology=topo,
            N=N,
            T_rounds=T_rounds,
            R_repeats=R_repeats,
            seed=seed,
            num_seeds=num_seeds,
            edge_prob=edge_prob,
            horizontal_prob=horizontal_prob,
            skip_prob=skip_prob,
            reciprocity=reciprocity,
            output_dir=os.path.join(output_dir, topo),
            log_file=f"{topo}_log.jsonl"
        )
        
        A, _, _ = build_topology(
            topo,
            N,
            edge_prob=config.edge_prob,
            random_seed=seed,
            horizontal_prob=horizontal_prob,
            skip_prob=skip_prob,
            reciprocity=reciprocity
        )
        agents = [f"Agent_{i+1}" for i in range(N)]
        print(describe_topology(A, agents))
        
        log_path = run_experiment(config, questions)
        
        print(f"Fitting {topo}...")
        results = fit_from_log(log_path, topo, N, T_rounds, A)
        print_fit_results(results)
        
        for r in results.values():
            all_results.append(r)
    
    summary_path = os.path.join(output_dir, "v1_summary.csv")
    save_results_csv(all_results, summary_path)
    print(f"\nSummary: {summary_path}")
    
    _print_comparison(all_results)
    return all_results


def _print_comparison(results):
    """Print comparison table with extended statistics"""
    by_topo = {}
    for r in results:
        by_topo.setdefault(r.topology, {})[r.infection_function] = r
    
    print("\n" + "=" * 80)
    print("Fitting Summary")
    print("=" * 80)
    print(f"| {'Topology':16} | Func | Beta  | Delta | MSE    | S(T)  | Std(T) | n_runs |")
    print("|" + "-" * 18 + "|------|-------|-------|--------|-------|--------|--------|")
    
    topos = ["centralized", "layered", "layered_horizontal", "decentralized"]
    for r in results:
        if r.topology not in topos:
            topos.append(r.topology)
    
    for topo in topos:
        if topo not in by_topo:
            continue
        for func in ["prod", "pois"]:
            if func in by_topo[topo]:
                r = by_topo[topo][func]
                S_final = r.S_observed[-1] if len(r.S_observed) > 0 else 0
                S_std_final = r.S_std[-1] if r.S_std is not None and len(r.S_std) > 0 else 0
                n_runs = r.n_runs if r.n_runs else "?"
                # Handle NaN for display
                S_final_str = f"{S_final:.3f}" if not np.isnan(S_final) else "N/A"
                S_std_str = f"{S_std_final:.3f}" if not np.isnan(S_std_final) else "N/A"
                print(f"| {topo:16} | {func:4} | {r.beta:.3f} | {r.delta:.3f} | {r.mse:.4f} | {S_final_str:5} | {S_std_str:6} | {n_runs:6} |")
    
    print("=" * 80)


def run_v1_layered_variants(
    questions: list = None,
    horizontal_probs: list = None,
    N: int = 10,
    T_rounds: int = 10,
    R_repeats: int = 30,
    seed: int = 42,
    skip_prob: float = 0.0,
    reciprocity: float = 0.5,
    output_dir: str = "outputs/v1_layered_variants"
):

    
    if questions is None:
        questions = load_sample_questions(5)
    if horizontal_probs is None:
        horizontal_probs = [0.1, 0.3, 0.5, 0.7]
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_trials = len(questions) * R_repeats
    print("=" * 60)
    print("V1: Layered Horizontal Variants Experiment")
    print(f"N={N}, T={T_rounds}, R={R_repeats}, Questions={len(questions)}")
    print(f"Total trials per variant: {total_trials}")
    print(f"Horizontal probs: {horizontal_probs}")
    print("=" * 60)
    
    # not relevant to this experiment
    config_path = os.path.join(output_dir, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "N": N, "T_rounds": T_rounds, "R_repeats": R_repeats,
            "seed": seed, "skip_prob": skip_prob, "reciprocity": reciprocity,
            "horizontal_probs": horizontal_probs,
            "n_questions": len(questions),
            "total_trials": total_trials
        }, f, indent=2)
    
    all_results = []
    
    for hp in tqdm(horizontal_probs, desc="Variants"):
        variant_name = f"lh_{hp:.1f}"
        variant_dir = os.path.join(output_dir, variant_name)
        
        config = ExperimentConfig(
            topology="layered_horizontal",
            N=N,
            T_rounds=T_rounds,
            R_repeats=R_repeats,
            seed=seed,
            horizontal_prob=hp,
            skip_prob=skip_prob,
            reciprocity=reciprocity,
            output_dir=variant_dir,
            log_file=f"{variant_name}_log.jsonl"
        )
        
        A, _, _ = build_topology(
            "layered_horizontal", N,
            horizontal_prob=hp,
            skip_prob=skip_prob,
            reciprocity=reciprocity,
            random_seed=seed
        )
        
        print(f"\n{variant_name}: {int(A.sum())} edges")
        log_path = run_experiment(config, questions)
        
        results = fit_from_log(log_path, variant_name, N, T_rounds, A)
        for r in results.values():
            all_results.append(r)
    
    summary_path = os.path.join(output_dir, "v1_layered_summary.csv")
    save_results_csv(all_results, summary_path)
    print(f"\nSummary: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_v1_experiments(
        questions=load_sample_questions(3),
        N=2, T_rounds=1, R_repeats=1, # number of nodes, number of rounds, number of repeats
        edge_prob=0.4, # edge probability
        horizontal_prob=0.3, # horizontal probability
        topologies=["layered"], # ["centralized", "layered", "layered_horizontal", "decentralized"]
        output_dir="outputs/v1_topology" # output directory
    )
    
    # results = run_v1_layered_variants(
    #     questions=load_sample_questions(3),
    #     horizontal_probs=[0.1, 0.3, 0.5, 0.7],
    #     N=10, T_rounds=10, R_repeats=20,
    #     output_dir="outputs/v1_layered_variants"
    # )
    _print_comparison(results)

# python ./prop_dynamics/run_v1_exp.py