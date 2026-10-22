"""
V1 Plot: Generate plots from existing experiment results.

"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prop_dynamics.topology import build_topology
from prop_dynamics.fit import fit_from_log, FitResult


def parse_log_params(log_path: str) -> dict:
    topo, N, T = None, None, 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if topo is None:
                topo = e.get("topology")
                N = e.get("N")
            T = max(T, e.get("t", 0))
    return {"topology": topo, "N": N, "T": T}


def find_log_files(path: str) -> list:
    if os.path.isfile(path) and path.endswith(".jsonl"):
        return [path]
    if os.path.isdir(path):
        logs = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith("_log.jsonl"):
                    logs.append(os.path.join(root, f))
        return sorted(logs)
    return []


def load_exp_config(path: str) -> dict:
    search_dir = path if os.path.isdir(path) else os.path.dirname(path)
    for _ in range(4):
        cfg_path = os.path.join(search_dir, "exp_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                print(f"Loaded config: {cfg_path}")
                return json.load(f)
        search_dir = os.path.dirname(search_dir)
    return {}


def fit_logs(log_paths: list, seed: int = 42, edge_prob: float = 0.3, T_rounds: int = None) -> list:
    results = []
    for log_path in log_paths:
        params = parse_log_params(log_path)
        topo, N = params["topology"], params["N"]
        T = T_rounds if T_rounds else params["T"]
        if not all([topo, N, T]):
            print(f"Skip {log_path}: missing params")
            continue
        
        A, _, _ = build_topology(topo, N, edge_prob=edge_prob, random_seed=seed)
        
        print(f"Fitting: {os.path.basename(log_path)} (topo={topo}, N={N}, T={T})")
        fit_results = fit_from_log(log_path, topo, N, T, A)
        results.extend(fit_results.values())
    return results


def plot_results(
    results: list,
    output_dir: str,
    show_ci: bool = True,
    ci_alpha: float = 0.2,
    band_mode: str = "std",
    filename_suffix: str = ""
):
    """
    Generate S(t) curve plot with optional confidence interval shading.
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Times New Roman'
    
    os.makedirs(output_dir, exist_ok=True)
    
    by_topo = {}
    for r in results:
        by_topo.setdefault(r.topology, {})[r.infection_function] = r
    
    topos = [t for t in ["centralized", "layered", "layered_horizontal", "decentralized"] if t in by_topo]
    if not topos:
        topos = sorted(by_topo.keys())
    if not topos:
        print("No results to plot")
        return
    
    n = len(topos)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    
    for idx, topo in enumerate(topos):
        ax = axes[idx]
        
        for func in ["prod", "pois"]:
            if func not in by_topo[topo]:
                continue
            r = by_topo[topo][func]
            times = np.arange(len(r.S_observed))
            
            if func == "prod":
                ax.plot(times, r.S_observed, 'ko-', label='Observed', markersize=6, linewidth=1.5)
                
                if show_ci and r.S_std is not None:
                    n_runs = r.n_runs if r.n_runs else 0
                    if band_mode == "se" and n_runs > 0:
                        band = r.S_std / np.sqrt(n_runs)
                    else:
                        band = r.S_std
                    S_lower = np.clip(r.S_observed - band, 0, 1)
                    S_upper = np.clip(r.S_observed + band, 0, 1)
                    ax.fill_between(times, S_lower, S_upper, alpha=ci_alpha, color='gray', 
                                    label=None)
            
            color = 'tab:blue' if func == 'prod' else 'tab:orange'
            ax.plot(times, r.S_predicted, '--', color=color, 
                    label=f'{func}', linewidth=2)
        
        ax.set_xlabel('Round t', fontsize=18)
        ax.set_ylabel('S(t)', fontsize=18)
        
        title_str = topo.replace('_', ' ').capitalize()
        ax.set_title(title_str, fontsize=20)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=14, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-0.2, len(times) - 0.5)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'v1_St_curves{filename_suffix}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_fit_quality(results: list, output_dir: str):
    """
    Generate a summary plot showing MSE comparison across topologies.
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    by_topo = {}
    for r in results:
        by_topo.setdefault(r.topology, {})[r.infection_function] = r
    
    topos = sorted(by_topo.keys())
    if not topos:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    x_pos = np.arange(len(topos))
    width = 0.35
    
    mse_prod = [by_topo[t].get("prod", None) for t in topos]
    mse_pois = [by_topo[t].get("pois", None) for t in topos]
    
    mse_prod_vals = [r.mse if r else 0 for r in mse_prod]
    mse_pois_vals = [r.mse if r else 0 for r in mse_pois]
    
    bars1 = ax1.bar(x_pos - width/2, mse_prod_vals, width, label='prod', color='tab:blue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, mse_pois_vals, width, label='pois', color='tab:orange', alpha=0.8)
    
    ax1.set_xlabel('Topology')
    ax1.set_ylabel('MSE')
    ax1.set_title('Fitting Quality: MSE by Topology')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([t[:12] for t in topos], rotation=30, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    for func, marker, color in [("prod", "o", "tab:blue"), ("pois", "^", "tab:orange")]:
        betas = [by_topo[t][func].beta for t in topos if func in by_topo[t]]
        deltas = [by_topo[t][func].delta for t in topos if func in by_topo[t]]
        labels = [t for t in topos if func in by_topo[t]]
        
        ax2.scatter(betas, deltas, marker=marker, s=100, label=func, color=color, alpha=0.8)
        for b, d, lbl in zip(betas, deltas, labels):
            ax2.annotate(lbl[:8], (b, d), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax2.set_xlabel('β (infection rate)')
    ax2.set_ylabel('δ (recovery rate)')
    ax2.set_title('Fitted Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'v1_fit_quality.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_summary(results: list):
    """Print summary table"""
    by_topo = {}
    for r in results:
        by_topo.setdefault(r.topology, {})[r.infection_function] = r
    
    print("\n| Topology | Func | Beta | Delta | MSE | S(T) |")
    print("|----------|------|------|-------|-----|------|")
    for topo in ["centralized", "layered", "layered_horizontal", "decentralized"]:
        if topo not in by_topo:
            continue
        for func in ["prod", "pois"]:
            if func in by_topo[topo]:
                r = by_topo[topo][func]
                S_final = r.S_observed[-1] if len(r.S_observed) > 0 else 0
                print(f"| {topo:10} | {func:4} | {r.beta:.3f} | {r.delta:.3f} | {r.mse:.4f} | {S_final:.3f} |")


def main():
    parser = argparse.ArgumentParser(description="Generate V1 plots from log files")
    parser.add_argument("path", help="Log file (.jsonl) or directory")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (auto from config)")
    parser.add_argument("--edge-prob", type=float, default=None, help="Edge probability (auto from config)")
    args = parser.parse_args()
    
    log_files = find_log_files(args.path)
    if not log_files:
        print(f"No log files found in: {args.path}")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    cfg = load_exp_config(args.path)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    edge_prob = args.edge_prob if args.edge_prob is not None else cfg.get("edge_prob", 0.3)
    T_rounds = cfg.get("T_rounds")  # Use config T to handle early stop
    print(f"Using: seed={seed}, edge_prob={edge_prob}, T_rounds={T_rounds}")
    
    results = fit_logs(log_files, seed=seed, edge_prob=edge_prob, T_rounds=T_rounds)
    if not results:
        print("No results to plot")
        return
    
    output_dir = args.output
    if output_dir is None:
        output_dir = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    
    print_summary(results)
    plot_results(results, output_dir, band_mode="std", filename_suffix="")
    plot_results(results, output_dir, band_mode="se", filename_suffix="_se")
    plot_fit_quality(results, output_dir)


if __name__ == "__main__":
    main()

# python ./prop_dynamics/run_v1_plot.py outputs/v1_topology