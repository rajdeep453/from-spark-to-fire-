"""Classifier Comparison."""

import os
import sys
import time
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prop_dynamics.defense.config import DefenseConfig, setup_openai

try:
    from sentence_transformers import CrossEncoder
    HAS_NLI = True
except ImportError:
    HAS_NLI = False


def load_real_samples(n_samples: int = None, dataset_path: str = None) -> List[Dict]:
    if dataset_path is None:
        v3_path = "outputs/defense/nli_real/real_samples_v3_balanced.jsonl"
        v2_path = "outputs/defense/nli_real/real_samples_v2.jsonl"
        v1_path = "outputs/defense/nli_real/real_samples_v1.jsonl"
        if os.path.exists(v3_path):
            dataset_path = v3_path
        elif os.path.exists(v2_path):
            dataset_path = v2_path
        else:
            dataset_path = v1_path
    
    reviewed_path = dataset_path.replace('.jsonl', '_reviewed.jsonl')
    if os.path.exists(reviewed_path):
        print(f"Using reviewed dataset: {reviewed_path}")
        dataset_path = reviewed_path
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset unavailable.")
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                samples.append({
                    'premise': item['premise'],
                    'hypothesis': item['hypothesis'],
                    'label': item['label'],
                    'task_id': item.get('task_id', ''),
                    'scenario': item.get('scenario', ''),
                    'framework': item.get('framework', ''),
                    'attack_strategy': item.get('attack_strategy', ''),
                    'label_source': item.get('label_source', ''),
                })
    
    if n_samples and len(samples) > n_samples:
        import random
        by_label = defaultdict(list)
        for s in samples:
            by_label[s['label']].append(s)
        
        samples = []
        for label, items in by_label.items():
            ratio = len(items) / sum(len(v) for v in by_label.values())
            n = max(1, int(n_samples * ratio))
            samples.extend(random.sample(items, min(n, len(items))))
        
        random.shuffle(samples)
        samples = samples[:n_samples]
    
    return samples


def load_snli_samples(n_samples: int = 100) -> List[Dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("snli", split="test")
        samples = []
        for item in ds:
            if item['label'] in [0, 1, 2] and len(samples) < n_samples:
                label = ['entailment', 'neutral', 'contradiction'][item['label']]
                samples.append({'premise': item['premise'], 'hypothesis': item['hypothesis'], 'label': label})
        return samples
    except Exception:
        fallback = get_fallback_samples()
        print("Using fallback samples.")
        return fallback


def get_fallback_samples() -> List[Dict]:
    return [
        {"premise": "A man is eating pizza.", "hypothesis": "A man eats something.", "label": "entailment"},
        {"premise": "A man is eating pizza.", "hypothesis": "A man is sleeping.", "label": "contradiction"},
        {"premise": "A man is eating pizza.", "hypothesis": "A man is at a restaurant.", "label": "neutral"},
        {"premise": "The dog is running in the park.", "hypothesis": "An animal is moving.", "label": "entailment"},
        {"premise": "The dog is running in the park.", "hypothesis": "The cat is sleeping.", "label": "contradiction"},
        {"premise": "A woman is playing guitar.", "hypothesis": "A person is making music.", "label": "entailment"},
        {"premise": "A woman is playing guitar.", "hypothesis": "A woman is cooking.", "label": "contradiction"},
        {"premise": "Children are playing soccer.", "hypothesis": "Kids are outside.", "label": "neutral"},
        {"premise": "The API endpoint is https://api.mathjs.org/v4/", "hypothesis": "Use https://api.mathjs.org/v4/", "label": "entailment"},
        {"premise": "The trusted URL is https://api.safe.com", "hypothesis": "Use https://malware.io/api instead.", "label": "contradiction"},
    ] * 10


def classify_llm(premise: str, hypothesis: str, openai_module) -> Tuple[str, int, str]:
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nOutput only: ENTAILMENT, CONTRADICTION, or NEUTRAL"
    resp = openai_module.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=20
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    text = raw.upper()
    tokens = resp.get("usage", {}).get("total_tokens", 0)
    if "ENTAILMENT" in text: return "entailment", tokens, raw
    if "CONTRADICTION" in text: return "contradiction", tokens, raw
    return "neutral", tokens, raw


def _llm_task(args: Tuple[int, Dict, object]) -> Tuple[int, str, int, float, str]:
    """(index, sample, openai_module) -> (index, pred, tokens, elapsed, raw_response). Per-request timing inside."""
    idx, s, openai_module = args
    t0 = time.time()
    pred, tokens, raw = classify_llm(s["premise"], s["hypothesis"], openai_module)
    elapsed = time.time() - t0
    return (idx, pred, tokens, elapsed, raw)


def classify_nli(premise: str, hypothesis: str, model):
    scores = model.predict([(premise, hypothesis)])[0]
    return ['contradiction', 'entailment', 'neutral'][scores.argmax()]


def classify_similarity(premise: str, hypothesis: str) -> str:
    wp, wh = set(premise.lower().split()), set(hypothesis.lower().split())
    overlap = len(wp & wh) / max(len(wh), 1)
    if overlap > 0.5: return "entailment"
    if overlap < 0.15: return "contradiction"
    return "neutral"


def compute_detailed_metrics(preds: List[str], golds: List[str], labels: List[str] = None) -> Dict:
    """Compute detailed metrics: accuracy, per-class P/R/F1, confusion matrix."""
    if labels is None:
        labels = ['entailment', 'neutral', 'contradiction']
    
    # Confusion matrix
    confusion = {l: {ll: 0 for ll in labels} for l in labels}
    for pred, gold in zip(preds, golds):
        if gold in confusion and pred in labels:
            confusion[gold][pred] += 1
    
    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(confusion[label].values())
        }
    
    # Macro-average F1
    macro_f1 = np.mean([per_class[l]['f1'] for l in labels])
    
    # Accuracy
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    accuracy = correct / len(preds) if preds else 0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'confusion_matrix': confusion,
    }


def run_experiment(samples: List[Dict], openai_module, skip_llm: bool = False) -> Tuple[Dict, List[Dict]]:
    """Returns (results, llm_logs). llm_logs is list of {premise, hypothesis, prompt, response, label} for LLM I/O."""
    results = {}
    llm_logs: List[Dict] = []
    nli_models = {}
    all_preds = {}  # Store predictions for detailed metrics
    
    if HAS_NLI:
        print("Loading NLI models (first run may download)...", flush=True)
        nli_models['deberta-small'] = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        nli_models['deberta-large'] = CrossEncoder("cross-encoder/nli-deberta-v3-large")
    
    methods = ['deberta-small', 'deberta-large', 'similarity']
    if not skip_llm:
        methods = ['gpt4o-mini'] + methods
    
    LLM_MAX_WORKERS = 10
    for method in methods:
        if method.startswith('deberta') and method not in nli_models:
            continue
        
        print(f"  Running {method}..." + (" (parallel, {} workers)".format(LLM_MAX_WORKERS) if method == 'gpt4o-mini' else ""), flush=True)
        preds, times, total_tokens = [], [], 0

        if method == 'gpt4o-mini':
            task_args = [(i, s, openai_module) for i, s in enumerate(samples)]
            with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
                out = list(executor.map(_llm_task, task_args))
            out.sort(key=lambda x: x[0])
            preds = [x[1] for x in out]
            total_tokens = sum(x[2] for x in out)
            times = [x[3] for x in out]
            for x in out:
                idx, pred, _, _, raw = x
                s = samples[idx]
                gt = s["label"]
                prompt = f"Premise: {s['premise']}\nHypothesis: {s['hypothesis']}\nOutput only: ENTAILMENT, CONTRADICTION, or NEUTRAL"
                llm_logs.append({
                    "premise": s["premise"],
                    "hypothesis": s["hypothesis"],
                    "prompt": prompt,
                    "response": raw,
                    "label": pred,
                    "ground_truth": gt,
                    "correct": pred == gt,
                })
            print(f"    {method}: {len(samples)}/{len(samples)}", flush=True)
        else:
            for i, s in enumerate(samples):
                t0 = time.time()
                if method in nli_models:
                    pred = classify_nli(s['premise'], s['hypothesis'], nli_models[method])
                else:
                    pred = classify_similarity(s['premise'], s['hypothesis'])
                times.append(time.time() - t0)
                preds.append(pred)
                if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                    print(f"    {method}: {i + 1}/{len(samples)}", flush=True)

        # Store predictions for detailed metrics
        all_preds[method] = preds
        golds = [s['label'] for s in samples]
        
        # Compute detailed metrics
        detailed = compute_detailed_metrics(preds, golds)
        
        results[method] = {
            'accuracy': detailed['accuracy'],
            'macro_f1': detailed['macro_f1'],
            'per_class': detailed['per_class'],
            'confusion_matrix': detailed['confusion_matrix'],
            'avg_time_ms': np.mean(times) * 1000 if times else 0,
            'total_tokens': total_tokens,
            'n_samples': len(samples)
        }
    
    return results, llm_logs


def print_results(results: Dict, dataset_name: str, n_samples: int):
    """Print formatted results table."""
    print(f"\n{'='*60}")
    print(f"NLI Comparison Results ({dataset_name.upper()}, n={n_samples})")
    print(f"{'='*60}\n")
    
    # Main metrics table
    print(f"{'Method':<16} {'Accuracy':>10} {'Macro-F1':>10} {'Avg Time':>12} {'Tokens':>10}")
    print("-" * 60)
    for m in ['gpt4o-mini', 'deberta-small', 'deberta-large', 'similarity']:
        if m in results:
            d = results[m]
            print(f"{m:<16} {d['accuracy']:>10.1%} {d['macro_f1']:>10.3f} {d['avg_time_ms']:>10.1f}ms {d['total_tokens']:>10}")
    
    # Per-class metrics
    print(f"\n{'='*60}")
    print("Per-Class Metrics (Precision / Recall / F1)")
    print(f"{'='*60}\n")
    
    labels = ['entailment', 'neutral', 'contradiction']
    for m in ['gpt4o-mini', 'deberta-small', 'deberta-large', 'similarity']:
        if m not in results:
            continue
        print(f"\n{m}:")
        per_class = results[m].get('per_class', {})
        print(f"  {'Label':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*47}")
        for label in labels:
            if label in per_class:
                pc = per_class[label]
                print(f"  {label:<15} {pc['precision']:>8.3f} {pc['recall']:>8.3f} {pc['f1']:>8.3f} {pc['support']:>8}")


def main(n_samples: int = 100, dataset: str = 'snli', skip_llm: bool = False):
    config = DefenseConfig()
    openai_module = setup_openai(config)
    
    # Load dataset
    if dataset == 'real':
        print(f"Loading real experiment dataset...")
        samples = load_real_samples(n_samples)
        out_prefix = 'real'
    else:
        print(f"Loading SNLI dataset...")
        samples = load_snli_samples(n_samples)
        out_prefix = 'snli'
    
    print(f"Loaded {len(samples)} samples")
    
    # Show label distribution
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s['label']] += 1
    print(f"Label distribution: {dict(label_counts)}")
    
    # Run experiment
    results, llm_logs = run_experiment(samples, openai_module, skip_llm=skip_llm)
    
    # Save results
    out_dir = "outputs/defense/nli_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    if llm_logs:
        logs_path = os.path.join(out_dir, f"{out_prefix}_logs.json")
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(llm_logs, f, ensure_ascii=False, indent=2)
        print(f"LLM I/O logs saved to {logs_path}")
    
    results_path = os.path.join(out_dir, f"{out_prefix}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}")
    
    # Print results
    print_results(results, dataset, len(samples))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NLI vs LLM Classifier Comparison')
    parser.add_argument("--n", type=int, default=100, help="Number of samples to use")
    parser.add_argument("--dataset", type=str, default='snli', choices=['snli', 'real'],
                       help="Dataset to use: 'snli' (Stanford NLI) or 'real' (experiment logs)")
    parser.add_argument("--skip-llm", action='store_true', help="Skip LLM evaluation (faster, no API cost)")
    args = parser.parse_args()
    main(args.n, args.dataset, args.skip_llm)
