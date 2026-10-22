# Defense

## Quick Start

### Configuration

Before running experiments, please follow these steps:

1. **Install dependencies**:
   ```bash
   pip install -r prop_dynamics/defense/requirements.txt
   ```

2. **Set your OpenAI API key** in `prop_dynamics/defense/config.py`:
   ```python
   api_key: Optional[str] = "your_api_key"
   ```
   Or set environment variable `OPENAI_API_KEY`.

3. **Configure model** (optional): default is `"gpt-4o-mini"` in `prop_dynamics/defense/config.py`:
   ```python
   model: str = "gpt-4o-mini"
   ```

## Running Experiments

Run from repository root with `-m`.

### Build Local KB (Optional)

```bash
python -m prop_dynamics.defense.experiments.build_local_kb
```

### Defense Main Experiment

```bash
python -m prop_dynamics.defense.experiments.run_defense_exp
```

### Defense Ablation

```bash
python -m prop_dynamics.defense.experiments.run_ablation_exp
```