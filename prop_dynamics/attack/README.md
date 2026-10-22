# Attack

## Quick Start

### Configuration

Before running experiments, follow these steps:

1. **Install dependencies**:
   ```bash
   pip install -r prop_dynamics/attack/requirements.txt
   ```

2. **Set your OpenAI API key** in `prop_dynamics/attack/config.py`:
   ```python
   api_key: Optional[str] = "your_api_key"
   ```
   Or set environment variable `OPENAI_API_KEY`.

3. **Configure model** (optional): default is `"gpt-4o-mini"` in `prop_dynamics/attack/config.py`:
   ```python
   model: str = "gpt-4o-mini"
   ```

## Running Experiments

Run from repository root with `-m`.

### Mega Experiment

```bash
python -m prop_dynamics.attack.experiments.run_mega_exp
```

### Analyze Mega Logs

```bash
python -m prop_dynamics.attack.experiments.mega_exp_analysis
```