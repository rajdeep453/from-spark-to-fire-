# Vulnerability

> for Vulnerability Validation in LLM Multi-Agent Systems

## Quick Start

### Configuration

Before running experiments, follow these steps:

1. **Install dependencies**:
   ```bash
   pip install -r threat/requirements.txt
   ```

2. **Set your OpenAI API key** in `threat/config.py`:
   ```python
   api_key: Optional[str] = "your_api_key"
   ```
   Or set environment variable `OPENAI_API_KEY`.

3. **Configure model** (optional): default is `"gpt-4o-mini"` in `threat/config.py`:
   ```python
   model: str = "gpt-4o-mini"
   ```

## Running Experiments

Run from repository root with `-m`.

### Experiment 1

```bash
python -m threat.run_exp1
```

### Experiment 2

```bash
python -m threat.run_exp2
```
### Experiment 3

```bash
python -m threat.exp_timing.run_timing
```