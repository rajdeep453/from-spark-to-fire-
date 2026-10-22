# Propagation Dynamics

> for Model Utility

## Quick Start

### Configuration

Before running experiments, follow these steps:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key** in `config.py`:
   ```python
   api_key: Optional[str] = 'your_api_key'
   ```
   Or set environment variable `OPENAI_API_KEY`.

3. **Configure model** (optional): Default is `"gpt-4o-mini"`. Change in `config.py` if needed:
   ```python
   model: str = "gpt-4o-mini"
   ```

### Running V1 Experiment

Run the topology comparison experiment:

```bash
python prop_dynamics/run_v1_exp.py
```

The experiment will:
- Compare different network topologies
- Run multiple trials with different questions (as noise)
- Fit Model parameters from observed data
- Save results to `outputs/v1_topology/`


### Module Dependencies

```
run_v1_exp.py
├── config.py              # ExperimentConfig
├── topology.py            # build_topology
├── runner.py              # run_experiment
│   ├── personas.py        # get_persona
│   ├── payload.py         # make_payload
│   └── topology.py        # build_topology
└── fit.py                 # fit_from_log
```
