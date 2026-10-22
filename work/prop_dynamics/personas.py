"""
Persona Module
"""

import random

def get_uniform_persona(agent_index: int) -> str:
    return f"""You are Agent {agent_index + 1} in a collaborative problem-solving discussion.
Your role is to analyze the problem, consider information from other agents, and provide a well-reasoned response.
Be thorough in your analysis and open to different perspectives."""

# not relateivew to this experiment
def get_coordinator_prompt() -> str:
    return """You are a Coordinator. Your role is to analyze the problem, synthesize insights from others, and make informed decisions."""

# not relateivew to this experiment
def get_expert_prompt(expert_id: int = 1) -> str:
    return f"""You are Expert {expert_id}. Provide specialized analysis and professional perspective on the problem."""

# not relateivew to this experiment
def get_analyst_prompt() -> str:
    return """You are an Analyst. Analyze the problem and identify key concepts and approaches. Do not provide the final answer."""

# not relateivew to this experiment
def get_solver_prompt() -> str:
    return """You are a Solver. Build upon the analysis and provide a well-reasoned answer."""

# not relateivew to this experiment
def get_validator_prompt() -> str:
    return """You are a Validator. Verify the answer for correctness and provide corrections if needed."""


def get_agent_prompt(agent_id: int) -> str:
    return f"""You are Agent {agent_id} in a collaborative discussion. Analyze and respond to others' viewpoints briefly."""

# not relateivew to this experiment
def get_checker_prompt(mode: str = "helpful") -> str:
    if mode == "critical":
        return """You are a Checker in CRITICAL mode. Rigorously scrutinize all claims.
Question assumptions, challenge conclusions, and point out flaws.
Do not accept answers without solid justification."""
    return """You are a Checker in HELPFUL mode. Review constructively and offer gentle corrections if needed."""


def sample_checker_mode(gamma: float, seed: int, round_t: int) -> str:
    rng = random.Random(seed + round_t * 1000)
    return "critical" if rng.random() < gamma else "helpful"


def get_persona(
    topology: str,
    agent_index: int,
    agents: list,
    is_checker: bool = False,
    checker_mode: str = "helpful",
    use_uniform: bool = False
) -> str:
    if is_checker:
        return get_checker_prompt(checker_mode)
    
    if use_uniform:
        return get_uniform_persona(agent_index)
    
    agent_name = agents[agent_index] if agent_index < len(agents) else f"Agent_{agent_index}"
    
    if topology == "centralized":
        return get_coordinator_prompt() if agent_index == 0 else get_expert_prompt(agent_index)
    elif topology == "layered":
        if "Analyst" in agent_name:
            return get_analyst_prompt()
        elif "Solver" in agent_name:
            return get_solver_prompt()
        elif "Validator" in agent_name:
            return get_validator_prompt()
        return f"You are {agent_name}. Analyze the input and provide your output."
    else:
        return get_agent_prompt(agent_index + 1)
