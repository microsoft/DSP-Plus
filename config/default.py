#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Record all your configurations here
"""

# Input data file path
data = "datasets/minif2f.jsonl"

# Data split names to process
split = ["test"]

# Directory to store outputs
target_dir = "result/dsp_minif2f"

# Number of concurrent processes allowed
concurrent_num = 64

# Attempts means the number of DSP workflow attempts for each problem
attempts = 32

# The maximum number of requests for each model server (will be counted and controlled locally)
draft_max_running_requests = 128

# The maximum number of requests for each model server (will be counted and controlled locally)
sketch_max_running_requests = 128

# Number of Lean servers used for sketch verification
sketch_leanserver_num = 8

# Number of Lean servers used for proof verification
prove_leanserver_num = 64

# Configuration for the draft model servers. Needs to be replaced before launching.
draft_model_config = [
    {
        "base_url": "http://localhost:20000/v1",
        "api_key": "EMPTY",
    },
    {
        "base_url": "http://localhost:20001/v1",
        "api_key": "EMPTY",
    },
    {
        "base_url": "http://localhost:20002/v1",
        "api_key": "EMPTY",
    },
]

# Sampling configuration for draft model
draft_sample_config = {
    "model": "Qwen/QwQ-32B",
    "temperature": 0.6,
    "top_p": 0.95,
    "timeout": 3600,
    "max_tokens": 32768,
}

# Configuration for the sketch model servers. Needs to be replaced before launching.
sketch_model_config = [
    {
        "base_url": "http://localhost:8081/v1",
        "api_key": "EMPTY",
    },
]

# Sampling configuration for the sketch model
sketch_sample_config = {
    "model": "deepseek-ai/DeepSeek-V3-0324",
    "temperature": 0.7,
    "top_p": 0.95,
    "timeout": 600,
    "max_tokens": 32768,
}

# Verify configuration used in sketch phase
sketch_verify_config = {
    "verify_timeout": 180,
}

# Configuration for the proving model servers. Needs to be replaced before launching.
prove_model_config = [
    {
        "base_url": "http://127.0.0.1:30001/v1",
        "api_key": "EMPTY",
    },
]

# Sampling configuration for the proving model
prove_sampling_config = {
    "name_lean_copilot": "BFS-Prover-API",
    "model": "bytedance-research/BFS-Prover",
    "temperature": 1.1,
    "top_p": 1,
    "timeout": 1800,
    "max_tokens": 64,
    "n": 8,
    "max_output": 4,
    "use_beam_search": False,
}

# Verify configuration used in proving phase
prove_verify_config = {
    "verify_timeout": 1200,
    "max_tree_size": 64,
    "search_attempts": 4,
    "port_lean_copilot": 23338,
}