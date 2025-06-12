#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
DSP+ Workflow

Run Command:
python dsp_workflow.py --config config/default.py

Count of Successfully Solved Problems:
find /path/to/your/target_dir -type f -name "finished.txt" | wc -l
"""


import os
import time
import uvicorn
import argparse
import multiprocessing as mp
from loguru import logger
from typing import Tuple

from dsp import Draft, Sketch, Prove
from dsp.proving import create_app
from dsp.utils import load_dataset, load_config


def process_single_task(tasks: Tuple[dict, str, int]) -> None:
    """
    1) For single task, do draft, sketch and prove.  
    2) Every intermediate file will be saved, so you can continue from the last progress after an interruption.  
    3) You can also comment out any of the steps to implement different steps on different machines.  
    4) You can check the 'finished.txt' in the subdirectory to count the number finished.  
    """
    try: 
        data, target_dir, idx = tasks
        
        name = data['name']
        res_dir = os.path.join(target_dir, name)
        os.makedirs(res_dir, exist_ok=True)
        
        if os.path.exists(f"{res_dir}/finished.txt"):
            # print(f"Already finished {name}, skipping.", flush=True)
            return

        Draft().run(data, res_dir, idx)
        Sketch().run(data, res_dir, idx)
        Prove().run(data, res_dir, idx)
        
    except Exception as e:
        logger.error(f"Error processing {data['name']} - {idx} : {e}")


if __name__ == "__main__":
    # parse cmd parameters
    parser = argparse.ArgumentParser(description="DSP+ Solver")
    parser.add_argument('--config', type=str)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # load dataset
    datasets = load_dataset(cfg.data, cfg.split, args.node_rank, args.world_size)
    
    # launch proving model server
    u_server = mp.Process(target=lambda: uvicorn.run(create_app(cfg), access_log=False, port=cfg.prove_verify_config["port_lean_copilot"]))
    u_server.start()
    
    # load dsp workflow config
    Draft.launch_llmserver(client_config = cfg.draft_model_config, max_llm_requests = cfg.draft_max_running_requests, **cfg.draft_sample_config)
    Sketch.launch_llmserver(client_config = cfg.sketch_model_config, max_llm_requests = cfg.sketch_max_running_requests, **cfg.sketch_sample_config)
    Sketch.launch_lean4server(max_lean4_requests = cfg.sketch_leanserver_num, **cfg.sketch_verify_config)
    Prove.launch_lean4server(max_lean4_requests = cfg.prove_leanserver_num, **cfg.prove_verify_config)
    
    # Multi-process submit in order
    all_tasks = [(data, cfg.target_dir, idx) for idx in range(cfg.attempts) for data in datasets]
    processes = []
    try:
        for task in all_tasks:
            while len(processes) >= cfg.concurrent_num:
                processes = [p for p in processes if p.is_alive()]
                time.sleep(0.1)
            p = mp.Process(target=process_single_task, args=(task,))
            p.start()
            processes.append(p)
    except KeyboardInterrupt:
        print("🛑 KeyboardInterrupt received! Terminating all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
    finally:
        for p in processes:
            p.join()
        print("✅ All processes joined.")

    # close workers
    u_server.terminate()
    u_server.join()
    Sketch.close_lean4server()
    Prove.close_lean4server()
    Draft.close_llmserver()
    Sketch.close_llmserver()