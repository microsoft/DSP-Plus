#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
python quick_start.py > output.txt
"""

import uvicorn
import argparse
import multiprocessing as mp

from dsp import Draft, Sketch, Prove
from dsp.proving import create_app
from dsp.utils import load_config


if __name__ == "__main__":
    # parse cmd parameters
    parser = argparse.ArgumentParser(description="DSP+ Solver")
    parser.add_argument('--config', type=str, default='config/default.py')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # load your data
    data = {"name": "imo_1959_p1", "formal_statement": "theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by\n", "goal": "n : ℕ\nh₀ : 0 < n\n⊢ (21 * n + 4).gcd (14 * n + 3) = 1", "header": "import Mathlib\nopen BigOperators Real Nat Topology Rat\n"}

    # launch proving model server
    u_server = mp.Process(target=lambda: uvicorn.run(create_app(cfg), access_log=False, port=cfg.prove_verify_config["port_lean_copilot"]))
    u_server.start()

    # load dsp workflow config
    Draft.launch_llmserver(client_config = cfg.draft_model_config, max_llm_requests = cfg.draft_max_running_requests, **cfg.draft_sample_config)
    Sketch.launch_llmserver(client_config = cfg.sketch_model_config, max_llm_requests = cfg.sketch_max_running_requests, **cfg.sketch_sample_config)
    Sketch.launch_lean4server(max_lean4_requests = cfg.sketch_leanserver_num, **cfg.sketch_verify_config)
    Prove.launch_lean4server(max_lean4_requests = cfg.prove_leanserver_num, **cfg.prove_verify_config)

    # Run Draft, Sketch and Prove
    for i in range(cfg.attempts):
        try:
            print(f"Attempt {i+1}/{cfg.attempts} for {data['name']}")
            
            draft = Draft().run_direct(data)
            print("-"*10 + "Draft" + "-"*10 + "\n" + draft + "\n" + "-"*20 + "\n")

            sketch = Sketch().run_direct(data, draft)
            print("-"*10 + "Sketch" + "-"*10 + "\n" + sketch + "\n" + "-"*20+ "\n")

            prove, repl_output = Prove().run_direct(data, sketch)
            print("-"*10 + "Prove" + "-"*10 + "\n" + prove + "\n" + "-"*20+ "\n")
            print(f"Prove Complete : {repl_output['complete']}")
            if repl_output['complete']:
                break
        except Exception as e:
            print(f"Error processing {data['name']} - {i} : {e}")
            continue

    # terminate the process
    u_server.terminate()
    u_server.join()
    Sketch.close_lean4server()
    Prove.close_lean4server()
    Draft.close_llmserver()
    Sketch.close_llmserver()
    
    
# Example Output:
'''
Complete launching Draft LLMServerProcesses
Complete launching Sketch LLMServerProcesses
Complete launching 8 sketch_verifier LeanServerProcesses
Complete launching 64 prove_verifier LeanServerProcesses
Attempt 1/32 for imo_1959_p1
   imo_1959_p1 drafting...
Done Draft imo_1959_p1
----------draft----------
### Step 1:  
\[ \text{GCD}(21n + 4, 14n + 3) \]  

### Step 2:  
\[ 21n + 4 = 1 \cdot (14n + 3) + (7n + 1) \]  

### Step 3:  
\[ \text{GCD}(14n + 3, 7n + 1) \]  

### Step 4:  
\[ 14n + 3 = 2 \cdot (7n + 1) + 1 \]  

### Step 5:  
\[ \text{GCD}(7n + 1, 1) \]  

### Step 6:  
\[ \text{GCD}(7n + 1, 1) = 1 \]
--------------------

   imo_1959_p1 sketching...
Done Sketch imo_1959_p1
----------sketch----------
open BigOperators Real Nat Topology Rat

theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  -- Step 1: Start with GCD(21n + 4, 14n + 3)
  have h1 : Nat.gcd (21 * n + 4) (14 * n + 3) = Nat.gcd (14 * n + 3) (7 * n + 1) := by
    -- Step 2: Apply Euclidean algorithm step 21n + 4 = 1*(14n + 3) + (7n + 1)
    have h2 : 21 * n + 4 = 1 * (14 * n + 3) + (7 * n + 1) := by
      prove_with[]
    prove_with[h2, Nat.gcd_add_mul_right_left]
  
  -- Step 3: Now consider GCD(14n + 3, 7n + 1)
  have h3 : Nat.gcd (14 * n + 3) (7 * n + 1) = Nat.gcd (7 * n + 1) 1 := by
    -- Step 4: Apply Euclidean algorithm step 14n + 3 = 2*(7n + 1) + 1
    have h4 : 14 * n + 3 = 2 * (7 * n + 1) + 1 := by
      prove_with[]
    prove_with[h4, Nat.gcd_add_mul_right_left]
  
  -- Step 5: Final step GCD(7n + 1, 1)
  have h5 : Nat.gcd (7 * n + 1) 1 = 1 := by
    prove_with[Nat.gcd_one_right]
  
  -- Step 6: Combine all steps
  prove_with[h1, h3, h5]
--------------------

   imo_1959_p1 proving...
   TaskQueue-Draft:  1 requests popped with avg batch_size 1.0 in last period  0 waiting in queue
   TaskQueue-Sketch:  1 requests popped with avg batch_size 1.0 in last period  0 waiting in queue
   TaskQueue-sketch_verifier:  1 requests popped with avg batch_size 1.0 in last period  0 waiting in queue
   TaskQueue-prove_verifier:  2 requests popped with avg batch_size 1.0 in last period  0 waiting in queue
Done Prove imo_1959_p1
----------prove----------
open BigOperators Real Nat Topology Rat

theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  -- Step 1: Start with GCD(21n + 4, 14n + 3)
  have h1 : Nat.gcd (21 * n + 4) (14 * n + 3) = Nat.gcd (14 * n + 3) (7 * n + 1) := by
    -- Step 2: Apply Euclidean algorithm step 21n + 4 = 1*(14n + 3) + (7n + 1)
    have h2 : 21 * n + 4 = 1 * (14 * n + 3) + (7 * n + 1) := by
      clear * -
      (linarith)
    rw [h2]
    rw [add_comm]
    rw [gcd_comm]
    simp [add_comm]
  
  -- Step 3: Now consider GCD(14n + 3, 7n + 1)
  have h3 : Nat.gcd (14 * n + 3) (7 * n + 1) = Nat.gcd (7 * n + 1) 1 := by
    -- Step 4: Apply Euclidean algorithm step 14n + 3 = 2*(7n + 1) + 1
    have h4 : 14 * n + 3 = 2 * (7 * n + 1) + 1 := by
      clear * -
      (linarith)
    (field_simp [*] at *)
  
  -- Step 5: Final step GCD(7n + 1, 1)
  have h5 : Nat.gcd (7 * n + 1) 1 = 1 := by
    (simp)
  
  -- Step 6: Combine all steps
  clear * - h1 h3 h5
  (linarith)
--------------------

Prove Complete : True
All 8 sketch_verifier LeanServerProcesses stopped
All 64 prove_verifier LeanServerProcesses stopped
Draft LLMServerProcesses stopped
Sketch LLMServerProcesses stopped
'''