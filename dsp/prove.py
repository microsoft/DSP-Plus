#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import re
import json
import warnings
from typing import Optional, List, Tuple

from .worker import Lean4ServerScheduler


class Prove:
    lean4_scheduler = None
    
    @classmethod
    def close_lean4server(cls):
        if cls.lean4_scheduler is not None:
            cls.lean4_scheduler.close()
    
    @classmethod
    def launch_lean4server(cls, max_lean4_requests: int=1, verify_timeout: int=1200, max_tree_size: int=64, search_attempts: int=4, port_lean_copilot: int=23337, name_lean_copilot: str="BFS-Prover-API", cwd: str='./mathlib4', name: str='prove_verifier'):
        command = f'''import Mathlib
import LeanCopilot
import Aesop
import RulesetInit
set_option maxHeartbeats 0
set_option maxRecDepth 1024
@[aesop 100% (rule_sets := [bfs])] def tacGen := LeanCopilot.tacGen
add_aesop_rules unsafe 90% [(by rfl), (by linarith), (by nlinarith), (by ring), (by positivity), (by omega), (by ring_nf), (by ring_nf at *), (by simp), (by simp_all), (by field_simp), (by field_simp [*] at *), (by norm_num), (by norm_num [*] at *), (by norm_cast), (by norm_cast at *)]
open Lean Meta LeanCopilot
def BFS : ExternalGenerator := {{
  name := "{name_lean_copilot}"
  host := "localhost"
  port := {port_lean_copilot}
}}
#eval registerGenerator "BFS-Prover" (.external BFS)
set_option LeanCopilot.suggest_tactics.model "BFS-Prover"
macro "bfsaesop" : tactic =>
  `(tactic| aesop? (config := {{ enableSimp := false, enableUnfold := false, maxGoals := {max_tree_size}, bfsScore := true, terminal := true }}) (rule_sets := [bfs, -builtin, -default]))
syntax "prove_with" ("[" term,* "]")? : tactic
macro_rules
| `(tactic| prove_with [$args,*]) => `(tactic| sorry)
| `(tactic| prove_with)           => `(tactic| sorry)
def bfsaesopLoop : Lean.Elab.Tactic.TacticM Unit := do
  try
    Lean.Elab.Tactic.evalTactic (← `(tactic| aesop? (config := {{ enableSimp := false, enableUnfold := false, maxGoals := {max_tree_size}, terminal := true }}) (rule_sets := [-builtin])))
    return
  catch _ =>
    pure ()
  try
    Lean.Elab.Tactic.evalTactic (← `(tactic| aesop? (config := {{ enableSimp := false, enableUnfold := false, maxGoals := {max_tree_size}, terminal := true }})))
    return
  catch _ =>
    pure ()
  for _ in [0:{search_attempts}] do
    try
      Lean.Elab.Tactic.evalTactic (← `(tactic| bfsaesop))
      return
    catch _ =>
      pure ()
  Lean.throwError "bfsaesopLoop failed"
elab "bfsaesopLoop" : tactic =>
  bfsaesopLoop
'''
        cls.lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests = max_lean4_requests, 
                                                   timeout = verify_timeout, 
                                                   memory_limit = 10, 
                                                   name = name, 
                                                   cwd = cwd, 
                                                   command = json.dumps({"cmd": command}), 
                                                   share_header = 'import Mathlib')

    def modify_content(self, ori_list: List[str], list: List[str], bfs_list: List[int], index: int, content: Optional[str], remove: bool=False) -> None:
        """Modify a specific line in the list. Can also remove the line.
        
        Args:
            ori_list (List[str]): The original list of code blocks.
            list (List[str]): The modifiable list of code blocks.
            bfs_list (List[int]): The list of rows using BFS.
            index (int): The global line index to modify.
            content (str): The new content for the line. If None, the line is removed.
            remove (bool): Whether to remove the index from bfs_list.
        """
        count = 0
        for i, item in enumerate(ori_list):
            line_list = item.split('\n')
            # Check if the current block contains the target index
            if count + len(line_list) > index:
                if content is not None:
                    # Replace the specific line with new content
                    line_list[index - count] = content
                else:
                    # Remove the specific line
                    line_list.pop(index - count)
                # Update the block with modified lines
                list[i] = '\n'.join(line_list)
                if remove:
                    try:
                        # Remove index from bfs_list if needed
                        bfs_list.remove(i)
                    except Exception as e:
                        print(f"Error key: {i}", flush=True)
                break
            # Track total line count across blocks
            count += len(line_list)
            
    def get_content(self, list: List[str], index: int) -> str:
        """Get a specific line of content from a code block list by global index"""
        count = 0
        for item in list:
            line_list = item.split('\n')
            if count + len(line_list) > index:
                # Return the line at the global index
                return line_list[index - count]
            count += len(line_list)
        raise IndexError("Index out of range")
            
    def remove_error_line(self, code: List[str], bfs_list: List[int]) -> List[str]:
        """Remove lines that cause errors from Lean4 output"""
        request_id = self.lean4_scheduler.submit_request(dict(code='\n'.join(code), allTactics=False))
        repl_output = self.lean4_scheduler.get_request_outputs(request_id)
        error_line = []
        
        # If system-level errors exist, raise an exception
        if repl_output['system_messages']:
            raise ValueError(repl_output['system_messages'])

        # Collect error line numbers
        for error in repl_output.get('errors', []):
            if error['data'].startswith('unsolved goals'):
                pass
            else:
                error_line.append(error['pos']['line']-1)

        # Remove all error lines
        code_copy = code.copy()
        for line in error_line:
            self.modify_content(code_copy, code, bfs_list, line, None)
        return code

    def send_request(self, formal_proof_bfs: List[str], bfs_list: List[int]) -> Tuple[dict, List[str]]:
        """Send code to Lean4 and integrate suggested tactics if any"""
        request_id = self.lean4_scheduler.submit_request(dict(code='\n'.join(formal_proof_bfs), allTactics=False))
        repl_output = self.lean4_scheduler.get_request_outputs(request_id)
        if repl_output['system_messages']:
            raise ValueError(repl_output['system_messages'])
        
        formal_proof_aesop = formal_proof_bfs.copy()
        for info in repl_output.get('infos', []):
            line = info['pos']['line'] - 1
            # Calculate indentation of the target line
            indent = len(self.get_content(formal_proof_bfs, line)) - len(self.get_content(formal_proof_bfs, line).lstrip())
            
            if info["data"].startswith("Try this:\n") and "sorry" not in info["data"]:
                # Extract suggested tactic lines
                code_list = info["data"].split("Try this:\n")[1].split("\n")
                
                # Apply indentation to suggested code
                if code_list[0].startswith(' '):
                    code_list = [' ' * (indent-2) + code for code in code_list]
                else:
                    code_list[0] = ' ' * indent + code_list[0]
                    code_list = [code_list[0]] + [' ' * (indent-2) + code for code in code_list[1:]]
                    
                # Join lines and update original proof
                code = '\n'.join(code_list)
                self.modify_content(formal_proof_bfs, formal_proof_aesop, bfs_list, line, code, remove=True)
                
        return repl_output, formal_proof_aesop
    
    def replace_prove_with(self, code: str, target: str) -> Tuple[List[int], List[str]]:
        """Replace 'prove_with' and 'sorry' with the given tactic. Collect BFS points"""
        bfs_list = []
        codes = code.split('\n')
        
        for i, line in enumerate(codes):
            if line.strip().startswith('prove_with'):
                # Handle multiline prove_with [..., ...]
                while '[' in line and ']' not in line:
                    codes[i] = codes[i].rstrip() + ' ' + codes[i+1].lstrip()
                    codes.pop(i+1)
                    line = codes[i]
                    
                match = re.search(r'( +)prove_with\s*(\[(.*?)\])?', line)
                blank = match.group(1)
                content = match.group(3)
                if content:
                    args = ' '.join(content.split(', '))
                    codes[i] = f'{blank}clear * - {args}\n{blank}{target}'
                else:
                    codes[i] = f'{blank}clear * -\n{blank}{target}'
                bfs_list.append(i)
                
            elif line.lstrip().startswith('sorry'):
                # Replace 'sorry' with the target tactic
                codes[i] = ' ' * (len(line) - len(line.lstrip())) + target
                bfs_list.append(i)
                
            elif 'prove_with' in line and not line.strip().startswith('--'):
                # Handle inline prove_with within a line
                match = re.search(r'prove_with\s*(\[(.*?)\])?', line)
                blank = ' ' * (len(line) - len(line.lstrip()) + 2)
                content = match.group(2)
                new_line = line.rsplit('prove_with', 1)[0].rstrip()
                if content:
                    args = ' '.join(content.split(', '))
                    codes.insert(i+1, f'{blank}clear * - {args}\n{blank}{target}')
                else:
                    codes.insert(i+1, f'{blank}clear * -\n{blank}{target}')
                codes[i] = new_line
                bfs_list.append(i+1)
                
        return bfs_list, codes

    def auto_theorem_prove(self, formal_proof_sorry: str, strict: bool=False, aesop: str='bfsaesopLoop') -> Tuple[str, dict]:
        """Automatically try to complete theorems using Lean4 tactic suggestions.

        Args:
            formal_proof_sorry (str): The proof script with 'sorry' placeholders.
            strict (bool): If True, retry proof without 'clear' even if errors remain.
            aesop (str): The tactic to replace 'sorry' with.

        Returns:
            Tuple: (proof script, Lean4 response)
        """
        # Replace all 'sorry' and 'prove_with' with the target tactic
        bfs_list, formal_proof_bfs = self.replace_prove_with(formal_proof_sorry, 'sorry')

        # Remove lines that cause parsing or type errors
        formal_proof_bfs = self.remove_error_line(formal_proof_bfs, bfs_list)
        
        # Replace all 'sorry' placeholders with the actual tactic
        for item in bfs_list:
            formal_proof_bfs[item] = re.sub(r'\bsorry\b', aesop, formal_proof_bfs[item])

        # Send request to Lean4 and receive suggestions
        repl_output, formal_proof_1 = self.send_request(formal_proof_bfs, bfs_list)

        # Revert 'aesop' back to 'sorry' for readability or further retry
        proof = re.sub(re.escape(aesop), 'sorry', '\n'.join(formal_proof_1))

        # If strict mode is enabled and proof is incomplete, retry without clear statements
        if strict and not repl_output['complete']:
            formal_proof_1_copy = formal_proof_1.copy()
            for k in bfs_list:
                indent = len(formal_proof_1[k]) - len(formal_proof_1[k].lstrip())
                formal_proof_1_copy[k] = ' ' * indent + aesop

            repl_output, formal_proof_2 = self.send_request(formal_proof_1_copy)
            proof = re.sub(re.escape(aesop), 'sorry', '\n'.join(formal_proof_2))

        # We no longer add extra header files for proof
        return proof, repl_output
    
    def check_proof(self, data: dict, proof: str, idx: int) -> None:
        """Check whether the proof problem has been tampered with to prevent LLM deception"""
        clean_formal_statement = re.sub(r'\s+', '', data['formal_statement'])
        clean_proof = re.sub(r'\s+', '', proof)
        if (clean_formal_statement not in clean_proof) or ("--" + clean_formal_statement in clean_proof):
            message = "{} - {} The proof may have been tampered with or annotated, please check.".format(data['name'], idx)
            warnings.warn(message)
    
    
    def judge_if_exist(self, res_dir: str, idx: int) -> bool:
        """judge whether the historical file exists"""
        return os.path.exists(os.path.join(res_dir, f"{idx}_prove.json"))
                
    def read_sketch(self, res_dir: str, idx: int) -> Optional[str]:
        """got sketch from file"""
        if os.path.exists(os.path.join(res_dir, f"{idx}_sketch.json")):
            with open(os.path.join(res_dir, f"{idx}_sketch.json"), 'r', encoding='utf-8') as fp:
                output = json.load(fp)
            return output['sketch']
        else:
            return None
                
    def save_prove(self, res_dir: str, idx: int, proof: str, repl_output: dict) -> None:
        """save prove to file"""
        with open(f"{res_dir}/{idx}_sketch.json", 'r', encoding='utf-8') as fps, open(f"{res_dir}/{idx}_prove.json", 'w', encoding='utf-8') as fpp:
            prev_result = json.load(fps)
            prev_result.update({
                "result": proof,
                "repl_output": repl_output,
            })
            json.dump(prev_result, fpp, indent=2, ensure_ascii=False)
        os.remove(f"{res_dir}/{idx}_sketch.json")
    
    def run(self, data: dict, res_dir: str, idx: int) -> None:
        """main interface"""
        if not self.judge_if_exist(res_dir, idx):
            sketch = self.read_sketch(res_dir, idx)
            if sketch is not None:
                print(f"   {data['name']} - {idx} proving...", flush=True)
                proof, repl_output = self.auto_theorem_prove(sketch)
                self.save_prove(res_dir, idx, proof, repl_output)
                print(f"Done Prove {data['name']} - {idx}", flush=True)
                
                if repl_output['complete']:
                    self.check_proof(data, proof, idx)
                    with open(f"{res_dir}/finished.txt", 'a+', encoding='utf-8') as fp:
                        fp.write(f"{idx}_prove.json\n")
                    print(f"Success DSP+ {data['name']} - {idx}", flush=True)
                    
    def run_direct(self, data: dict, sketch: str) -> Tuple[str, dict]:
        """Interface that directly returns results"""
        print(f"   {data['name']} proving...", flush=True)
        proof, repl_output = self.auto_theorem_prove(sketch)
        print(f"Done Prove {data['name']}", flush=True)
        return proof, repl_output
