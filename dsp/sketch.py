#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import re
import json
from typing import List, Tuple, Optional

from .worker import Lean4ServerScheduler, LLMServerScheduler


class Sketch:
    llm_scheduler = None
    lean4_scheduler = None

    @classmethod
    def launch_llmserver(cls, client_config: List[dict], max_llm_requests: int=128, name: str='Sketch', **kwargs):
        cls.llm_scheduler = LLMServerScheduler(
            client_config = client_config,
            max_running_requests_per_client = max_llm_requests,
            name = name,
            **kwargs,
        )
        
    @classmethod
    def close_llmserver(cls):
        if cls.llm_scheduler is not None:
            cls.llm_scheduler.close()
    
    @classmethod
    def launch_lean4server(cls, max_lean4_requests: int=1, verify_timeout: int=180, cwd: str='./mathlib4', name: str='sketch_verifier'):
        command = '''import Mathlib
import Aesop
set_option maxHeartbeats 0
syntax "prove_with" ("[" term,* "]")? : tactic
macro_rules
| `(tactic| prove_with [$args,*]) => `(tactic| sorry)
| `(tactic| prove_with)           => `(tactic| sorry)
'''
        cls.lean4_scheduler = Lean4ServerScheduler(
            max_concurrent_requests = max_lean4_requests, 
            timeout = verify_timeout, 
            memory_limit = 10,
            name = name,
            cwd = cwd, 
            command = json.dumps({"cmd": command}), 
            share_header = 'import Mathlib',
        )
        
    @classmethod
    def close_lean4server(cls):
        if cls.lean4_scheduler is not None:
            cls.lean4_scheduler.close()
        
    def error_masking(self, sketch: str) -> str:
        """Remove the erroneous lines until there are no more errors"""
        
        def get_error_line(code: str) -> Tuple[List[int], dict]:
            """Verify and return all erroneous lines."""
            # code = re.sub('import .*\n', '', code)
            request_id = self.lean4_scheduler.submit_request(dict(code=code, allTactics=False))
            repl_output = self.lean4_scheduler.get_request_outputs(request_id)
            error_line = []
            if repl_output['system_messages']:
                raise ValueError(repl_output['system_messages'])
            for error in repl_output.get('errors', []):
                if error['data'].startswith('unsolved goals'):
                    pass
                else:
                    error_line.append(error['pos']['line'])
            return error_line, repl_output
        
        def parse_lean_code(lines: List[str]) -> List[dict]:
            """Parse Lean code into a tree structure based on indentation"""
            nonlocal error_line
            stack = []
            tree = []

            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue
                else:
                    indent_level = len(line) - len(line.lstrip())
                node = {"content": clean_line, "children": [], "error": i+1 in error_line}

                if not stack:
                    tree.append(node)
                    stack.append((indent_level, node))
                else:
                    while stack and indent_level <= stack[-1][0]:
                        stack.pop()

                    if stack:
                        stack[-1][1]["children"].append(node)
                    else:
                        tree.append(node)
                    stack.append((indent_level, node))

            return tree
        
        def print_tree(tree: List[dict], level: int=0, comment: bool=False, if_sorry: bool=False, strict: bool=False) -> None:
            """
            Recursively prints a structured proof tree, replacing content with 'sorry' 
            for error nodes or subtrees based on configuration.

            Parameters:
                tree (List[dict]): The current list of nodes at this level of the proof tree.
                                Each node is a dict with keys: 'content' (str), 
                                'error' (bool), and 'children' (List[dict]).
                level (int): The current indentation level (used for visual formatting).
                comment (bool): Whether to comment out this and all child levels 
                                (typically set if the parent node had an error).
                if_sorry (bool): Indicates whether this subtree has already been replaced with 'sorry'.
                                If True, suppress further output within this branch.
                strict (bool): If True, any error in a child node will cause its parent 
                            to be replaced with 'sorry', even if the parent has no error.

            External (nonlocal) variables:
                ori_line_count (int): Counts total number of printed lines.
                error_line (int): Tracks the line where an error occurred (if used).
                formal_proof_sorry (List[str]): Stores the final output lines (indented strings).
                sorries (List[int]): Tracks line indices where 'sorry' was inserted.

            Behavior:
                - For deep levels (level > 1), directly replace error nodes or their parents with 'sorry'.
                - For top levels (level <= 1), optionally comment out or replace based on error presence.
                - Recursive processing continues unless the current branch has been 'sorry'-blocked.
            """
            nonlocal ori_line_count, error_line, formal_proof_sorry, sorries
            for node in tree:
                if level > 1:
                    # Deeper level: directly suppress error nodes with 'sorry'
                    if node["error"]:
                        if not if_sorry:
                            sorries.append(len(formal_proof_sorry))
                            formal_proof_sorry.append("  " * level + '-- ' * comment + "sorry")
                            if_sorry = True
                    else:
                        if not if_sorry:
                            # Check if any child has error to decide if we still print content
                            child_error = any(child["error"] for child in node["children"])
                            if strict and child_error:
                                formal_proof_sorry.append("  " * level + '-- ' * comment + "sorry")
                                if_sorry = True
                            else:
                                formal_proof_sorry.append("  " * level + '-- ' * comment + node["content"])
                    ori_line_count += 1
                    if node["children"]:
                        print_tree(node["children"], level + 1, comment, if_sorry, strict)
                        
                else:
                    # Top-level: decide whether to comment or replace with 'sorry'
                    if_comment = comment or node["error"]
                    if not if_sorry:
                        child_error = any(child["error"] for child in node["children"])
                        if strict and child_error:
                            formal_proof_sorry.append("  " * level + '-- ' * if_comment + "sorry")
                            if_sorry = True
                        else:
                            formal_proof_sorry.append("  " * level + '-- ' * if_comment + node["content"])
                    ori_line_count += 1
                    if node["children"]:
                        print_tree(node["children"], level + 1, if_comment, if_sorry, strict)
        
        # Split the initial sketch into lines
        formal_proof_sorry = sketch.split('\n')
        last_error_line = []
        count = 0
        
        while True:
            # Run REPL to get error line and output
            error_line, repl_output = get_error_line('\n'.join(formal_proof_sorry))
            tree = parse_lean_code(formal_proof_sorry)

            if not error_line:
                break
            
            # Reset state for this iteration
            ori_line_count = 1
            sorries = []
            formal_proof_sorry = []
            
            # Use strict mode if same error repeats
            if error_line == last_error_line and count > 1:
                print_tree(tree, strict=True)
            else:
                print_tree(tree)
                
            last_error_line = error_line
            count += 1
            
        # # Append 'sorry' if unsolved goals remain
        if any(error['data'].startswith('unsolved goals') for error in repl_output.get('errors', [])):
            formal_proof_sorry.append('  sorry')
            
        return '\n'.join(formal_proof_sorry)
        
    def llm_generator(self, data: dict, draft: str) -> str:
        """get raw output from llm"""
        # get what you need
        formal_statement = data['formal_statement']
        header = data['header']
        
        # get llm prompt
        prompt = f'''informal_proof:
{draft}

Prove the theorem in Lean 4 code. You should translate steps in the informal proof in a series of 'have'/'let'/'induction'/'match'/'suffices' statements, but you do not need to prove them. You only need to use placeholder `by{{new_line}}prove_with[h1, step5, ...{{hypothesises used here which are proposed ahead}}]`. We want to have as many lemmas as possible, and every lemma must be easy to proof.

When using a / b, you must specify **a's or b's type**, because (1:ℝ) / 2 is 0.5, but (1:ℤ) / 2 is 0.
When using a - b, you must specify **a's or b's type**, because (1:ℤ) - 2 is -1, but (1:ℕ) - 2 is 0.
n! is incorrect, you should use (n)!.

Here is an example:
```lean4
import Mathlib

example (x y : ℝ) (h1 : x ≤ 1 / 2) (h2 : x > 0) (t: y < Real.sin (x)): y < 1 / 2 := by
  -- Step 1
  have h3 : y < (1:ℝ) / 2 := by
    -- Step 2
    have h4 : Real.sin x ≤ x := by
      prove_with[h2]
    -- Step 3
    have h5 : y < x := by
      prove_with[h4, t]
    prove_with[h1, h5]
  exact h3
```

formal_statement:
```lean4
{header}
{formal_statement}
'''
    
        # get llm output
        request_id = self.llm_scheduler.submit_request(prompt)
        output = self.llm_scheduler.get_request_outputs(request_id)
        if not output:
            raise RuntimeError("Sketch LLM request failed")
        else:
            return output
        
    def extract_sketch(self, raw_sketch: str) -> str:
        """extract sketch from llm output"""
        sketch = raw_sketch.split('</think>')[-1].strip()
        sketch = re.search(r'```lean4?\n(.+?)\n```', sketch, re.DOTALL).group(1)
        sketch = re.sub(r'import .+?\n', '', sketch)
        sketch = sketch.strip()
        return sketch
    
    
    def judge_if_exist(self, res_dir: str, idx: int) -> bool:
        """judge whether the historical file exists"""
        return os.path.exists(os.path.join(res_dir, f"{idx}_sketch.json")) \
                or os.path.exists(os.path.join(res_dir, f"{idx}_prove.json"))
                
    def read_draft(self, res_dir: str, idx: int) -> Optional[str]:
        """got draft from file"""
        if os.path.exists(os.path.join(res_dir, f"{idx}_draft.json")):
            with open(os.path.join(res_dir, f"{idx}_draft.json"), 'r', encoding='utf-8') as fp:
                output = json.load(fp)
            return output['draft']
        else:
            return None
                
    def save_sketch(self, res_dir: str, idx: int, raw_sketch: str, sketch: str) -> None:
        """save sketch to file"""
        with open(f"{res_dir}/{idx}_draft.json", 'r', encoding='utf-8') as fpd, open(f"{res_dir}/{idx}_sketch.json", 'w', encoding='utf-8') as fps:
            prev_result = json.load(fpd)
            prev_result.update({
                "raw_sketch": raw_sketch,
                "sketch": sketch,
            })
            json.dump(prev_result, fps, indent=2, ensure_ascii=False)
        os.remove(f"{res_dir}/{idx}_draft.json")
                
    def run(self, data: dict, res_dir: str, idx: int) -> None:
        """main interface"""
        if not self.judge_if_exist(res_dir, idx):
            draft = self.read_draft(res_dir, idx)
            if draft is not None:
                print(f"   {data['name']} - {idx} sketching...", flush=True)
                raw_sketch = self.llm_generator(data, draft)
                extract_sketch = self.extract_sketch(raw_sketch)
                sketch = self.error_masking(extract_sketch)
                self.save_sketch(res_dir, idx, raw_sketch, sketch)
                print(f"Done Sketch {data['name']} - {idx}", flush=True)
                
    def run_direct(self, data: dict, draft: str) -> str:
        """Interface that directly returns results"""
        print(f"   {data['name']} sketching...", flush=True)
        raw_sketch = self.llm_generator(data, draft)
        extract_sketch = self.extract_sketch(raw_sketch)
        sketch = self.error_masking(extract_sketch)
        print(f"Done Sketch {data['name']}", flush=True)
        return sketch
