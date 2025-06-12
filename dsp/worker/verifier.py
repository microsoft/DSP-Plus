#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
#
#  This file includes code adapted from:
#  - DeepSeek-Prover-V1.5 (https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/prover/lean/verifier.py)
#    Licensed under the MIT License.
#    Modifications made by Microsoft are noted inline or below.

import os
import re
import time
import json
import signal
import ctypes
import pexpect
import resource
import warnings
import traceback
import subprocess
import multiprocessing as mp
from pprint import pprint
from typing import Union
from easydict import EasyDict as AttrDict

from .scheduler import ProcessScheduler


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'
RETRIES = 2
DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat"
DEFAULT_COMMAND = '{"cmd": "import Mathlib\\nimport Aesop\\n\\nset_option maxHeartbeats 0\\n\\nopen BigOperators Real Nat Topology Rat"}'


def remove_specific_string(target_str: str, input_str: str):
    return input_str.replace(target_str, "/-" + target_str[2:-2] + "-/")


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, timeout=300, allTactics=True, ast=False, premises=False, tactics=False, infotree=str(), p_idx: int=-1, t_idx: Union[int, str]=-1, server=None, share_header=DEFAULT_HEADER) -> dict:
    command = dict(cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    if infotree:
        command['infotree'] = infotree
    if last_env is not None:
        command.update(env=last_env)
    if share_header:
        command['cmd'] = remove_specific_string(share_header, command['cmd'])
    message_str = json.dumps(command, ensure_ascii=False)

    start_time = time.time()
    system_messages = ''
    try:
        if time.time() - server.start_time > 360:
            raise MemoryError("restart REPL to clean memory")

        server.proc.sendline(message_str)
        server.proc.expect_exact(message_str + "\r\n")
        server.proc.sendline()
        server.proc.expect_exact("\r\n")
        server.proc.expect_exact("\r\n\r\n")
        output = server.proc.before

        # if output start with "Exception: ", go to exception handling
        if "Exception: " in output:
            raise SyntaxError(output)

        # Exclude progress bars and other miscellaneous items
        start = output.find('{')
        if start != -1:
            output = output[start:]
        result = json.loads(output)

        if 'ast' in result and result['ast']:
            warnings.warn("No need to parse the AST")
        ast_results = {}
        
        result = {
            "sorries" : result.get('sorries', []), 
            "tactics" : result.get('tactics', []),
            "errors" : [m for m in result.get('messages', []) if m['severity'] == 'error'],
            "warnings" : [m for m in result.get('messages', []) if m['severity'] == 'warning'],
            "infos" : [m for m in result.get('messages', []) if m['severity'] == 'info'],
            "system_messages" : system_messages,
            "system_errors" : None,
            "ast" : ast_results,
            "infotree": result.get('infotree', []),
            "verified_code" : code,
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
        result['verify_time'] = time.time() - start_time
        
    except Exception as e:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc()
        }
        result['verify_time'] = time.time() - start_time
        
        if isinstance(e, SyntaxError):
            if verbose:
                print(f"LeanProcess {p_idx} Theorem {t_idx} REPL SyntaxError error: {repr(e)}")
            result["system_messages"] = 'SyntaxError'
        elif isinstance(e, pexpect.exceptions.EOF):
            if verbose:
                print(f"LeanProcess {p_idx} Theorem {t_idx} REPL error: EOF")
            result["system_messages"] = 'pexpect.exceptions.EOF'
        elif isinstance(e, pexpect.exceptions.TIMEOUT):
            if verbose:
                print(f"LeanProcess {p_idx} Theorem {t_idx} REPL error: TIMEOUT")
            result["system_messages"] = 'pexpect.exceptions.TIMEOUT'
        elif isinstance(e, json.decoder.JSONDecodeError):
            if verbose:
                print(f"LeanProcess {p_idx} Theorem {t_idx} REPL error: JSONDecodeError")
            result["system_messages"] = 'json.decoder.JSONDecodeError'
        elif isinstance(e, MemoryError):
            result["system_messages"] = 'MemoryError'
        else:
            if verbose:
                print(f"LeanProcess {p_idx} Theorem {t_idx} REPL error: {e}")
            result["system_messages"] = str(e)
            
    return result


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict(), command=DEFAULT_COMMAND, share_header=DEFAULT_HEADER, cwd=DEFAULT_LEAN_WORKSPACE):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
        self.command = command
        self.share_header = share_header
        self.cwd = cwd
        
    def start_lean4_server(self):
        self.start_time = time.time()
        self.proc = pexpect.spawn(
            f"{DEFAULT_LAKE_PATH} exe repl", cwd=self.cwd, encoding="utf-8", timeout=self.timeout
        )
        output = self.proc.before
        if self.command:
            self.proc.sendline(self.command) 
            self.proc.expect_exact(self.command + "\r\n") 
            self.proc.sendline()
            self.proc.expect_exact("\r\n")
            self.proc.expect_exact("\r\n\r\n")
            output = self.proc.before
        return output
    
    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        try:
            output = self.start_lean4_server()
            last_env = 0
            while True:
                inputs = self.task_queue.get()
                if inputs is None: # Terminate when receiving None
                    break
                for _, request_id, task in inputs:
                    if isinstance(task, str):
                        task = dict(code=task)

                    task['server'] = self
                    if self.share_header:
                        task['share_header'] = self.share_header
                    if self.command:
                        task['last_env'] = last_env
                    task['p_idx'] = self.idx
                    
                    # Match the theorem name for better positioning 
                    match = re.search(r'(theorem|lemma|example)\s+(\S+)', task['code'])
                    if match:
                        task['t_idx'] = match.group(2)
                    else:
                        task['t_idx'] = "unknown"
                    
                    result = verify_lean4_file(**task)
                    attempt_num = 1
                    retry_start_time = time.time()
                    
                    # Perform different operations based on the type of error
                    while len(result['system_messages']) > 0:
                        if result['system_messages'] == 'pexpect.exceptions.TIMEOUT':
                            self.proc.kill(signal.SIGKILL)
                            self.proc.close()
                            output = self.start_lean4_server()
                            break
                        elif result['system_messages'] == 'SyntaxError':
                            break
                        elif result["system_messages"] == 'MemoryError':
                            self.proc.kill(signal.SIGKILL)
                            self.proc.close()
                            output = self.start_lean4_server()
                            result = verify_lean4_file(**task)
                        elif time.time() - retry_start_time < self.timeout and attempt_num < RETRIES:
                            self.proc.kill(signal.SIGKILL)
                            self.proc.close()
                            time.sleep(0.5)
                            output = self.start_lean4_server()
                            result = verify_lean4_file(**task)
                            attempt_num += 1
                        else:
                            self.proc.kill(signal.SIGKILL)
                            self.proc.close()
                            output = self.start_lean4_server()
                            print(f"LeanProcess {self.idx} Theorem {task['t_idx']} REPL restart timeout or too many")
                            break
                    result['p_idx'] = self.idx
                    
                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1
            self.proc.kill(signal.SIGKILL)
            self.proc.close()
            
        except Exception as e:
            if isinstance(e, pexpect.exceptions.EOF):
                print(f"LeanProcess {self.idx} Lean4Server failed to start, please check the status of mathlib4 and REPL")
            elif isinstance(e, pexpect.exceptions.TIMEOUT):
                print(f"LeanProcess {self.idx} Lean4Server startup timeout")
            else:
                print(f"LeanProcess {self.idx} Lean4Server encountered an exception: {e}")

            self.proc.kill(signal.SIGKILL)
            self.proc.close()


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier', command=DEFAULT_COMMAND, share_header=DEFAULT_HEADER, cwd=DEFAULT_LEAN_WORKSPACE):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                ),
                command=command,
                share_header=share_header,
                cwd=cwd
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Complete launching {len(self.processes)} {self.name} LeanServerProcesses')

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        # self._monitor_process = mp.Process(target=self._monitor)
        # self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(['killall', 'repl', f'--older-than={int(self.timeout) + 10}s'], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        # self._monitor_process.join()
        print(f'All {len(self.processes)} {self.name} LeanServerProcesses stopped')


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.code.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    pprint(outputs_list)