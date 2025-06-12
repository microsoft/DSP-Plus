#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import time
import openai
import warnings
import threading
import multiprocessing as mp
from multiprocessing.synchronize import Lock as MpLock
from typing import Optional, Dict, List, Any


from .scheduler import TaskQueue, ProcessScheduler


class LLMServerProcess(mp.Process):
    """
    A multiprocessing process responsible for managing and distributing requests to OpenAI clients 
    with load balancing, retry, and failover logic.

    Args:
        task_queue (TaskQueue): A multiprocessing queue holding incoming tasks to process.
        request_statuses (Dict[str, Any]): A shared dictionary storing the results of completed requests.
        lock (MpLock): A multiprocessing lock for synchronizing access to shared resources.
        
        client_lib (Dict[str, List[dict]]): A mapping from string index to OpenAI clients.
        load_running_requests (Dict[str, Any]): Tracks current running requests per client.
        load_max_requests (Dict[str, Any]): Maximum number of concurrent requests allowed per client.
        load_disabled_time (Dict[str, Any]): Tracks the last disabled time of a client (None if active).
        load_lock (MpLock): A multiprocessing lock to synchronize access to load-related structures.
        
        model (str): Model name to be used (e.g., "gpt-4", "gpt-3.5-turbo").
        max_retries (Optional[int]): Number of retry attempts before giving up. None means infinite retries.
        temperature (float): Sampling temperature for OpenAI completions.
        top_p (float): Top-p nucleus sampling parameter.
        max_tokens (int): Maximum number of tokens in the response.
        timeout (int): Timeout in seconds for the OpenAI request.

        max_running_requests (int): Maximum allowed load per client.
        wait_time (int): Time (in seconds) to wait before retrying when all client are busy or disabled.
        redo_time (int): Time (in seconds) to wait before retrying a failed request.
        recovery_time (int): Time (in seconds) after which a disabled client is considered for recovery.
    """
    
    def __init__(self,
                task_queue: TaskQueue, 
                request_statuses: Dict[str, Any], 
                lock: MpLock, 
                
                client_lib: Dict[str, List[dict]],
                load_running_requests: Dict[str, Any],
                load_max_requests: Dict[str, Any],
                load_disabled_time: Dict[str, Any],
                load_lock: MpLock,
                
                model: str,
                max_retries: Optional[int] = None,
                temperature: float = 0.6,
                top_p: float = 0.95,
                max_tokens: int = 32768,
                timeout: int = 1800,
                
                max_running_requests: int = 128,
                wait_time: int = 600,
                redo_time: int = 30,
                recovery_time: int = 1800,
                ):
        super().__init__()
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.wait_time = wait_time
        self.redo_time = redo_time
        self.recovery_time = recovery_time
        
        self.max_running_requests = max_running_requests
        self.client_lib = client_lib
        self.load_running_requests = load_running_requests
        self.load_max_requests = load_max_requests
        self.load_disabled_time = load_disabled_time
        self.load_lock = load_lock
        
    def send_request(self, prompt: str, client: openai.Client) -> str:
        """Sends a single request to the OpenAI client"""
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model = self.model,
            messages = messages,
            # n = 1,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            top_p = self.top_p,
            timeout = self.timeout,
        )
        return completion.choices[0].message.content
    
    def recover_disabled_client(self) -> None:
        """Check all disabled clients and recover those whose downtime has exceeded recovery_time"""
        with self.load_lock:
            current_time = time.time()
            for idx, disabled_time in self.load_disabled_time.items():
                if disabled_time is not None and current_time - disabled_time >= self.recovery_time:
                    print(f"Client {idx} recovered to load 1")
                    self.load_disabled_time[idx] = None
                    self.load_max_requests[idx] = 1
                    
    def send_and_return(self, prompt: str, request_id: int) -> None:
        """
        Send the request using a load-balanced client, with retry and failure handling.
        Also records the result into request_statuses.
        """
        attempt = 0
        while self.max_retries is None or attempt < self.max_retries:
            AVAILABLE = False
            
            with self.load_lock:
                # Filter out available services (not disabled and the number of requests does not exceed the maximum load)
                available_ports = [p for p in self.client_lib.keys() if self.load_disabled_time[p] is None and self.load_running_requests[p] < self.load_max_requests[p]]
                
                if available_ports:
                    # Select the available service with the least current request count
                    available_port = min(available_ports, key=lambda p: self.load_running_requests[p])
                    self.load_running_requests[available_port] += 1
                    AVAILABLE = True
                
            if AVAILABLE:
                try:
                    result = self.send_request(prompt, self.client_lib[available_port])
                    with self.load_lock:
                        self.load_running_requests[available_port] -= 1
                    # Request successful, increase the maximum load for this service (if the limit has not been reached)
                    with self.load_lock:
                        if self.load_max_requests[available_port] < self.max_running_requests:
                            self.load_max_requests[available_port] += 1
                    # print(f"send request to Client {available_port}")
                    # Records the result into request_statuses
                    with self.lock:
                        self.request_statuses[request_id] = result
                    return
                
                except Exception as e:
                    with self.load_lock:
                        self.load_running_requests[available_port] -= 1
                    
                    if isinstance(e, openai.BadRequestError):
                        # If the error comes from the local, then return an empty string
                        warnings.warn(f"Client {available_port} received BadRequestError: {repr(e)}")
                        with self.lock:
                            self.request_statuses[request_id] = str()
                        return
                    else:
                        # If the error comes from the server, then retry
                        print(f"Client {available_port} encountered exception: {repr(e)}")
                        # When the request fails, reduce the maximum load of the service (if it is greater than 0)
                        with self.load_lock:
                            if self.load_max_requests[available_port] > 0:
                                self.load_max_requests[available_port] -= 1
                            # If the maximum load drops to 0, disable the service
                            if self.load_max_requests[available_port] == 0:
                                self.load_disabled_time[available_port] = time.time()
                                # print(f"Client {available_port} has been disabled")
                        time.sleep(self.redo_time)
            else:
                # If all services are full or disabled, please wait
                print(f"Model {self.model}: all clients are busy or disabled, waiting...")
                time.sleep(self.wait_time)
                
            # Check if the disabled service can be restored
            self.recover_disabled_client()
            attempt += 1
            
            if attempt % 10 == 0 or attempt == self.max_retries:
                warnings.warn(f"Some requests to the {self.model} have failed {attempt} times, please pay attention to the logs!")
        
        # After multiple failures without results, return an empty string
        # Notice that to prevent the LLM process from crashing, we delegate exception handling to the upper layer
        with self.lock:
            self.request_statuses[request_id] = str()
        return
                    
    def run(self):
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            
            for _, request_id, prompt in inputs:
                thread = threading.Thread(target=self.send_and_return, args=(prompt, request_id), daemon=True)
                thread.start()
                
                
class LLMServerScheduler(ProcessScheduler):
    def __init__(self, client_config: List[dict], max_running_requests_per_client=128, name='llm', **kwargs):
        super().__init__(batch_size=1, name=name)
        
        self.client_lib = {str(i): openai.Client(**client) for i, client in enumerate(client_config)}
        self.load_running_requests = self.manager.dict({str(i): 0 for i, client in enumerate(client_config)})
        self.load_max_requests = self.manager.dict({str(i): max_running_requests_per_client for i, client in enumerate(client_config)})
        self.load_disabled_time = self.manager.dict({str(i): None for i, client in enumerate(client_config)})
        self.load_lock = mp.Lock()
        
        self.process = LLMServerProcess(
            task_queue = self.task_queue,
            request_statuses = self.request_statuses,
            lock = self.lock,
            
            max_running_requests = max_running_requests_per_client,
            client_lib = self.client_lib,
            load_running_requests = self.load_running_requests,
            load_max_requests = self.load_max_requests,
            load_disabled_time = self.load_disabled_time,
            load_lock = self.load_lock,
            
            **kwargs
        )
        
        self.process.start()
        print(f'Complete launching {self.name} LLMServerProcesses')
    
    def close(self):
        super().close()
        self.process.join()
        print(f'{self.name} LLMServerProcesses stopped')