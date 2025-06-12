#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import json
from typing import List

from .worker import LLMServerScheduler


class Draft:
    llm_scheduler = None

    @classmethod
    def launch_llmserver(cls, client_config: List[dict], max_llm_requests: int=128, name = 'Draft', **kwargs):
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
    
    def llm_generator(self, data: dict) -> str:
        """get raw output from llm"""
        # get what you need, we just need formal_statement
        formal_statement = data['formal_statement']
        
        # get llm prompt
        prompt = f'''formal_statement:
{formal_statement}
Please provide an extremely detailed mathematical calculation following your thinking. Each step can only contain **one** equation without any explanation.

Here is an example:
### Step 1: 
\[ x + y + xy = 80 \]
...
### Step 5: 
\[ x + y + xy + 1 = 81 \]
'''
    
        # get llm output
        request_id = self.llm_scheduler.submit_request(prompt)
        output = self.llm_scheduler.get_request_outputs(request_id)
        if not output:
            raise RuntimeError("Draft LLM request failed")
        else:
            return output
        
    def post_process(self, raw_draft: str) -> str:
        """remove thinking token, not affect non-reasoning model"""
        draft = raw_draft.split('</think>')[-1].strip()
        return draft
    
    
    def judge_if_exist(self, res_dir: str, idx: int) -> bool:
        """judge whether the historical file exists"""
        return os.path.exists(os.path.join(res_dir, f"{idx}_draft.json")) \
            or os.path.exists(os.path.join(res_dir, f"{idx}_sketch.json")) \
                or os.path.exists(os.path.join(res_dir, f"{idx}_prove.json"))
                
    def save_draft(self, res_dir: str, idx: int, raw_draft: str, draft: str) -> None:
        """save draft to file"""
        with open(f"{res_dir}/{idx}_draft.json", 'w', encoding='utf-8') as fpd:
            json.dump({
                "raw_draft": raw_draft,
                "draft": draft,
            }, fpd, indent=2, ensure_ascii=False)
                
    def run(self, data: dict, res_dir: str, idx: int) -> None:
        """main interface"""
        if not self.judge_if_exist(res_dir, idx):
            print(f"   {data['name']} - {idx} drafting...", flush=True)
            raw_draft = self.llm_generator(data)
            draft = self.post_process(raw_draft)
            self.save_draft(res_dir, idx, raw_draft, draft)
            print(f"Done Draft {data['name']} - {idx}", flush=True)
            
    def run_direct(self, data: dict) -> str:
        """Interface that directly returns results"""
        print(f"   {data['name']} drafting...", flush=True)
        raw_draft = self.llm_generator(data)
        draft = self.post_process(raw_draft)
        print(f"Done Draft {data['name']}", flush=True)
        return draft