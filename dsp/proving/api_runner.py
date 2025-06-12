#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from loguru import logger
from typing import List, Tuple
import random
import openai

from .external_parser import *


class RequestSender:
    def __init__(self, model: str, base_urls: List[str], api_keys: List[str], timeout: int):
        self.model = model
        self.base_urls = base_urls
        self.api_keys = api_keys
        self.timeout = timeout

    def send_request_once_openai(self, prompt: str, base_url: str, api_key: str, sampling_params: dict):
        # print(f"Sending request to url {base_url}...", flush=True)
        client = openai.Client(
            base_url = base_url,
            api_key = api_key,
        )
        use_beam_search = sampling_params.pop('use_beam_search', False)
        response = client.completions.create(
            model = self.model,
            prompt = prompt,
            logprobs = 1,
            **sampling_params,
            extra_body={
                "use_beam_search": use_beam_search,
            }
        )
        # print(f"   Received response from url {base_url}.", flush=True)
        return response

    def generate(self, prompt: str, sampling_params: dict):
        for _ in range(3):
            select_idx = random.choice(range(len(self.base_urls)))
            base_url = self.base_urls[select_idx]
            api_key = self.api_keys[select_idx]
            try:
                return self.send_request_once_openai(prompt, base_url, api_key, sampling_params)
            except Exception as e:
                pass
        raise RuntimeError("Failed to get response from API.")


class APITacticGenerator(Generator):
    def __init__(self, **args) -> None:
        self.name = args["model"]

        if "base_urls" not in args:
            raise ValueError("base_urls is required")

        self.llm = RequestSender(
            model = self.name,
            base_urls = args["base_urls"],
            api_keys = args.get('api_keys', ["EMPTY"]),
            timeout = args.get('timeout', 1800),
        )

        self.sampling_params = dict(
            n=args["n"],
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
            frequency_penalty=0,
            presence_penalty=0,
            use_beam_search=args.get("use_beam_search", False),
        )

        self.max_output = args.get("max_output", -1)
        logger.info(f"Loading {self.name} using API on {self.llm.base_urls}")

    def generate(self, input: str, target_prefix: str = "") -> List[Tuple[str, float]]:
        prompt = input + target_prefix
        prompt = pre_process_input(self.name, prompt)

        api_outputs = self.llm.generate(prompt, self.sampling_params)

        result = []
        for output in api_outputs.choices:  # bsz=1 for now
            out = output.text
            result.append(
                (out, sum(output.logprobs.token_logprobs))
            )

        result = choices_dedup(result)
        if self.max_output > 0:
            result = sample_outputs_by_logprob(result, self.max_output)
        return result
