#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
#
#  This file includes code adapted from:
#  - LeanCopilot (https://github.com/lean-dojo/LeanCopilot/blob/main/python/server.py)
#    Licensed under the MIT License.
#    Modifications made by Microsoft are noted inline or below.

from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from .api_runner import APITacticGenerator


def create_app(cfg) -> FastAPI:
    app = FastAPI()

    models = {
        cfg.prove_sampling_config["name_lean_copilot"]: APITacticGenerator(
            model=cfg.prove_sampling_config["model"],
            base_urls=[config["base_url"] for config in cfg.prove_model_config],
            api_keys=[config["api_key"] for config in cfg.prove_model_config],
            temperature=cfg.prove_sampling_config["temperature"],
            max_tokens=cfg.prove_sampling_config["max_tokens"],
            top_p=cfg.prove_sampling_config["top_p"],
            n=cfg.prove_sampling_config["n"],
            timeout=cfg.prove_sampling_config["timeout"],
            max_output=cfg.prove_sampling_config["max_output"],
            use_beam_search=cfg.prove_sampling_config["use_beam_search"],
        )
    }
    
    for model_name, model in models.items():
        try:
            model.generate("n : ℕ\n⊢ gcd n n = n")
            logger.info(f"Model {model_name} is ready.")
        except Exception as e:
            logger.error(f"Model {model_name} failed to initialize.")
            raise e

    class GeneratorRequest(BaseModel):
        name: str
        input: str
        prefix: Optional[str]

    class Generation(BaseModel):
        output: str
        score: float

    class GeneratorResponse(BaseModel):
        outputs: List[Generation]

    @app.post("/generate")
    async def generate(req: GeneratorRequest) -> GeneratorResponse:
        model = models[req.name]
        target_prefix = req.prefix or ""
        if isinstance(model, APITacticGenerator):
            outputs = await asyncio.to_thread(model.generate, req.input, target_prefix)
        else:
            outputs = model.generate(req.input, target_prefix)
        return GeneratorResponse(
            outputs=[Generation(output=out[0], score=out[1]) for out in outputs]
        )

    return app
