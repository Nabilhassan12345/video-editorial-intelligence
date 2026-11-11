"""
Azure AI Foundry / Microsoft Foundry Model Inference Client.

Connects to Microsoft Foundry Serverless Model Endpoints (Mistral Large, LLaMA 3.1, GPT-4o)
using azure-ai-inference SDK and DefaultAzureCredential or AzureKeyCredential.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AzureFoundryClient:
    """Enterprise client for Azure AI Foundry / Microsoft Foundry Model Catalog endpoints."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        force_simulation: bool = False,
    ):
        self.config = self._load_config(config_path)
        self.endpoint = endpoint or os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        self.force_simulation = force_simulation

        self.client: Optional[ChatCompletionsClient] = None
        if not force_simulation:
            self._init_client()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        cfg_p = Path(config_path or Path(__file__).parent / "configs" / "foundry_config.json")
        if cfg_p.exists():
            with open(cfg_p, "r") as f:
                return json.load(f)
        return {}

    def _init_client(self):
        """Initialize ChatCompletionsClient with Entra ID or Key credential."""
        if not self.endpoint or "video-intelligence-foundry" in str(self.endpoint) or "your-foundry" in str(self.endpoint):
            logger.info("Azure AI Foundry endpoint unconfigured or placeholder. Operating in local simulation mode.")
            self.client = None
            return

        try:
            if self.api_key:
                credential = AzureKeyCredential(self.api_key)
                logger.info(f"Connecting to Foundry endpoint using AzureKeyCredential: {self.endpoint}")
            else:
                credential = DefaultAzureCredential()
                logger.info(f"Connecting to Foundry endpoint using DefaultAzureCredential (Entra ID): {self.endpoint}")

            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=credential,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Azure AI Foundry client: {e}. Simulation mode active.")
            self.client = None

    @property
    def is_live(self) -> bool:
        return self.client is not None

    def complete_editorial_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute chat completion with retry backoff and JSON schema parsing.
        """
        model = model_name or self.config.get("default_model", "mistral-large-2407")

        if not self.is_live:
            logger.info("[Foundry Simulation] Returning grounded mock inference for offline execution.")
            return self._simulate_foundry_response()

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]

        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Foundry model endpoint '{model}' (attempt {attempt + 1}/{max_retries})...")
                response = self.client.complete(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json_object",
                )

                content = response.choices[0].message.content
                parsed = json.loads(content)
                parsed["simulated"] = False
                parsed["model"] = model
                return parsed

            except Exception as e:
                logger.warning(f"Foundry endpoint call failed ({e}). Retrying with backoff...")
                if attempt < max_retries - 1:
                    time.sleep(2.0 ** attempt)
                else:
                    logger.error("Max retries exceeded on Azure AI Foundry endpoint.")
                    raise RuntimeError(f"Azure AI Foundry inference failed after {max_retries} attempts: {e}")

        return self._simulate_foundry_response()

    def _simulate_foundry_response(self) -> Dict[str, Any]:
        """Hardcoded offline example used when no endpoint is configured. NOT model output."""
        return {
            "simulated": True,
            "simulation_note": "No Azure AI Foundry endpoint configured; this is a fixed example response, not model output.",
            "hook": "[00:00] In the next 60 seconds, I'll reveal the exact retention strategy top creators use.",
            "hook_confidence": 0.88,
            "drop_off_risks": [
                {
                    "timestamp": "[00:24]",
                    "risk_score": 0.65,
                    "reason": "Dead air and technical stalling",
                    "explanation": "Spoken cadence ceased while presenter adjusted microphone, creating a 4-second engagement void."
                }
            ],
            "quality_score": 8.4,
            "improvements": [
                "Front-load the core value proposition at [00:00] by immediately showing the end result.",
                "Cut out the hesitation at [00:24] to avoid viewer drop-off."
            ]
        }
