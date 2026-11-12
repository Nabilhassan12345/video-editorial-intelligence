# Azure AI Foundry Integration

Optional LLM critique of a transcript through a chat-completion model you deploy in Azure AI Foundry. It is **separate** from the fine-tuned DistilBERT model and is **not required** to run the project.

> Status: client code and config only. It has **not** been run against a live Foundry endpoint in this repository; no Foundry results are reported anywhere.

## Files

| File | Purpose |
|---|---|
| `azure/foundry/client.py` | `FoundryEditorialClient`: wraps `azure.ai.inference.ChatCompletionsClient`, JSON response format, retry with exponential backoff |
| `azure/foundry/inference.py` | CLI: run one transcript through the configured model |
| `azure/foundry/evaluation.py` | CLI: compare candidate models on response structure / grounding checks |
| `azure/foundry/configs/`, `azure/foundry/prompts/` | Default model name and system/user prompts |

## Configuration

Set in `.env` (see `.env.example`):

~~~bash
AZURE_AI_FOUNDRY_ENDPOINT=https://<your-resource>.services.ai.azure.com/models
AZURE_AI_FOUNDRY_API_KEY=<key>   # optional; omit to use DefaultAzureCredential (Entra ID / az login)
~~~

Authentication: `AzureKeyCredential` when a key is set, otherwise `DefaultAzureCredential`.

## Usage

~~~bash
python -m azure.foundry.inference --transcript examples/sample_transcript.json --model <deployment-name>
python -m azure.foundry.evaluation --models <model-a> <model-b>
~~~

Calling a live endpoint is billed per token by Azure.

## Offline simulation

If no endpoint is configured (or it is a placeholder), the client returns a **fixed example response** so the code path can be exercised in tests/CI. Every response carries `"simulated": true|false`; simulated output is not model output and must not be reported as a result.

## Error handling

- Up to `max_retries` (default 3) attempts with `2^attempt` second backoff, then `RuntimeError`.
- Responses are requested as `json_object` and parsed with `json.loads`; a malformed response counts as a failed attempt.
