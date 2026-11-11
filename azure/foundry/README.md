# 🌐 Azure AI Foundry / Microsoft Foundry Integration

This directory contains the integration for **Azure AI Foundry** (formerly Azure AI Studio) and **Microsoft Foundry Model Catalog**.

Azure AI Foundry provides serverless API endpoints for state-of-the-art foundation models (such as Mistral Large 2407, Meta LLaMA 3.1 70B, and GPT-4o) without managing dedicated VM infrastructure.

---

## 🏗️ Architectural Distinction: Azure ML vs. Azure AI Foundry

| Dimension | Azure Machine Learning (AML) | Azure AI Foundry |
| :--- | :--- | :--- |
| **Primary Purpose** | Full MLOps lifecycle, custom PyTorch training, PEFT LoRA adapter fine-tuning, and compute cluster orchestration. | Serverless foundation model inference, enterprise prompt management, and LLM output evaluation. |
| **Compute Model** | Dedicated/Spot VM clusters (`Standard_NC6s_v3`, `Standard_DS3_v2`) with custom Docker containers. | Pay-per-token Serverless Model Catalog API endpoints. |
| **Model Scope** | Custom multi-head transformer (`VideoEditorialTransformer`) with low-rank adapters. | Frontier foundation models (Mistral Large, LLaMA 3.1, GPT-4o). |
| **Use Case in Project** | Supervised sequence modeling over token embeddings and acoustic feature vectors. | High-level editorial critique, transcript restructuring, and narrative coaching. |

---

## 📁 Directory Structure

```text
azure/foundry/
├── README.md               # Architectural and operational guide
├── configs/
│   └── foundry_config.json # Endpoint URLs, default models, and generation parameters
├── prompts/
│   └── editorial_analysis.json # Versioned prompt templates with strict JSON schema
├── client.py               # Enterprise client using azure-ai-inference SDK
├── inference.py            # Editorial analysis runner using Foundry endpoints
└── evaluation.py           # Multi-model benchmarking and response quality scoring
```

---

## 🔑 Authentication & Configuration

The Foundry client supports two enterprise authentication methods:

1. **Microsoft Entra ID (DefaultAzureCredential):**
   ```bash
   az login
   export AZURE_AI_FOUNDRY_ENDPOINT="https://<your-resource>.services.ai.azure.com/models"
   ```

2. **API Key Fallback:**
   ```bash
   export AZURE_AI_FOUNDRY_ENDPOINT="https://<your-resource>.services.ai.azure.com/models"
   export AZURE_AI_FOUNDRY_API_KEY="<your-foundry-api-key>"
   ```

*Note:* If credentials are not configured, the client automatically falls back to an offline simulation mode. This allows all automated tests and local verification steps to run with zero cost.

---

## 🚀 Execution Commands

### 1. Run Editorial Analysis via Foundry
```bash
python -m azure.foundry.inference --transcript "examples/sample_transcript.json"
```

To specify a specific model from the catalog:
```bash
python -m azure.foundry.inference --model mistral-large-2407
```

### 2. Compare Foundation Model Candidates
Benchmark model candidates (e.g., Mistral Large vs. LLaMA 3.1) on schema compliance, grounded evidence traceability, and latency:
```bash
python -m azure.foundry.evaluation --models mistral-large-2407 Meta-Llama-3.1-70B-Instruct
```
Results will be saved to `outputs/foundry_evaluation/foundry_comparison.json`.

