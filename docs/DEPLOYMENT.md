# Deployment

Nothing is deployed by this repository. All options below are opt-in.

## Local

~~~bash
python -m src.cli analyze --transcript examples/sample_transcript.json --output-dir outputs/reports
python -m src.cli demo                 # Streamlit UI on http://localhost:8501
cd server && npm install && npm start  # Express API (calls the Python pipeline)
cd client && npm install && npm run dev
~~~

The Express server (`server/services/pythonRunner.js`) runs `python -m src.cli analyze` (uses `.venv/bin/python` if present, override with `PYTHON_BIN`, timeout `PYTHON_PIPELINE_TIMEOUT_MS`, disable with `DISABLE_PYTHON_PIPELINE=true`). If Python fails it falls back to OpenAI (`OPENAI_API_KEY`) or, without a key, to an illustrative mock. The response field `source` is `python_model`, `openai` or `mock` and the client shows a warning for mock / untrained results.

## Azure ML managed online endpoint (optional, billed)

`azure/ml/deploy_model.py` builds a `ManagedOnlineEndpoint` + `ManagedOnlineDeployment` (default `Standard_DS3_v2`, 1 instance) and a scoring script. It prints the spec by default; only `--deploy` creates cloud resources.

~~~bash
python -m azure.ml.register_model --checkpoint-dir checkpoints/editorial_model             # dry run
python -m azure.ml.register_model --checkpoint-dir checkpoints/editorial_model --register  # creates model asset
python -m azure.ml.deploy_model --endpoint video-editorial-endpoint                         # dry run
python -m azure.ml.deploy_model --endpoint video-editorial-endpoint --deploy                # billed
~~~

Requires `azure/ml/config.json` (copy `config.json.example`) and `az login`. Delete the endpoint after use to stop charges. These commands have not been run against a live workspace from this repo.
