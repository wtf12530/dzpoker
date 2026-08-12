# Poker Suggestion API - Demo

This demo implements the OpenAPI spec (openapi.yaml) and a simple FastAPI server (main.py).
It uses Treys to estimate winrate via Monte Carlo and returns a demo policy suggestion.

Requirements
- Python 3.10+
- Install dependencies:
  pip install -r requirements.txt

Run the server
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Test the /v1/suggest endpoint
  curl -X POST "http://localhost:8000/v1/suggest" -H "Content-Type: application/json" \
    -d @example_request.json

Notes
- This is a demo. The policy is a simple heuristic using Monte Carlo winrate estimate and thresholds.
- For production, replace the policy logic with a trained model (NFSP / DeepCFR / policy+value).
- When using in online environments, ensure you comply with the target platform's terms of service.
