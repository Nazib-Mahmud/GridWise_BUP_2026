# GridWise — BUP CSE Fest 2026 Preliminary

A deployable HTTP API for the **LLM-Assisted Smart Campus Energy Optimization** preliminary challenge.

## Architecture

`operator_notes -> LLM -> deterministic guardrails -> directive constraints -> LP optimizer -> replay validation -> JSON response`

The LLM is **mandatory and directly produces the structured `directive_interpretation` used by the optimizer**. Deterministic code validates every model output before it can affect optimization.

## Required endpoints

- `GET /health`
- `POST /optimize-energy`

`GET /health` returns:

```json
{"status":"ok"}
```

## Technology

- Python 3.12
- FastAPI
- LLM provider: Gemini, OpenAI, or an OpenAI-compatible endpoint
- SciPy HiGHS linear-programming optimizer
- Deterministic schema/guardrail validation
- Docker

## Supported directives

- `solar_reduction`
- `minimum_battery_reserve`
- `no_charge_window`
- `no_discharge_window`
- `max_grid_window`
- `no_op`

## Local quickstart

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`, fill in your provider/model/API key, and **do not commit `.env`**. The service loads `.env` automatically for local development.

Gemini example (PowerShell):

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="YOUR_KEY"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

OpenAI example (PowerShell):

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="YOUR_MODEL_NAME"
```

For another OpenAI-compatible provider, use `LLM_PROVIDER=openai_compatible` and set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.

### 4. Start the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Check health

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

## Public sample validation

The organizer-provided 10-case JSON pack is included at:

`samples/public_sample_cases.json`

### Test optimizer + official reference directives (no LLM/API key required)

```bash
python scripts/test_optimizer_samples.py
```

Expected final line:

```text
Optimizer public samples: 10/10 passed
```

### Test the complete live LLM -> guardrail -> optimizer API

First start the server with a real LLM key, then in another terminal run:

```bash
python scripts/test_live_api_samples.py --base-url http://127.0.0.1:8000
```

This checks both the LLM semantics and optimal cost against the public pack. Public phrases/case IDs/reference schedules are **not** used by the production interpretation code.

## Example POST request

Use any `case.input` object from `samples/public_sample_cases.json`:

```bash
curl -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data @request.json
```

Successful response fields:

- `scenario_id`
- `directive_interpretation`
- `hourly_plan` (24 entries)
- `total_grid_kwh`
- `total_cost_bdt`
- `peak_grid_kwh`
- `plan_summary`

## Deterministic guardrails

Before optimization, the service enforces:

- exactly one interpretation per note
- `note_index` coverage/order
- supported directive types only
- exact `structured_adjustment` shape
- `no_op` => `applies=false` + `null`
- all other directives => `applies=true`
- unique integer hours 0..23, normalized to ascending order
- solar factor in [0,1]
- reserve within battery capacity
- non-negative finite grid cap

Malformed or unsupported LLM output is never silently converted into a new constraint. The service retries model interpretation and otherwise fails safely.

## Optimizer rules

The LP minimizes:

`sum(grid_kwh[h] * tariff_bdt_per_kwh[h])` for h=0..23

while enforcing:

- demand/energy balance every hour
- effective-solar upper bound
- battery capacity and active reserve floor
- hourly charge/discharge limits
- no-charge/no-discharge windows
- max-grid windows
- end-of-day battery energy = initial battery energy

The returned plan is replayed deterministically before the API responds.

## Docker fallback

Build:

```bash
docker build -t gridwise-bup-2026:1.0 .
```

Run with Gemini:

```bash
docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY="YOUR_KEY" \
  -e GEMINI_MODEL="gemini-2.5-flash" \
  gridwise-bup-2026:1.0
```

Then verify:

```bash
curl http://127.0.0.1:8000/health
```

For submission, push a tested image to Docker Hub/GHCR and submit an exact tag or digest. Do not bake credentials into the image.

## Render deployment

A starter `render.yaml` is included. Create a web service from the repository, add `GEMINI_API_KEY` as a secret environment variable, deploy, then test both endpoints **from outside your development machine**.

## Security

- Never commit `.env`, API keys, tokens, or passwords.
- Provider errors are returned as controlled generic failures; raw secrets/prompts are not returned by the API.
- This solution uses only synthetic request data.

## Reliability notes

- LLM temperature is 0.
- JSON output is requested from the provider.
- Model output is deterministically validated.
- One repair/retry path is available when configured with `LLM_MAX_ATTEMPTS=2`.
- Repeated identical note+battery interpretations are cached in memory to reduce latency/provider load.
- The optimizer is deterministic and very small (24 hours), using SciPy HiGHS.

## Known limitations

- A hosted LLM still depends on provider credentials, quota, rate limits, and provider availability.
- The service does not fine-tune or train a model at runtime.
- The API intentionally has no dashboard/login because the judging contract requires direct public HTTP access.
- Exact hidden-test language performance depends on the chosen language model.

## Pre-submission checklist

1. `GET /health` works publicly.
2. `POST /optimize-energy` works publicly with 1–3 notes.
3. Run all 10 public cases through the live API.
4. Confirm no secrets exist in Git history or Docker image.
5. Confirm README commands work on a clean machine.
6. Push a tested Docker fallback image with an exact tag/digest.
7. Keep the event repository private during the event and make it public only according to the official submission rule.
8. Prepare the <=3-minute architecture/solution video.
