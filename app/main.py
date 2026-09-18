from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .llm import LLMServiceError, interpreter
from .models import OptimizeRequest, OptimizeResponse
from .optimizer import OptimizationError, optimize_energy

app = FastAPI(
    title="GridWise BUP CSE Fest 2026",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    # The challenge contract specifies 400 for malformed/structurally invalid requests.
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "message": "Request JSON does not match the required schema."},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    try:
        directives = await interpreter.interpret(request.operator_notes, request.battery)
        optimized = optimize_energy(request, directives)
    except LLMServiceError:
        raise HTTPException(
            status_code=500,
            detail={"error": "llm_unavailable", "message": "Operator-note interpretation failed safely."},
        )
    except OptimizationError:
        raise HTTPException(
            status_code=500,
            detail={"error": "optimization_failed", "message": "A valid schedule could not be produced."},
        )

    directive_names = [d.directive_type for d in directives if d.applies]
    if directive_names:
        applied_text = ", ".join(directive_names)
        summary = (
            f"Applied validated operator directives ({applied_text}), then minimized grid-electricity cost "
            "while enforcing solar, battery, hourly energy-balance, and end-of-day neutrality constraints."
        )
    else:
        summary = (
            "No operator note changed the energy constraints. The schedule minimizes grid-electricity cost "
            "while enforcing solar, battery, hourly energy-balance, and end-of-day neutrality constraints."
        )

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directives,
        hourly_plan=optimized["hourly_plan"],
        total_grid_kwh=optimized["total_grid_kwh"],
        total_cost_bdt=optimized["total_cost_bdt"],
        peak_grid_kwh=optimized["peak_grid_kwh"],
        plan_summary=summary,
    )
