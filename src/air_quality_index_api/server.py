from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from typing import Any
import asyncio
import time
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from src.schema.schema import VienThamData, InferenceResponse

from src.request_handler.request_handler import RequestHandler

class AppState:
    req_handler: RequestHandler
    start_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = AppState
    ctx.start_time = time.time()
    ctx.req_handler = RequestHandler()
    app.state.ctx = ctx
    print("Server started")
    yield

app = FastAPI(
    title="AQI PM2.5 prediction from VienTham data",
    verison="1.0.0",
    lifespan=lifespan
)

# CORS handler
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    uptime = time.time() - app.state.ctx.start_time
    return {
        "status": "OK",
        "uptime_seconds": uptime
    }

@app.post(
    "/predict-pm25-using-lstms2s-and-lstm",
    response_model=InferenceResponse,
    description="test"
)
async def predict_pm25_using_lstms2s_and_lstm(vienthamdata: VienThamData):
    event_loop = asyncio.get_event_loop()
    try:
        outputs = await asyncio.wait_for(
            event_loop.run_in_executor(None, app.state.ctx.req_handler.handle, vienthamdata),
            timeout=3600000 / 1000.0
        )
        res = InferenceResponse(code=200, message="predicted_pm25", data=outputs)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

