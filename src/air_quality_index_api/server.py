from fastapi import FastAPI, HTTPException
from typing import Any
import asyncio
import time
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from schema.schema import InferenceVienThamRequest

from prediction.prediction import Prediction
from schema.schema import TestSchema

class AppState:
    batcher: Any
    model: Any
    start_time: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    ctx = AppState
    ctx.start_time = time.time()
    ctx.prediction = Prediction()
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

@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post(
    "/predict-pm25-using-lstms2s-and-lstm",
    description="test"
)
# async def predict_pm25_using_lstms2s_and_lstm(req: InferenceVienThamRequest):
async def predict_pm25_using_lstms2s_and_lstm(req: TestSchema):
    event_loop = asyncio.get_event_loop()
    try:
        outputs = await asyncio.wait_for(
            event_loop.run_in_executor(None, app.state.ctx.prediction.dummy_action, req),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")
    return outputs

# if ___name__ == "__main__":
#     port = 8000
#     #port = int(os.getenv("SERVER_PORT", "8000"))
#     uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)

