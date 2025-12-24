import re
from fastapi import FastAPI, HTTPException
import asyncio
import time
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from src.schema.schema import NOCMAQRequest, PredictionResponse, CMAQRequest, VienThamRequest, QuanTracRequest, QuanTracSO2Request

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
    title="Air Quality Index API",
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
    "/predict-pm25-from-vientham-using-lstms2s-lstm",
    response_model=PredictionResponse,
    description="Predict PM2.5 values from VienTham data using LSTM-Seq2Seq and LSTM models"
)
async def predict_pm25_from_vientham_using_lstms2s_lstm(vientham_request: VienThamRequest):
    event_loop = asyncio.get_event_loop()
    reduction_model_name = "LSTMSeq2SeqReduction"
    prediction_model_name = "LSTMPrediction"
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleVienThamRequest,
                                       vientham_request,
                                       reduction_model_name,
                                       prediction_model_name),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-pm25-from-vientham-using-grus2s-lstm",
    response_model=PredictionResponse,
    description="Predict PM2.5 values from VienTham data using GRU-Seq2Seq and LSTM models"
)
async def predict_pm25_from_vientham_using_grus2s_lstm(vientham_request: VienThamRequest):
    event_loop = asyncio.get_event_loop()
    reduction_model_name = "GRUSeq2SeqReduction"
    prediction_model_name = "LSTMPrediction"
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleVienThamRequest,
                                       vientham_request,
                                       reduction_model_name,
                                       prediction_model_name),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-pm25-from-vientham-using-cnnlstms2s-lstm",
    response_model=PredictionResponse,
    description="Predict PM2.5 values from VienTham data using CNNLSTM-Seq2Seq and LSTM models"
)
async def predict_pm25_from_vientham_using_cnnlstms2s_lstm(vientham_request: VienThamRequest):
    event_loop = asyncio.get_event_loop()
    reduction_model_name = "CNNLSTMSeq2SeqReduction"
    prediction_model_name = "LSTMPrediction"
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleVienThamRequest,
                                       vientham_request,
                                       reduction_model_name,
                                       prediction_model_name),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-no-from-cmaq-using-lstms2s-lstm",
    response_model=PredictionResponse,
    description="Predict NO values from CMAQ data using LSTM-Seq2Seq and LSTM models"
)
async def predict_no_from_cmaq_using_lstms2s_lstm(cmaq_request: CMAQRequest):
    event_loop = asyncio.get_event_loop()
    reduction_model_name = "LSTMSeq2SeqReduction"
    prediction_model_name = "LSTMPrediction"
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleCMAQRequest,
                                       cmaq_request,
                                       reduction_model_name,
                                       prediction_model_name),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-no2-from-quantrac-using-lightgbm",
    response_model=PredictionResponse,
    description="Predict NO2 values from QuanTrac data using LightGBM models"
)
async def predict_no2_from_quantrac_using_lightgbm(request: QuanTracRequest):
    event_loop = asyncio.get_event_loop()
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleLightGBMRequest,
                                       request,
                                       "NO2_quantrac"),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-o3-from-quantrac-using-lightgbm",
    response_model=PredictionResponse,
    description="Predict O3 values from QuanTrac data using LightGBM models"
)
async def predict_o3_from_quantrac_using_lightgbm(request: QuanTracRequest):
    event_loop = asyncio.get_event_loop()
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleLightGBMRequest,
                                       request,
                                       "O3_quantrac"),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-co-from-quantrac-using-lightgbm",
    response_model=PredictionResponse,
    description="Predict CO values from QuanTrac data using LightGBM models"
)
async def predict_co_from_quantrac_using_lightgbm(request: QuanTracRequest):
    event_loop = asyncio.get_event_loop()
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleLightGBMRequest,
                                       request,
                                       "CO_quantrac"),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-so2-from-quantrac-using-lightgbm",
    response_model=PredictionResponse,
    description="Predict SO2 values from QuanTrac data using LightGBM models"
)
async def predict_so2_from_quantrac_using_lightgbm(request: QuanTracSO2Request):
    event_loop = asyncio.get_event_loop()
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleLightGBMRequest,
                                       request,
                                       "SO2_quantrac"),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res

@app.post(
    "/predict-no-from-cmaq-using-lightgbm",
    response_model=PredictionResponse,
    description="Predict NO values from CMAQ data using LightGBM models"
)
async def predict_no_from_cmaq_using_lightgbm(request: NOCMAQRequest):
    event_loop = asyncio.get_event_loop()
    try:
        res = await asyncio.wait_for(
            event_loop.run_in_executor(None,
                                       app.state.ctx.req_handler.handleLightGBMRequest,
                                       request,
                                       "NO_cmaq"),
            timeout=3600000 / 1000.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out")

    return res
