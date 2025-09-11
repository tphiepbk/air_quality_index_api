from pydantic import BaseModel, Field
from typing import Any, List, Optional

from typing import Optional, List
from pydantic import BaseModel, Field

class VienThamAttributes(BaseModel):
    station:           Optional[List[float]] = Field(default=None, description="list of station values, at least 7 values")
    pm25:              Optional[List[float]] = Field(default=None, description="list of daily pm25 values, at least 7 values")
    lat:               Optional[List[float]] = Field(default=None, description="list of lat values, at least 7 values")
    lon:               Optional[List[float]] = Field(default=None, description="list of lon values, at least 7 values")
    tmp:               Optional[List[float]] = Field(default=None, description="list of daily tmp values, at least 7 values")
    rh:                Optional[List[float]] = Field(default=None, description="list of daily rh values, at least 7 values")
    hpbl:              Optional[List[float]] = Field(default=None, description="list of daily hpbl values, at least 7 values")
    wspd:              Optional[List[float]] = Field(default=None, description="list of daily wspd values, at least 7 values")
    pop:               Optional[List[float]] = Field(default=None, description="list of pop values, at least 7 values")
    road_den_1km:      Optional[List[float]] = Field(default=None, description="list of road_den_1km values, at least 7 values")
    prim_road_len_1km: Optional[List[float]] = Field(default=None, description="list of prim_road_len_1km values, at least 7 values")
    near_dist:         Optional[List[float]] = Field(default=None, description="list of near_dist values, at least 7 values")
    bareland:          Optional[List[float]] = Field(default=None, description="list of bareland values, at least 7 values")
    builtup:           Optional[List[float]] = Field(default=None, description="list of builtup values, at least 7 values")
    cropland:          Optional[List[float]] = Field(default=None, description="list of cropland values, at least 7 values")
    grassland:         Optional[List[float]] = Field(default=None, description="list of grassland values, at least 7 values")
    treecover:         Optional[List[float]] = Field(default=None, description="list of treecover values, at least 7 values")
    water:             Optional[List[float]] = Field(default=None, description="list of water values, at least 7 values")
    ndvi:              Optional[List[float]] = Field(default=None, description="list of ndvi values, at least 7 values")
    aod:               Optional[List[float]] = Field(default=None, description="list of daily aod values, at least 7 values")

class InferenceVienThamRequest(BaseModel):
    data: dict[str, VienThamAttributes] = Field(..., description="input data to predict")

class PredictionResponse(BaseModel):
    value_1d: Optional[List[float]] = Field(default=None, description="1 next day predicted values")
    value_2d: Optional[List[float]] = Field(default=None, description="2 next days predicted values")
    value_3d: Optional[List[float]] = Field(default=None, description="3 next days predicted values")

class InferenceResponse(BaseModel):
    code: int = Field(..., description="error code. 0 means success otherwise means fail")
    message: str  = Field(..., description="error message")
    data: PredictionResponse = Field(..., description="prediction response")

class TestSchema(BaseModel):
    data: int
