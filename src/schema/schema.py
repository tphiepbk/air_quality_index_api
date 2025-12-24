from pydantic import BaseModel, Field
from typing import Any, List, Optional

from typing import Optional, List
from pydantic import BaseModel, Field

class VienThamInputData(BaseModel):
    pm25:              Optional[List[float]] = Field(default=None, description="list of daily pm25 values, 7 values")
    lat:               Optional[List[float]] = Field(default=None, description="list of lat values, 7 values")
    lon:               Optional[List[float]] = Field(default=None, description="list of lon values, 7 values")
    tmp:               Optional[List[float]] = Field(default=None, description="list of daily tmp values, 7 values")
    rh:                Optional[List[float]] = Field(default=None, description="list of daily rh values, 7 values")
    hpbl:              Optional[List[float]] = Field(default=None, description="list of daily hpbl values, 7 values")
    wspd:              Optional[List[float]] = Field(default=None, description="list of daily wspd values, 7 values")
    pop:               Optional[List[float]] = Field(default=None, description="list of pop values, 7 values")
    road_den_1km:      Optional[List[float]] = Field(default=None, description="list of road_den_1km values, 7 values")
    prim_road_len_1km: Optional[List[float]] = Field(default=None, description="list of prim_road_len_1km values, 7 values")
    near_dist:         Optional[List[float]] = Field(default=None, description="list of near_dist values, 7 values")
    bareland:          Optional[List[float]] = Field(default=None, description="list of bareland values, 7 values")
    builtup:           Optional[List[float]] = Field(default=None, description="list of builtup values, 7 values")
    cropland:          Optional[List[float]] = Field(default=None, description="list of cropland values, 7 values")
    grassland:         Optional[List[float]] = Field(default=None, description="list of grassland values, 7 values")
    treecover:         Optional[List[float]] = Field(default=None, description="list of treecover values, 7 values")
    water:             Optional[List[float]] = Field(default=None, description="list of water values, 7 values")
    ndvi:              Optional[List[float]] = Field(default=None, description="list of ndvi values, 7 values")
    aod:               Optional[List[float]] = Field(default=None, description="list of daily aod values, 7 values")

class VienThamRequest(BaseModel):
    n_future: int = Field(default=None, description="n_future")
    data: VienThamInputData = Field(default=None, description="VienTham input data")

class VienThamResponse(BaseModel):
    data: Optional[List[float]] = Field(default=None, description="List of daily predicted values")

class CMAQInputData(BaseModel):
    pm25:    Optional[List[float]] = Field(default=None, description="list of hourly pm25 values, 168 values")
    pm10:    Optional[List[float]] = Field(default=None, description="list of hourly pm10 values, 168 values")
    o3:      Optional[List[float]] = Field(default=None, description="list of hourly o3 values, 168 values")
    so2:     Optional[List[float]] = Field(default=None, description="list of hourly so2 values, 168 values")
    no2:     Optional[List[float]] = Field(default=None, description="list of hourly no2 values, 168 values")
    no:      Optional[List[float]] = Field(default=None, description="list of hourly no values, 168 values")

class CMAQRequest(BaseModel):
    n_future: int = Field(default=None, description="n_future")
    data: CMAQInputData = Field(default=None, description="VienTham input data")

class CMAQResponse(BaseModel):
    data: Optional[List[float]] = Field(default=None, description="List of hourly predicted values")

class QuanTracInputData(BaseModel):
    date:        Optional[List[str]] = Field(default=None, description="list of date values, 73 values")
    no2:         Optional[List[float]] = Field(default=None, description="list of no2 values, 73 values")
    pm25:        Optional[List[float]] = Field(default=None, description="list of pm25 values, 73 values")
    o3:          Optional[List[float]] = Field(default=None, description="list of o3 values, 73 values")
    co:          Optional[List[float]] = Field(default=None, description="list of co values, 73 values")
    temperature: Optional[List[float]] = Field(default=None, description="list of temperature values, 73 values")
    humid:       Optional[List[float]] = Field(default=None, description="list of humid values, 73 values")
    station_id:  Optional[List[int]] = Field(default=None, description="list of station values, 73 values")

class QuanTracRequest(BaseModel):
    data: QuanTracInputData = Field(default=None, description="QuanTrac input data")

class QuanTracResponse(BaseModel):
    data: Optional[List[float]] = Field(default=None, description="List of 72 predicted values")

class QuanTracSO2InputData(BaseModel):
    date:        Optional[List[str]] = Field(default=None, description="list of date values, 73 values")
    no2:         Optional[List[float]] = Field(default=None, description="list of no2 values, 73 values")
    pm25:        Optional[List[float]] = Field(default=None, description="list of pm25 values, 73 values")
    o3:          Optional[List[float]] = Field(default=None, description="list of o3 values, 73 values")
    co:          Optional[List[float]] = Field(default=None, description="list of co values, 73 values")
    so2:         Optional[List[float]] = Field(default=None, description="list of so2 values, 73 values")
    station_id:  Optional[List[int]] = Field(default=None, description="list of station values, 73 values")

class QuanTracSO2Request(BaseModel):
    data: QuanTracSO2InputData = Field(default=None, description="QuanTrac SO2 input data")

