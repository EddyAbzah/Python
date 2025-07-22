"""
Write a RESTful server using fastapi that has a list of 10 parameters. Every parameter can take an integer value from 0 to 100.
The server should have the following endpoints:
* set param
* get param
* get all params
* clear all params (set to 0)
"""


from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


app = FastAPI()


devices: dict[str, list[int]] = {}


class ParamValue(BaseModel):
    value: int


class ParamList(BaseModel):
    values: list[int]


@app.post("/devices/{device_id}")
async def create_device(device_id: str):
    if device_id in devices:
        raise HTTPException(status_code=400, detail="Device already exists")
    devices[device_id] = [0] * 10
    return "Success"


@app.delete("/devices/{device_id}")
async def delete_device(device_id: str):
    devices.pop(device_id, None)
    return "Success"


@app.put("/devices/{device_id}/params/{param_id}")
async def set_param(device_id: str, param_id: int, param_value: ParamValue):
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    if 0 <= param_id < 10 and 0 <= param_value.value <= 100:
        devices[device_id][param_id] = param_value.value
        return "Success"
    else:
        raise HTTPException(status_code=400, detail="Invalid parameter ID or value")


@app.get("/devices/{device_id}/params/{param_id}")
async def get_param(device_id: str, param_id: int):
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    if 0 <= param_id < 10:
        return ParamValue(value=devices[device_id][param_id])
    else:
        raise HTTPException(status_code=404, detail="Parameter not found")


@app.get("/devices/{device_id}/params")
async def get_all_params(device_id: str):
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found")
    return ParamList(values=devices[device_id])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
