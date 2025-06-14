from pydantic import BaseModel
from typing import Dict, Optional

class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    Department: str
    Category: str
    Sub_Category: str
    Operational_Entity: str
    Status: str
    Operational_Trigger: str
    Location_Type: str
    Location: str

class ErrorResponse(BaseModel):
    detail: str 