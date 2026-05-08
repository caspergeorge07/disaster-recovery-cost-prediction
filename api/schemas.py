from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictRequest(BaseModel):
    """
    Input schema for disaster recovery cost prediction.
    """

    declaration_year: int = Field(..., ge=1950, le=2100)
    declaration_month: int = Field(..., ge=1, le=12)
    incident_duration_days: float = Field(..., ge=0)
    state_5yr_disaster_count: float = Field(..., ge=0)

    high_cost_incident: bool
    fyDeclared: int = Field(..., ge=1950, le=2100)
    tribalRequest: bool

    fipsStateCode: int = Field(..., ge=0)
    fipsCountyCode: int = Field(..., ge=0)
    placeCode: int = Field(..., ge=0)
    region: int = Field(..., ge=0)

    state: str = Field(..., min_length=2, max_length=2)
    incidentType: str = Field(..., min_length=1)
    declarationType: str = Field(..., min_length=1)
    season: Optional[str] = None
    census_region: Optional[str] = None

    @field_validator("state")
    @classmethod
    def uppercase_state(cls, value: str) -> str:
        return value.upper()

    @field_validator("season")
    @classmethod
    def validate_season(cls, value: Optional[str]) -> Optional[str]:
        allowed = {"Winter", "Spring", "Summer", "Autumn", "Unknown"}
        if value is not None and value not in allowed:
            raise ValueError(f"season must be one of {allowed}")
        return value

    @field_validator("census_region")
    @classmethod
    def validate_region(cls, value: Optional[str]) -> Optional[str]:
        allowed = {"Northeast", "Midwest", "South", "West", "Territory", "Unknown"}
        if value is not None and value not in allowed:
            raise ValueError(f"census_region must be one of {allowed}")
        return value


class PredictResponse(BaseModel):
    """
    Output schema for prediction response.
    """

    predicted_log_cost: float
    predicted_cost_usd: float
    model_name: str
    model_version: str
    target: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    training_date: str
    target_column: str
    metrics: dict
    numeric_features: list[str]
    categorical_features: list[str]