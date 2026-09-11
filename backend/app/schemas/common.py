from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

Provenance = Literal["official_usnic", "historical_training_dataset", "derived", "interpolated", "predicted"]
T = TypeVar("T")


class ApiModel(BaseModel):
    """Base for responses: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)


class Page(ApiModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorResponse(ApiModel):
    detail: str
    request_id: str | None = None


ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Resource not found"},
    422: {"description": "Invalid parameters"},
    503: {"model": ErrorResponse, "description": "Database or model unavailable"},
}
