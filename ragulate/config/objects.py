from pydantic import BaseModel

from typing import Any, Dict

class Step(BaseModel):
    name: str
    script: str
    method: str

class Recipe(BaseModel):
    name: str
    ingest: Step | None
    query: Step
    cleanup: Step | None
    ingredients: Dict[str, Any]

class Config(BaseModel):
    recipes: Dict[str, Recipe] = {}
