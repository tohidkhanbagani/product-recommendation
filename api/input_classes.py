import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing import List, Annotated, Union
import json

class RecommendationInput(BaseModel):
    id: Annotated[List[str], Field(..., description="Enter User IDs")]
    products: Annotated[List[str], Field(..., description="Enter Purchased Products")]
    counts: Annotated[List[int], Field(..., description="Purchase Units of each product")]


class BulkProductRecommendationInputs(BaseModel):
    N:int=10
    entries:int=50