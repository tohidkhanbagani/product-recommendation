import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.input_classes import RecommendationInput, BulkProductRecommendationInputs

from src.pipeline.inference_pipeline import ModelInference


app = FastAPI()




def load_infrence():
    global inference
    print("Initializing Infrence...")
    infrence = ModelInference()
    print("Initialization Finished...")
    return infrence


infrence = load_infrence()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---
@app.post('/prediction/product-recommender')
async def product_recommender(user_input:RecommendationInput):
    global inference
    recommended_products, product_prices, confidence_scores = infrence.product_recommendation(
        id=user_input.id, 
        product=user_input.products, 
        counts=user_input.counts
        )
    result_content = {'recommended_products':recommended_products,
                      'product_prices':product_prices,
                      'confidence_scores':confidence_scores
                      }
    return JSONResponse(status_code=200, content=result_content)


@app.post('/predict/product-recommendation-bulk')
async def product_recommender_bulk(params:BulkProductRecommendationInputs):
    global inference
    id, recommendations, prices = infrence.product_recommendation(multiple=True, N=params.N, fake=True,entries=params.entries)
    result_content = {'id':id,'recommendations':recommendations,'prices':prices}
    return JSONResponse(status_code=200, content=result_content)


@app.get("/products")
def get_all_products():
    """
    Returns a list of all products with their details.
    """
    global infrence

    all_products_data = infrence.model_inputs.product_list

    return JSONResponse(content=all_products_data)

# The /products/search endpoint can be removed if the frontend filters the /products list,
# or it can be updated to search and return the rich data format.