import time
from fastapi import FastAPI
from src.m2_query import (
    tfidf
)
from src.r1_query import (
    r1_dumb
)

tfidf_object = tfidf()
r1_obj = r1_dumb(extra_feature=True) 

api = FastAPI()

@api.get("/")
async def root():
    return {"message": "Hello World"}

@api.get("/get_subcategory/{user_query}")
async def query(user_query: str):
    try:
        start_time = time.time()
        result_obj, category_keys, maxi = tfidf_object.M2_query_to_category_function(user_query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "message": f"Found {len(category_keys)} result(s) for {user_query}",
            "user_query": user_query,
            "subcategory_found": result_obj,
            "time_elapsed": elapsed_time,
            "similarity_score": maxi
        }
    except Exception as found_error:
        return {
            "message": "No result found",
            "error": found_error
        }

FILTER_OPTIONS = ["top_features","top_value","top_sellers","top_ratings"]

@api.get("/get_products/")
async def get_products(
    user_query: str="",
    category_id: int=0,
    filter_type: str=FILTER_OPTIONS[0]
):
    try:
        start_time = time.time()
        result = r1_obj.lambda_R1(user_query, category_id, filter_type)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "message": f"Found {len(result)} result(s) for {user_query}",
            "user_query": user_query,
            "result": result,
            "time_elapsed": elapsed_time,
        }
    except Exception as found_error:
        print(found_error)
        return {
            "message": "No result found",
            "error": found_error
        }
