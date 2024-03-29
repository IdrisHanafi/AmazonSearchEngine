import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.m1_query import (
    M1Lsi
)
from src.m2_query import (
    tfidf
)
from src.m2_plus_query import (
    TfIdfThreshold
)
from src.r1_query import (
    R1Baseline
)
from src.r2_query import (
    R2Smart
)

m1_obj = M1Lsi()
tfidf_object = tfidf()
tfidf_threshold_object = TfIdfThreshold()
r1_obj = R1Baseline(extra_feature=True) 
r2_obj = R2Smart(extra_feature=True) 

api = FastAPI()
origins = [
    "http://ec2-18-220-136-214.us-east-2.compute.amazonaws.com/",
    "http://localhost",
    "http://localhost:3000",
    "*",
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
async def root():
    return {"message": "Hello World"}

@api.get("/get_subcategory/m1/{user_query}")
async def m1_query(user_query: str):
    try:
        start_time = time.time()
        result_obj = m1_obj.query_category(user_query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "message": f"Found {len(result_obj)} result(s) for {user_query}",
            "user_query": user_query,
            "subcategory_found": result_obj,
            "time_elapsed": elapsed_time,
        }
    except Exception as found_error:
        return {
            "message": "No result found",
            "error": found_error
        }

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

@api.get("/get_subcategory/v2/{user_query}")
async def query_v2(user_query: str):
    try:
        start_time = time.time()
        result_obj, max_sim_score, _, _ = \
                tfidf_threshold_object.M2_query_to_category_function(user_query)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {
            "message": f"Found {len(result_obj)} result(s) for {user_query}",
            "user_query": user_query,
            "subcategory_found": result_obj,
            "time_elapsed": elapsed_time,
            "similarity_score": max_sim_score
        }
    except Exception as found_error:
        print("ERROR")
        print(found_error)
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
        result = await r1_obj.lambda_R1(user_query, category_id, filter_type)
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

FILTER_OPTIONS_V2 = [
    "top_matches",
    "top_quality",
    "top_value",
    "top_sellers",
    "top_ratings",
    "top_reviews"
]

@api.get("/get_products/v2/")
async def get_products_v2(
    user_query: str="",
    category_id: int=0,
    filter_type: str=FILTER_OPTIONS_V2[0]
):
    try:
        start_time = time.time()
        result = await r2_obj.lambda_R2(user_query, category_id, filter_type)
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
