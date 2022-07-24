from fastapi import FastAPI
from src.tfidf_query import (
    tfidf
)


ob = tfidf()

api = FastAPI()

@api.get("/")
async def root():
    return {"message": "Hello World"}

@api.get("/get_subcategory/{user_query}")
async def query(user_query: str):
    try:
        category_keys, maxi = ob.M2_query_to_category_function(user_query)
        return {
            "message": f"Found {len(category_keys)} result(s) for {user_query}",
            "user_query": user_query,
            "subcategory_found": category_keys,
            "similarity_score": maxi
        }
    except Exception as found_error:
        return {
            "message": "No result found",
            "error": found_error
        }

