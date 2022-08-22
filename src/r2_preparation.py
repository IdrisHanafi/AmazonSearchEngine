"""
This module creates the R2 Ranking Model and it's dependencies.
This algorithm incoporates quality and price scores 
instead of value and top features.

Input_file:
- "datasets/R1_Rank_mtx"                        # initial R1 rank 
- "index/R1_index_asin.pickle"                  # correct index for sparse matrix
- "datasets/review_price_scores.csv"            # extracted from reviews NLP technique
- "datasets/review_quality_scores.csv"          # extracted from reviews" 
- "datasets/review_sentiment_scores.csv"        # sentiment extracted from reviews

Output_files: 
- "datasets/R2_Rank_mtx"                        # asin pre-ranked products for ["top_features","top_value","top_sellers","top_ratings"]
"""
import os
import pandas as pd
import pickle
import numpy as np
import joblib
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from scipy import sparse
from scipy.sparse import csr_matrix
from argparse import ArgumentParser, Namespace

def load_data(file_name_and_path):
    products = pd.read_json(file_name_and_path, lines=True)
    
    return products

def create_dir(file):
    """
    Write the file
    """
    filepath = "/".join(file.split("/")[:-1])
    if filepath:
        os.makedirs(filepath, exist_ok=True)


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + ".npz")

    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=loader["shape"]
    )

def df_to_sparse(df):
    # Conversion via COO matrix
    coo = sparse.coo_matrix(df)
    csr = coo.tocsr()

    return csr

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(
        filename, 
        data=array.data, 
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape
    )

def main_process(
    df_price,
    df_quality,
    df_sentiment,
    arr,
    index_asin,
    output_r2_dir,
):
    cols = ["top_features", "top_value", "top_sellers", "top_ratings"] 
    df = pd.DataFrame(arr, columns=cols)
    
    df["asin"] = index_asin.values()

    del df["top_features"]
    del df["top_value"]

    R2_asin = df.merge(df_price, on="asin", how="left")
    R2_asin = R2_asin.merge(df_quality, on="asin", how="left")
    R2_asin = R2_asin.merge(df_sentiment, on="asin", how="left")
    R2_asin.set_index("asin", inplace=True)
    
    # Final Data Frame
    
    R2_asin.columns = ["top_sellers", "top_ratings", "top_value", "top_quality", "text_scores"]
    R2_asin = R2_asin[["top_quality", "top_value", "top_sellers", "top_ratings", "text_scores"]]
    
    R2_asin.fillna(0, inplace=True)
    
    # upload to a spare matrix
    array = df_to_sparse(R2_asin[["top_quality", "top_value", "top_sellers", "top_ratings", "text_scores"]])
    filename = f"{output_r2_dir}/R2_Rank_mtx"
    create_dir(filename)
    save_sparse_csr(filename, array)
    
    return R2_asin

def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_r1_dataset_dir",
        help="The input R1 dataset directory",
        type=str
    )
    parser.add_argument(
        "--input_r1_index",
        help="The input index for R1",
        type=str
    )
    parser.add_argument(
        "--input_review_sentiment_dir",
        help="The directory where the review sentiment files are located",
        type=str
    )
    parser.add_argument(
        "--output_r2_dir",
        help="The output model directory for r2",
        type=str
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = parse_args()

    input_r1_dataset_dir = parsed_args.input_r1_dataset_dir
    input_r1_index = parsed_args.input_r1_index
    input_review_sentiment_dir = parsed_args.input_review_sentiment_dir
    output_r2_dir = parsed_args.output_r2_dir
    
    df_price = pd.read_csv(f"{input_review_sentiment_dir}/review_price_scores.csv")
    df_quality = pd.read_csv(f"{input_review_sentiment_dir}/review_quality_scores.csv")
    df_sentiment = pd.read_csv(f"{input_review_sentiment_dir}/review_sentiment_scores.csv").iloc[:,:2]
    arr = load_sparse_csr(f"{input_r1_dataset_dir}/R1_Rank_mtx").toarray()
    
    with open(input_r1_index, "rb") as filehandle:
        index_asin = pickle.load(filehandle)

    res_df = main_process(
        df_price,
        df_quality,
        df_sentiment,
        arr,
        index_asin,
        output_r2_dir,
    )

    print(res_df.head())
    print("Completed")
