"""
This module creates the M1 Model and the dependencies.

Input_file:
- "datasets/cleaned_data.json"
- "datasets/m1_data/category_string_index.pickle"
    
Output_files: 
- f"{output_model_dir}/lsi.joblib"
- f"{output_dataset_dir}/sk_vectorizer.joblib"
- f"{output_dataset_dir}/lsi_text_transformed.joblib"
- f"{output_index_dir}/sk_lsi_index.joblib"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from argparse import ArgumentParser, Namespace

def load_data(file_name_and_path):
    products = pd.read_json(file_name_and_path, lines=True)
    
    return products


def write_to_file(df, file):
    """
    Write the file
    """
    filepath = "/".join(file.split("/")[:-1])
    if filepath:
        os.makedirs(filepath, exist_ok=True)
    df.to_json(
        file,
        orient="records",
        lines=True
    )


def create_M1(df, features="4_features_combined", components=10, ngrams=2):
    # create vectorizer title and category plus description
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        ngram_range=(1, ngrams),
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df[features])
    #create LSI model
    lsi = TruncatedSVD(n_components=components)  
    lsi_text_transformed = lsi.fit_transform(X)
    
    return vectorizer, lsi, lsi_text_transformed


def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="The cleaned metadata file",
        type=str
    )
    parser.add_argument(
        "--input_file_category_string_index",
        help="The category string to index file",
        type=str
    )
    parser.add_argument(
        "--output_dataset_dir",
        help="The output dataset dir",
        type=str
    )
    parser.add_argument(
        "--output_index_dir",
        help="The output index directory",
        type=str
    )
    parser.add_argument(
        "--output_model_dir",
        help="The output model directory",
        type=str
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = parse_args()

    input_file = parsed_args.input_file
    input_file_category_string_index = parsed_args.input_file_category_string_index

    output_dataset_dir = parsed_args.output_dataset_dir
    output_index_dir = parsed_args.output_index_dir
    output_model_dir = parsed_args.output_model_dir
    
    df = load_data(input_file)
    with open(input_file_category_string_index, "rb") as filehandle:
        category_string_index = pickle.load(filehandle)

    def get_index(x):
        try:
            return(category_string_index[x])
        except:
            return(np.nan)

    df["subcategory_index"] = df["category_string"].apply(get_index)
    df.dropna(inplace=True)

    # create lsi_index as smaller vehicle to look up subcategory information
    lsi_index = df[["subcategory_index", "category_group_count"]].to_numpy()
    vectorizer, lsi, lsi_text_transformed = create_M1(
        df,
        features="4_features_combined",
        components=10,
        ngrams=2
    )

    joblib.dump(lsi, f"{output_model_dir}/lsi.joblib")
    joblib.dump(vectorizer, f"{output_dataset_dir}/sk_vectorizer.joblib")
    joblib.dump(lsi_text_transformed, f"{output_dataset_dir}/lsi_text_transformed.joblib")
    joblib.dump(lsi_index, f"{output_index_dir}/sk_lsi_index.joblib")

    print("Completed")
