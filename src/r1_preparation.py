"""
This module creates the R1 Ranking Model and it's dependencies.

Input_file:
- "datasets/R1_data_indexed.json"   # 700k+ row data set indexed and ordered by category/asin. Contains all features
- "datasets/ratings.csv"            # ratings is merged on asin and to populate "top_sellers" and "top_ratings" 

Output_files: 
Datasets:
- "datasets/R1_Rank_mtx"            # asin pre-ranked products for ["top_features","top_value","top_sellers","top_ratings"]
- "datasets/R1_tfidf_mtx"           # tfidf matrix trained on "title" field

Models:
- "models/R1_tfidf.joblib"          # tfidf model trained on "title" field
"""
import os
import pandas as pd
import numpy as np
import joblib
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from scipy import sparse
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
    df.to_json(file)


def transform(x, test):
    stemmer = PorterStemmer()
    stop_words= set(stopwords.words("english"))
    if test: # for untokenized text
        liste = [stemmer.stem(w) for w in word_tokenize(x) if not w.lower() in stop_words]
    else: # for tokenized text
        liste = [stemmer.stem(w) for w in x]
    string=" ".join(element for element in liste)
    return string


def tf_idf(df1, field, test):
    # field to tokenize
    df1["stemmed"] = df1[field].apply(lambda x: transform(x, test))
    model_tf = TfidfVectorizer().fit(df1["stemmed"])
    sparse_matrix = model_tf.transform(df1["stemmed"]) 
    
    return sparse_matrix,model_tf


def vectorizing_value(df_full, combined): #takes the data frame and returns full and grouby df
    # creating the good value tfidf model
    stemmer = PorterStemmer()
    words = "great value hot sale best price free good value money hot reduction giveaway firesale super great bargain cheap quality deal excellent fantastic discount special offer promotion"
    # stem strings
    string = transform(words, True)
    
    model_count = TfidfVectorizer(ngram_range=(1, 2)).fit([string])
    cdf = TfidfVectorizer().fit_transform([string])

    # transforming combined string to "value" tfidf and summing the score
    sparse_matrix = model_count.transform(df_full[combined].astype(str))
    df_full["best_value"] = np.sum(sparse_matrix.todense(), axis=1)
    df_full[["asin","best_value"]].sort_values(by="best_value", ascending=False).head(5)
    df_full["len_features"] = df_full["5_features_combined"].apply(lambda x: len(x))

    return df_full


# retrieving ratings
def gb_ratings(ratings_df):
    gb_rat = ratings_df.groupby(["asin"]).agg(
        {
            "unknown": "size",
            "stars": "mean"
        }
    )
    gb_rat.rename(
        columns={"unknown": "best_sellers", "stars": "top_stars"},
        inplace=True
    )
    gb_rat.reset_index(inplace=True)
    return gb_rat


# normalizing from 0 to 5 
def transform_five(x, mini, maxi):
    y = (x - mini) / (maxi - mini) * 5

    return np.round(y, 2)


def final_transformation(new_df, combined): 
    new_df.rename(
        columns={
            "len_features": "top_features",
            "best_value": "top_value",
            "best_sellers": "top_sellers",
            "top_stars": "top_ratings"
        }, 
        inplace = True
    )
    new_col_names = ["top_features", "top_value", "top_sellers", "top_ratings"]
    
    for col in new_col_names:
        mini = np.min(new_df[col])
        maxi = np.max(new_df[col])
        new_df[col] = new_df[col].apply(
            lambda x: transform_five(x, mini, maxi)
        )
        
    for col in ["top_features", "top_value", "top_ratings"]:
        new_df[col] = (new_df[col] + (new_df["top_sellers"]) * .2) / 1.2
        mini = np.min(new_df[col])
        maxi = np.max(new_df[col])
        new_df[col] = new_df[col].apply(
            lambda x: transform_five(x, mini, maxi)
        )
     
    return new_df


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
    indexed_data,
    ratings_df,
    output_dataset_dir,
    output_model_dir,
):
    """
    R1_model:
    INPUT FILES: takes "indexed_data" and "ratings.csv", 
    return a file with a group by object (category_tuple / asin)
    with a score for each dimentsion.

    OUTPUT:
    1- Reads R1_data_indexed.json"
    2- Creates a best_value field with tf_idf vectorizing
    3- Import ratings .csv dataset and merge
    4- creates 4 columns of scores from 0 to 5: ["top_features","top_value","top_sellers","top_ratings"]
    5- Saves these pre-ranked scores in a csr sparse matrix R1_Rank_mtx
    6- Creates tfidf model and matrix on title for R1_not_so_dumb model
    """
    # selecting data to vectorize for best value
    combined = "4_features_tokenized"
    df_full = vectorizing_value(indexed_data, combined)
    
    # merging data_set with ratings_df
    full_ratings = gb_ratings(ratings_df)    
    R1_asin = df_full.merge(full_ratings, on="asin", how="left")
    
    # aditional score transformation to make sure everything is from 0 to 5
    R1_asin = final_transformation(R1_asin, combined)
    
    # 1- converting the ranking df to a sparse matrix and save it
    array = df_to_sparse(
        R1_asin[["top_value", "top_features", "top_sellers", "top_ratings"]]
    )
    filename = f"{output_dataset_dir}/R1_Rank_mtx"
    save_sparse_csr(filename, array)
    
    # 2- fitting tf_idf on title, field can be selected. saving matrix and model
    field = "title"
    string = True # select false if field is already tokenized
    filename = f"{output_dataset_dir}/R1_tfidf_mtx"
    matrix, model_tf = tf_idf(R1_asin, field, string)
    save_sparse_csr(filename, matrix)

    filename = f"{output_model_dir}/R1_tfidf.joblib"
    joblib.dump(model_tf, filename) 


def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_data_index",
        help="The input data index file",
        type=str
    )
    parser.add_argument(
        "--input_ratings",
        help="The input ratings.csv file",
        type=str
    )
    parser.add_argument(
        "--output_dataset_dir",
        help="The output dataset dir",
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

    input_data_index = parsed_args.input_data_index
    input_ratings = parsed_args.input_ratings
    output_dataset_dir = parsed_args.output_dataset_dir
    output_model_dir = parsed_args.output_model_dir
    
    # 1- step one, retrive the data
    # importing datasets
    indexed_data = pd.read_json("datasets/R1_data_indexed.json")  
    cols = ["asin", "review", "stars", "unknown"]
    ratings_df = pd.read_csv("datasets/ratings.csv", header=None, names=cols)

    main_process(
        indexed_data,
        ratings_df,
        output_dataset_dir,
        output_model_dir,
    )

    print("Completed")
