"""
This module creates the M2 Model:
Input_file:
- "datasets/cleaned_data.json"              # from clean_metadata
    
Output_files: 
1. Datasets
- "datasets/R1_data_indexed.json"           # Fully indexed with cat / asin and all strings and tokenized features
- "datasets/M2_categories.json"             # Contains only 1018 categories
- "datasets/M2_tfidf_mtx"                   # tfidf csr matrix trained on 4-features combined
- "datasets/M2_rank.data"                   # 1018 long vector part of adjustable Coca-Cola formula to favor categories with more asin counts.
                                            # Coca-Cola formula: [np.log(gb_indexed_df["count"] + 1) * 0.35] 

2. Models
- "models/M2_tfidf.joblib"                  # tfidf trained on 4-features

3. Indices and labels
- "index/M2_category_labels.data"           # list of category labels ordered according to category index
- "index/R1_asin_labels.data"               # list of asin + title labels ordered according to category index
- "index/R1_index_asin.pickle"
- "index/R1_asin_index.pickle"
- "index/M2_index_category.pickle"
- "index/M2_category_index.pickle"  
- "index/M2R1_big_dico.pickle"              # Big index to facilitate matrix filtering and asin retrieval
- "index/M2R1_asin_category.pickle"         # NEW
"""
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import joblib
import json
import pickle
import os
import warnings
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

def character_cut_off(data, cut_off):  
    data["characters"] = data["category_string"].apply(
        lambda x: len(x)
    )
    data_short = data[data["characters"] < cut_off]
    print("before: {}".format(len(data)), "after: {}:".format(len(data_short)))

    return data_short

def indexation(full_df, output_index_dir):
    # groubing by categories and asin to create an index
    cols = ["category_list", "asin"]
    gb_df = full_df.groupby(cols).agg(
        {
            "category_string": "first",
            "title": "first",
            "4_features_combined": "first",
            "5_features_combined": "first",
            "4_features_tokenized": "first",
            "characters": "first"
        }
    ).reset_index()    
     
    asin_category = dict(zip(gb_df.asin, gb_df.category_list))
    
    # creating category indices and their reverse. add a column with category idx
    list_of_categories = list(gb_df.category_list.unique())
    index_category = {idx: cat for idx, cat in enumerate(list_of_categories)}
    category_index = {cat: idx for idx, cat in index_category.items()}    
    gb_df["cat_idx"] = gb_df["category_list"].apply(lambda cat: category_index[cat])
    

    # creating asin and their reverse.
    list_of_asin = list(gb_df.asin.unique())
    list_of_title = list(gb_df.title.unique())
    index_asin = {i: asi for i, asi in enumerate(list_of_asin)}
    asin_index = {asi: i for i, asi in index_asin.items()}
             
    # creates M2 indexed by category
    gb_indexed = gb_df.groupby(by="category_list").agg(
        {
            "cat_idx": "first",
            "category_string": "first",
            "title": "size"
        }
    ).reset_index()
    gb_indexed.rename(columns={"title": "count"}, inplace=True)
    gb_indexed.sort_values(by="cat_idx", ascending=True, inplace=True)   
    print("full_data: {}, gb: {} , gb_indexed: {}".format(len(full_df), len(gb_df), len(gb_indexed))) 
    
    # creates big R2 index
    gb2 = gb_indexed.copy()
    gb2["finish"] = gb2["count"].cumsum()
    gb2["start"] = gb2["finish"] - gb2["count"]
    big_list = list(zip(gb2.category_list, gb2.start, gb2.finish))
    large_dictionary = {i: liste for i, liste in enumerate(big_list)}
    asin_category = {asin: category_index[category] for asin, category in asin_category.items()}
    
    # Store indices
    with open(f"{output_index_dir}/R1_index_asin.pickle", "wb") as handle:
        pickle.dump(index_asin, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{output_index_dir}/R1_asin_index.pickle", "wb") as handle:
        pickle.dump(asin_index, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    with open(f"{output_index_dir}/M2_index_category.pickle", "wb") as handle:
        pickle.dump(index_category, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{output_index_dir}/M2_category_index.pickle", "wb") as handle:
        pickle.dump(category_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("index/M2R1_big_dico.pickle", "wb") as handle:
        pickle.dump(large_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f"{output_index_dir}/M2R1_asin_category.pickle", "wb") as handle:
        pickle.dump(asin_category, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return gb_df, gb_indexed, gb2

def transform(x):
    """
    takes a TOKENIZED field and returns the stem in a string version
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    liste = [stemmer.stem(w) for w in word_tokenize(x) if not w.lower() in stop_words]
    string = " ".join(element for element in liste)

    return string


def tf_idf(sub_df):
    """
    creates the stemmed field, model and tf_idf matrix
    """
    sub_df["stemmed"] = sub_df["category_string"].apply(lambda x: transform(x))
    model_tf = TfidfVectorizer(
        ngram_range = (1, 2)
    ).fit(sub_df["stemmed"])
    sparse_matrix = model_tf.transform(sub_df["stemmed"]) 

    return sparse_matrix, model_tf


def main_process(
    data,
    output_dataset_dir,
    output_index_dir,
    output_model_dir,
):
    """
    INPUT file: cleaned_data, OUTPUT: M2 model and dependencies
    1- reads clean json file
    2- returns a data frame with 99% of categories < 131 characters (cut_off)
    3- creates indices and indexed datasets
    4- creates tfidf matrix and M2 model on stemmed category strings 
    5- saves everything in data, index and models directories
    """
    
    # 2- convert to tuple
    data["category_list"] = data["category_list"].apply(
        lambda x: tuple(x)
    )
    
    # 3- remove fields with description above cut_off
    cut = 131
    data_short = character_cut_off(data, cut)
    
    # 4- indexation
    full_indexed_df, gb_indexed_df, gb2 = indexation(data_short, output_index_dir)
    write_to_file(full_indexed_df, f"{output_dataset_dir}/R1_data_indexed.json")
    write_to_file(gb_indexed_df, f"{output_dataset_dir}/M2_categories.json")
    
    # 5- creating and saving model and matrix
    sparse_mtx, model_tf = tf_idf(gb_indexed_df)
    array = sparse_mtx
    # note that .npz extension is added automatically
    np.savez(
        f"{output_dataset_dir}/M2_tfidf_mtx",
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape
    ) 

    filename = f"{output_model_dir}/M2_tfidf.joblib"
    joblib.dump(model_tf, filename)

    # 6- algorithm ranking formula based on count and returning labels
    # FORMULA THAT WEIGHTS THE COUNT:
    gb_indexed_df["rank"] = np.log(gb_indexed_df["count"] + 1) * 0.35
    
    ## creating lists for fast retrieval
    with open(f"{output_dataset_dir}/M2_rank.data", "wb") as filehandle:
        pickle.dump(gb_indexed_df[["rank"]], filehandle)

    with open(f"{output_index_dir}/M2_category_labels.data", "wb") as filehandle:
        pickle.dump(gb_indexed_df["category_list"], filehandle)
    
    # dual labels
    list_of_asin = list(full_indexed_df.asin)
    list_of_title = list(full_indexed_df.title)
    dual_labels = list(zip(list_of_asin, list_of_title))
    with open(f"{output_index_dir}/R1_asin_labels.data", "wb") as filehandle:
        pickle.dump(dual_labels, filehandle)    
     
    return full_indexed_df, gb_indexed_df, gb2, sparse_mtx

def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="The cleaned metadata file",
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
    output_dataset_dir = parsed_args.output_dataset_dir
    output_index_dir = parsed_args.output_index_dir
    output_model_dir = parsed_args.output_model_dir
    
    # 1- step one, retrive the data
    df = load_data(input_file)
    full_indexed_df, gb_indexed_df, gb2, df_mtx = main_process(
        data,
        output_dataset_dir,
        output_index_dir,
        output_model_dir,
    )

    print(
        full_indexed_df.shape,
        gb_indexed_df.shape,
        gb2.shape,
        df_mtx.shape
    )
