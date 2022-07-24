''' 
Lambda function: takes a query and returns the top 5 category keys of recommendations
INPUT files: "M2_tfidf_mtx.csv" , "M2.joblib" ,  "M2_categories.json" + query
OUTPUT: top 5 tuples of matching categoris
1- Creates a class tfidf and download dependencies in the first instance, tries to save in cache
2- Takes each query and vectorize
3- Calculate simirality tuples
4- Ranks recommendation based on similarity and np.log(count of asin in each category)
'''
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
## tf_idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import joblib
import gensim
import gzip
import json
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import pickle
# from keras.preprocessing.text import Tokenizer
import collections
from collections import Counter
import re as regex
from scipy.sparse import csr_matrix
import os
import boto3

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
print("DIRECTORY PATH")
print(dir_path)

stop_words = set(stopwords.words('english'))

# creating class to import necessary matrices and models for retrival
class tfidf:
    def __init__(self):
        # load cleaned group_by data sets and models from disk
        print("!!!!!!!!!!!!!!!!!!")
        print("LOADING MODEL...")
        self.df = pd.read_csv(f"{dir_path}/../datasets/M2_tfidf_mtx.csv")
        gb = pd.read_json(f"{dir_path}/../datasets/M2_categories.json")
        gb["categoryi_tuple"] = gb["category_tuple"].apply(lambda x: tuple(x))
        self.gb = gb
        self.mtx = csr_matrix(self.df.astype(pd.SparseDtype("float64",0)).sparse.to_coo())
        self.M2 = joblib.load(f"{dir_path}/../models/M2.joblib")
        self.stemmer = PorterStemmer()
        
    ## retrieve match tfidf (LAMBDA FUNCTION)
    def M2_query_to_category_function(self, q):
        # stemming the query (this part will be replaced by a utility function)
        liste = [self.stemmer.stem(w) for w in word_tokenize(q) if not w.lower() in stop_words]
        string = ' '.join(element for element in liste)
        print("Query:", liste, string)

        # copying sparse matrix
        df3 = self.gb.reset_index()
        quer = self.M2.transform([string])
        simi_mtx = cosine_similarity(ob.mtx,quer)
        df3.sort_values(by="index",inplace=True)

        match=set(self.M2.get_feature_names_out()) & set(liste)

        if not match:
            ## MOVE TO ALTERNATIVE ALGORITHM 
            ## (EXTEND MATCH TO DESCRIPTION  + BRAND + TITLE)
            return None,None

        df3["simi"]=simi_mtx
        df3.sort_values(by="simi",inplace=True,ascending=False)
        df3["coef"]=df3.apply(lambda x: np.log(x["count"] + 1) * x["simi"], axis=1)

        df3.sort_values(by="coef",inplace=True,ascending=False)

        try:
            df_retour = df3.head(5)
            maxi = np.round(np.max(df_retour.simi.values),2)
            category_keys = list(df_retour.category_tuple)
            return category_keys, maxi

        except:
            df_retour = df3
            maxi = np.round(np.max(df_retour.simi.values),2)
            return list(df_retour.category_tuple), maxi

def process_user_query():
    k = ["AC adapter", "I want a screen", "DESKTOP HP","headphone", "I want an sd card",
      "computer","camera","screen","car GPS","sports watch",
      "a tablet Apple","a television","hewlet packard","microsoft office","laptop case",
       "data science"]

    quer = ["headphone"]
    for q in k:
        print(q)
        category_keys, maxi = ob.M2_query_to_category_function(q)
        
        if category_keys:
            for category in category_keys:
                print(category)
            # print("Query: {}, simi_max {}".format(q,maxi))
            # for i,category in enumerate(category_keys):
            #     print(i+1," ",category)
            # print(" ")
        else:
            print("There is no sufficient match for '{}', please enter new terms".format(q))
            print(" ")

ob = tfidf()

if __name__=="__main__":
    print("Starting queries")
    process_user_query()
