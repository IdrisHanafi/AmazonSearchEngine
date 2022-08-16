import pandas as pd
import random
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
import warnings
# warnings.filterwarnings('ignore')
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim import utils
from bs4 import BeautifulSoup as BSHTML
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS #, remove_stopword_tokens

dir_path = os.path.dirname(os.path.realpath(__file__))


''' Lambda function: takes a query and returns the top 5 category keys of recommendations

INPUT files: 
# INPUT FILES:
"datasets/M2_tfidf_mtx"     
'models/M2_tfidf.joblib'
'index/M2_category_labels.data'
'datasets/M2_rank.data'
'index/M2_category_index.pickle'


NEW, additional input files: 
'datasets/R1_tfidf_mtx' 
'models/R1_tfidf.joblib'
'index/M2R1_big_dico.pickle'
'index/M2_index_category.pickle'
'index/M2R1_asin_category.pickle'
'index/R1_index_asin.pickle'

OUTPUT: top 5 tuples of matching categories
1- Creates a class tfidf and download dependencies in the first instance, tries to save in cache
2- Takes each query and vectorize
3- Calculate simirality tuples M2
4- Ranks recommendation based on similarity and np.log(count of asin in each category) 
5- If M2's similarity score < threashold, will activate R2 algorithm based on similarity score of asin products
and returns its associated category indices 
6- For testing purposes, returns the ASIN labels, this feature can be removed in deployment

'''

# match=set(self.M2.get_feature_names_out()) & set(liste)
# creating class to import necessary matrices and models for retrival
class TfIdfThreshold:
    def __init__(self):
        # load cleaned group_by data sets and models from disk
        self.mtx=self.load_sparse_csr(f"{dir_path}/../datasets/m2_data/M2_tfidf_mtx")        
        self.M2 = joblib.load(f"{dir_path}/../models/M2_tfidf.joblib")
        self.stemmer = PorterStemmer() 
        
        with open(f"{dir_path}/../index/M2R1_big_dico.pickle", "rb") as filehandle:
            self.big_index = pickle.load(filehandle)
        
        ###################### PLEASE OPTIMIZE ###########################
        with open(f"{dir_path}/../index/R1_asin_labels.data", "rb") as filehandle:
             self.asin_labels = pickle.load(filehandle)
        ##################################################################   
        with open(f'{dir_path}/../datasets/m2_data/M2_rank.data', 'rb') as filehandle:
            self.rank = np.array(pickle.load(filehandle))
        ##################################################################
        
        ##################### NEW UPLOAD #################################
        self.tfidf_mtx = self.load_sparse_csr(f"{dir_path}/../datasets/r1_data/R1_tfidf_mtx")
        self.model = joblib.load(f'{dir_path}/../models/R1_tfidf.joblib')
        #####################################################################
        
        with open(f'{dir_path}/../index/M2_category_index.pickle', 'rb') as filehandle:
            self.category_index = pickle.load(filehandle)
            
        with open(f'{dir_path}/../index/M2_index_category.pickle', 'rb') as filehandle:
            self.index_category = pickle.load(filehandle)
            
        with open(f'{dir_path}/../datasets/m2_plus_data/M2R1_asin_category.pickle', 'rb') as filehandle:
            self.asin_category = pickle.load(filehandle)
            
        with open(f'{dir_path}/../index/R1_index_asin.pickle', 'rb') as filehandle:
            self.index_asin = pickle.load(filehandle)
            
    def load_sparse_csr(self, filename):
        # here we need to add .npz extension manually
        loader = np.load(filename + '.npz')
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape']) 
    
    #### This function is for testing purpose ######
    def asin_of_categories(self,cat_idx):
        
        info = self.big_index[cat_idx]
        cat = info[0]
        start = info[1]
        finish = info[2]
            
        self.simi = cosine_similarity(self.tfidf_mtx,self.quer_R1)
        simi_mtx = self.simi[start:finish,]
            
        arr2 = np.argsort(-simi_mtx,axis=0)[:5] + start
        dim = arr2.shape[0]
        idx_asin = list(arr2.reshape(dim,)) 
        
        return idx_asin
    ##################################################
    
    #################################################
    def R1_query_to_category(self):
        # runs R1 model and similarity
               
        similarity = cosine_similarity(self.tfidf_mtx,self.quer_R1)
        
        # returns labels 
        idx_asin = list(np.argsort(-similarity,axis=0)[:5].reshape(5,))
        idx_category = [ self.asin_category[x] for x in idx_asin]
        result = [self.index_category[x] for x in idx_category]
        maxi = np.round(np.max(similarity),2)
        
        return result, maxi, idx_category
    #################################################
    
    ## retrieve match tfidf (LAMBDA FUNCTION) at a category level
    def M2_query_to_category_function(self, q, threshold=0.45):
        
        stop_words= set(stopwords.words('english'))
        liste=[self.stemmer.stem(w) for w in word_tokenize(q) if not w.lower() in stop_words]
        query_string=' '.join(element for element in liste)       
        quer = self.M2.transform([query_string])
        similarity = cosine_similarity(self.mtx,quer) 
        
        
        Beta_factor = 0 # neutralizes the factor with Beta = 0
        simi_mtx = similarity # + self.rank * Beta_factor # rank is a coeff that you can tweak np.log(gb["count"] + 1) * 0.35
        idx = list(np.argsort(-simi_mtx,axis=0)[:5].reshape(5,))
        
        # returns the label of the category
        result = [self.index_category[x] for x in idx]
        maxi = np.round(np.max(similarity),2)
        self.quer_R1 = self.model.transform([query_string]) 

        # IF similarity scores is not above the theshold, then will look for closes ASINs and return its
        # associated category label
        if maxi < threshold:
            result, maxi, idx = self.R1_query_to_category()
        
        # For testing purpose, score calculation, retrieves the asin list associated with each category
        liste_asin = []
        retour = []
        
        for category_idx in idx:
            ##### M2 or R1 returns indices ###############
            retour = self.asin_of_categories(category_idx)
            liste_asin = liste_asin + retour

        result_obj = []
        for curr_idx in idx:
            item_obj = {
                "label": self.index_category[curr_idx],
                "index": curr_idx,
            }
            result_obj.append(item_obj)

        return result_obj, maxi, query_string, liste_asin
    
k = ['black backpack for laptop', 'European AC DC adapter', 'I need NOW! an apple ipad headphone earphone',
 'I want a computer screen', 'headphone', 'I want a Kingston sd card', 'dell inspiron laptop', 'canon camera',
 'screen for laptop', 'garmin sports watch', 'a tablet Apple 16', 'a television', 'microsoft office', 'Apple laptop case',
 '64GB kingston SD card', 'books', 'The holy bible', 'a garmin sports watch forerunner', 'a waterproof watch garmin']

if __name__=="__main__":
    ob=TfIdfThreshold()
    dico={}
    for q in k:
        print(q)
        
        threshold = 0.45
        result, maxi, ret, liste_asin = ob.M2_query_to_category_function(q, threshold)
        
        if maxi> threshold:
            print("Query: {}, simi_max {}, Quer: {}".format(q,maxi, ret))
            for i,category in enumerate(result):
                if i==0:
                    dico[q]=ob.category_index[category]
                print(i+1," ",category)
            print(" ")
            ##### FOR TESTING PURPOSE ONLY, will return top ASIN #############
            # print("Now top asin")
            # for j, asin in enumerate(liste_asin[:15]):
            #     print(j+1," ",ob.asin_labels[asin][1])
            # print(" ")
            #################################################################
        else:
            print("There is no sufficient match for '{}', please enter new terms".format(q))
            print(" ")
