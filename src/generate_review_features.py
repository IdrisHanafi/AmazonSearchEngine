"""
Generates extra features for the ranking algorithm by
extracting user sentiment from review text.

Input Files:
    - Raw Electronics_5.json

Output Files:
    - "datasets/review_price_scores.csv"
    - "datasets/review_quality_scores.csv"
    - "datasets/review_sentiment_scores.csv"
"""
import copy
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.corpora import Dictionary
from argparse import ArgumentParser, Namespace

WORD_SET_GOOD_QUALITY = set([
    "good_quality", "great_quality", "high_quality", "excellent_quality",
    "quality_product", "nice_quality", "better_quality", "best_quality",
    "highest_quality", "amazing_quality", "awesome_quality", "top_quality",
    "quality_item", "quality_made", "quality_material", "works_great",
    "works_well", "works_perfectly", "worked_great", "work_great", "work_well",
    "worked_perfectly","works_flawlessly", "worked_flawlessly", "worked_perfect",
    "working_perfectly", "works_wonderfully", "works_amazingly", "work_wonderfully",
    "work_excellent", "work_awesome", "worked_excellent", "quality_construction",
    "built_quality", "fantastic_quality", "perfect_quality", "superior_quality"
])

WORD_SET_BAD_QUALITY = set([
    "poor_quality", "low_quality", "cheap_quality", "bad_quality",
    "quality_control", "poor_build", "stopped_working", "stop_working",
    "never_worked", "quit_working", "nothing_works", "stop_working" 
])

WORD_SET_GOOD_PRICE = set([
    'great_price','good_price','low_price','excellent_price',
    'best_price','cheap_price','affordable_price','awesome_price',
    'well_priced','amazing_price','discounted_price','fantastic_price',
    'priced_well','bargain_price','perfect_price','super_price',
    'great_prices','unbeatable_price','inexpensive_price','lowest_price',
    'incredible_price','terrific_price','wonderful_price','great_value',
    'good_value','excellent_value','best_value','better_value','fantastic_value',
    'amazing_value','outstanding_value','incredible_value','awesome_value'
])

WORD_SET_BAD_PRICE = set([
'little_pricey','pricey','bit_pricey','overpriced','high_price',
'way_overpriced','higher_price','premium_price'
])

STOP_WORDS = set(stopwords.words("english"))

def load_data(file_name_and_path):
    df = pd.read_json(file_name_and_path, lines=True)
    
    return df

def clean_review_df(df):
    # Cleaning the NAN
    review_df["reviewText"] = review_df["reviewText"].fillna(0)
    review_df["summary"] = review_df["summary"].fillna(0)
    # convert reviewText and summary to text objects
    review_df["reviewText"] = review_df["reviewText"].astype("string")
    review_df["summary"] = review_df["summary"].astype("string")

    return review_df


def quality_text_score(text):
    score = 0 
    for token in text:
        if token in WORD_SET_GOOD_QUALITY:
            score += 1
        if token in WORD_SET_BAD_QUALITY:
            score -= 1
    
    return score


def price_text_score(text):
    score = 0 
    for token in text:
        if token in WORD_SET_GOOD_PRICE:
            score += 1

        if token in WORD_SET_BAD_PRICE:
            score -= 1
    
    return score


def process_text(docs, scoring_fxn):
    # lower case
    docs = docs.apply(lambda x: x.lower())
    # tokenize
    docs = docs.apply(lambda x: word_tokenize(x))
    # Remove numbers, but not words that contain numbers.
    docs = docs.apply(lambda x: [token for token in x if not token.isnumeric()])
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    # remove stopwords 
    docs = [[token for token in doc if token not in STOP_WORDS] for doc in docs]
    # Compute bigrams.

    # Add bigrams
    bigram = Phrases(docs, min_count=1)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if "_" in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    return [scoring_fxn(doc) for doc in docs]


def transform_quality_score(x):
    if x > 2:
        return 5
    elif x == 2:
        return 4.0
    elif x == 1:
        return 3.5
    elif x == 0:
        return 2.5
    elif x == -1:
        return 1
    else:
        return 0


def transform_price_score(x):
    if x > 2:
        return 5
    elif x == 2:
        return 4.5
    elif x == 1:
        return 3.5
    elif x == 0:
        return 2.5
    elif x == -1:
        return 1
    else:
        return 0


def generate_quality_features(review_df, output_dataset_dir):
    review_df = review_df.copy()
    list_of_scores = []
    sizes = np.linspace(0, review_df.shape[0], num=11).astype(int)

    for x in range(len(sizes) - 1):
        list_of_scores.extend(
            process_text(
                copy.deepcopy(
                    review_df.reviewText[sizes[x]:sizes[x+1]]
                ),
                quality_text_score
            )
        )

    review_df["quality_score"] = list_of_scores
    sentiment_review = review_df[["asin", "quality_score"]].groupby("asin").sum()
    sentiment_review["Quality_Score"] = sentiment_review.quality_score.apply(
        lambda x: transform_quality_score(x)
    )
    sentiment_review.reset_index(inplace=True)
    del sentiment_review["quality_score"]

    sentiment_review.to_csv(
        f"{output_dataset_dir}/review_quality_scores.csv",
        index=False
    )


def generate_price_features(review_df, output_dataset_dir):
    review_df = review_df.copy()
    list_of_scores = []
    sizes = np.linspace(0, review_df.shape[0], num=11).astype(int)

    for x in range(len(sizes) - 1):
        list_of_scores.extend(
            process_text(
                copy.deepcopy(
                    review_df.reviewText[sizes[x]:sizes[x+1]]
                ),
                price_text_score
            )
        )

    review_df["p_score"] = list_of_scores
    sentiment_review = review_df[["asin", "p_score"]].groupby("asin").sum()
    sentiment_review["Price_Score"] = sentiment_review.p_score.apply(
        lambda x: transform_price_score(x)
    )
    sentiment_review.reset_index(inplace=True)
    del sentiment_review["p_score"]

    sentiment_review.to_csv(
        f"{output_dataset_dir}/review_price_scores.csv",
        index=False
    )


def generate_sentiment(review_df, output_dataset_dir):
    analyzer = SentimentIntensityAnalyzer()
    review_df = review_df.copy()

    review_df["reviewText_score"] = review_df["reviewText"].apply(lambda x: np.nan if x == "0" else analyzer.polarity_scores(x)["compound"])
    review_df["reviewSummary_score"] = review_df["summary"].apply(lambda x: np.nan if x == "0" else analyzer.polarity_scores(x)["compound"])

    sentiment_review = review_df[
        ["asin", "reviewText_score", "reviewSummary_score", "overall"]
    ].groupby("asin").mean()
    sentiment_review.reset_index(inplace = True)

    sentiment_review.to_csv(f"{output_dataset_dir}/review_sentiment_scores.csv", index=False)


def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_data",
        help="The input review data file",
        type=str
    )
    parser.add_argument(
        "--output_dataset_dir",
        help="The output dataset dir",
        type=str
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = parse_args()

    input_data = parsed_args.input_data
    output_dataset_dir = parsed_args.output_dataset_dir
    
    review_df = load_data(input_data)
    review_df = clean_review_df(review_df)

    generate_sentiment(
        review_df,
        output_dataset_dir,
    )
    generate_quality_features(
        review_df,
        output_dataset_dir,
    )
    generate_price_features(
        review_df,
        output_dataset_dir,
    )

    print("Completed")
