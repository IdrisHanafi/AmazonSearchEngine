import os
import pandas as pd
import re
from bs4 import BeautifulSoup as BSHTML
from gensim.parsing.preprocessing import (
    preprocess_string,
    STOPWORDS,
    remove_stopword_tokens
)
from argparse import ArgumentParser, Namespace

def load_data(file_name_and_path):
    products = pd.read_json(file_name_and_path, lines=True)
    
    return products

# set elements to remove
remove = ['\t', '\n', '</b>', '<br />', '<br>', '<BR>', '<B>', '</B>', '<b>', '<em>', '</em>', '<span>', '</span>', '&quot;', '<i>', '</i>', '<I>', '</I>', '<a>', '</a>', '<strong>', '</strong>', '<ul>' ,'</ul>', '<li>', '</li>']
regex = re.compile('|'.join(re.escape(r) for r in remove))

def clean_text(text_val):
    """
    This function cleans and tokenizes the "description" column
    and saves it in a column called 'tokens'.
    The cleaning does the following:
    1. strips punctuation
    2. remove multiple whitespace
    3. remove STOPWORDS (imported from gensim's Stone, Denis, Kwantes (2010) dataset)
    4. remove common stopwords, defined above
    5. remove numeric characters
    6. tokenize words
    7. return a lower-case stemmed version of the text
    """
    # clean the combined column
    text_val = re.sub('<.*?>', '', str(text_val)) # remove HTML tags
    text_val = regex.sub('', text_val).strip() # replace remove elements
    text_val = text_val.replace("\\\'", "'")
    text_val = text_val.replace('&amp;', '&')
    text_val = text_val.replace('&#8220;', '"')
    text_val = text_val.replace('&#8221;', '"')
    text_val = text_val.replace('&#8212;', '-')
    text_val = text_val.replace('&#8230;', '...')
    text_val = text_val.replace('&#8217;', "'")
    text_val = text_val.replace('%2C', ",")
    text_val = text_val.replace('%2E', ".")
    text_val = text_val.replace('%2D', "-")
    text_val = text_val.replace('%2F', "/")
    text_val = text_val.replace('%94', '"')
    text_val = text_val.replace('&reg;', '')
    text_val = text_val.replace('&iacute;', "'")
    
    # Remove stop words
    text_val = remove_stopword_tokens(preprocess_string(text_val), stopwords=STOPWORDS)
    
    return text_val

def clean_text_exclude_preprocessing(text_val):
    """
    This function cleans and tokenizes the "description" column
    and saves it in a column called 'tokens'.
    The cleaning does the following:
    1. strips punctuation
    2. remove multiple whitespace
    3. remove STOPWORDS (imported from gensim's Stone, Denis, Kwantes (2010) dataset)
    4. remove common stopwords, defined above
    5. remove numeric characters
    6. tokenize words
    7. return a lower-case stemmed version of the text
    """
    # clean the combined column
    text_val = re.sub('<.*?>', '', str(text_val)) # remove HTML tags
    text_val = regex.sub('', text_val).strip() # replace remove elements
    text_val = text_val.replace("\\\'", "'")
    text_val = text_val.replace('&amp;', '&')
    text_val = text_val.replace('&#8220;', '"')
    text_val = text_val.replace('&#8221;', '"')
    text_val = text_val.replace('&#8212;', '-')
    text_val = text_val.replace('&#8230;', '...')
    text_val = text_val.replace('&#8217;', "'")
    text_val = text_val.replace('%2C', ",")
    text_val = text_val.replace('%2E', ".")
    text_val = text_val.replace('%2D', "-")
    text_val = text_val.replace('%2F', "/")
    text_val = text_val.replace('%94', '"')
    text_val = text_val.replace('&reg;', '')
    text_val = text_val.replace('&iacute;', "'")
    
    return text_val

def clean_data(df):
    """
    Prepares the category, main_cat, description, title, and brand columns
    of the product dataset for model ingestions and further transformations
    ---

    Input:
        df - pandas DataFrame; ideally the meta_Electronics.json file
    Output:
        Returns a pandas DataFrame
    """
    # takes only columns needed from original DataFrame
    df = df[
        ['asin', 'category', 'main_cat', 'description', 'title', 'brand']
    ].rename(columns={'category' : 'original_category'})

    # get unique index column
    df = df.reset_index()

    # cleans category column, preserves list format
    df['category_list'] = df.original_category.map(
        lambda x: [cat.strip().replace('&amp;', '&') for cat in x]
    )

    # regex does the trick in iteratively removing unwanted HTML codes
    df['category_list'] = df.category_list.map(
        lambda x: [regex.sub('', cat) for cat in x]
    )

    # problem is that now we have an empty value in the list; remove that
    df['category_list'] = df.category_list.map(lambda x: list(filter(None, x)))
    
    # problem is that now we have an empty value in the list; remove that
    df['category_tuple'] = df.category_list.apply(lambda x: tuple(x))

    # since category_list is all clean, we just create a string from that column
    df['category_string'] = df.category_list.map(lambda x: ', '.join(map(str, x)))
    #df['category_string'] = df.category_string.map(lambda x: x.strip().replace('&amp;', '&'))

    # clean up main_cat column
    df['main_cat'] = df.main_cat.map(lambda x: x.strip().replace('&amp;', '&'))
    
    # some values in main_cat are img strings; this is the bottleneck
    for i, row in df.iterrows():
        if row['main_cat'] != '' and row['main_cat'][0] == '<':
            soup = BSHTML(row['main_cat']).findAll('img')
            for img in soup:
                df.at[i, 'main_cat'] = img['alt']

    # combine description, title, and brand columns
    df['description'] = df.description.map(lambda x: list(filter(None, x))) # remove empty entries in the list
    df['description'] = df.description.map(lambda x: ', '.join(map(str, x))) # convert description column list into string
    
    df['4_features_combined'] = df['title'] + ' ' + df['category_string'] + ' ' + df['main_cat'] + ' ' + df['brand']
    df['5_features_combined'] = df['title'] + ' ' + df['category_string'] + ' ' + df['main_cat'] + ' ' + df['brand'] + ' ' + df['description']

    # clean title column
    df['title'] = df['title'].apply(clean_text_exclude_preprocessing)

    # clean the combined column
    df['4_features_tokenized'] = df['4_features_combined'].apply(clean_text)
    df['5_features_tokenized'] = df['5_features_combined'].apply(clean_text)

    # get count of unique category groups
    # caveat: since we cannot merge on a list, we used the category_string column as the joining column
    category_group_counts = df.category_string.value_counts().reset_index().rename(
        columns={
            'index': 'category_string',
            'category_string': 'category_group_count'
        }
    )
    df = df.merge(category_group_counts, how='left', on='category_string')

    return df[
        [
            'asin',
            'original_category',
            'category_list',
            'category_tuple',
            'category_string',
            'main_cat',
            'title',
            '4_features_combined',
            '5_features_combined',
            '4_features_tokenized',
            '5_features_tokenized',
            'category_group_count'
        ]
    ]


def write_to_file(df, file):
    """
    Write the file
    """
    filepath = "/".join(file.split("/")[:-1])
    if filepath:
        os.makedirs(filepath, exist_ok=True)
    df.to_json(file)

def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="The Input Raw input File that contains the product metadata",
        type=str
    )
    parser.add_argument(
        "--output_file",
        help="The cleaned output file",
        type=str
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = parse_args()

    input_file = parsed_args.input_file
    output_file = parsed_args.output_file
    
    print(input_file, output_file)
    
    df = load_data(input_file)
    df = clean_data(df)
    print(df.head())
    
    write_to_file(df, output_file)
