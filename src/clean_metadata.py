import os
import boto3
import pandas as pd
from argparse import ArgumentParser, Namespace

def load_data_from_s3(bucket, file_name_and_path):
    data_location = 's3://{}/{}'.format(bucket, file_name_and_path)
    df = pd.read_csv(data_location)
    
    return df

def clean_data(df):
    """
    Input:
        df - The metadata to clean
    Parses the needed features and explodes columns
    """
    features = [
        'category',
        'description',
        'title',
        'also_buy',
        'brand',
        'feature',
        'rank',
        'also_view',
        'main_cat',
        'similar_item',
        'date',
        'price',
        'asin',
        'imageURL',
        'imageURLHighRes',
    ]
    
    df = df[features]
    
    return df

def write_to_csv(df, file):
    """
    Write the file
    """
    filepath = "/".join(file.split("/")[:-1])
    if filepath:
        os.makedirs(filepath, exist_ok=True)
    df.to_csv(file)

def parse_args() -> Namespace:
    """parse arguments"""
    parser = ArgumentParser()
    parser.add_argument(
        "--input_bucket",
        help="The S3 Bucket that contains input files to read from",
        type=str
    )
    parser.add_argument(
        "--input_file",
        help="The Input S3 File Directory that contains csv files to read from",
        type=str
    )
    parser.add_argument(
        "--output_file",
        help="The output file",
        type=str
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    
    parsed_args = parse_args()

    bucket = parsed_args.input_bucket
    input_file = parsed_args.input_file
    output_file = parsed_args.output_file
    
    print(bucket, input_file)
    
    df = load_data_from_s3(bucket, input_file)
    df = clean_data(df)
    print(df.head())
    
    write_to_csv(df, output_file)