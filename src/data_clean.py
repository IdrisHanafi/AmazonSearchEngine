import boto3
import pandas as pd

def load_data_from_s3(bucket, file_name_and_path):
    data_location = 's3://{}/{}'.format(bucket, data_key)
    df = pd.read_csv(data_location)
    
    return df

if __name__ == "__main__":
    bucket = '697-datasets' # Or whatever you called your bucket
    data_key = 'meta_Electronics-0.1-percent.csv' # Where the file is within your bucket
    
    df = load_data_from_s3(bucket, data_key)
    print(df.head())