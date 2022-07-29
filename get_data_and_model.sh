curl -o datasets.zip https://697-public-datasets.s3.us-east-2.amazonaws.com/datasets.zip
curl -o index.zip https://697-public-datasets.s3.us-east-2.amazonaws.com/index.zip
curl -o models.zip https://697-public-datasets.s3.us-east-2.amazonaws.com/models.zip
curl -o meta_Electronics.json.gz https://697-public-datasets.s3.us-east-2.amazonaws.com/meta_Electronics.json.gz
curl -o R1_data_indexed_lines.json https://697-public-datasets.s3.us-east-2.amazonaws.com/R1_data_indexed_lines.json

unzip datasets.zip
unzip index.zip
unzip models.zip
mv meta_Electronics.json.gz datasets/ && cd datasets/ && gzip -d meta_Electronics.json.gz
mkdir datasets/r1_data/ && mv R1_data_indexed_lines.json datasets/r1_data/
