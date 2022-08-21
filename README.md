# 697 Capstone: Amazon Product Search Engine 

Built by: James Rosenkoetter, Frederic Gigou, Curtis Guo, Idris Hanafi

## Overview

![Product Demo](https://697-public-datasets.s3.us-east-2.amazonaws.com/SearchEngine.gif)

An extremely light weight end-to-end search engine using simple NLP algorithms.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/downloads)
* [Python3.8+](https://www.python.org/downloads/)
* [PostgreSQL](https://www.postgresql.org/download/)

### 1. Environment Setup

To get started, Let's setup the environment.
If you've already created a virtual environment like below, then you can just run the following:
```
$ source setup.sh
```

Create a Virtual Environment (`venv`) with `python3`:
```
$ python3 -m venv 697_venv
```

Use the virtual env when developing the backend: 
```
$ source 697_venv/bin/activate
```

Install the `python` dependencies:
```
(697_venv) $ pip install -r requirements.txt
```

Moreover, whenever you want to add a new package to our backend, run the following to document the dependencies:
```
(697_venv) $ pip install <package_name> && pip freeze > requirements.txt
```

### 2. Data and Model Setup

Pull all of the data and models to our current directory:

```
(697_venv) $ source ./get_data_and_model.sh
```

### 3. Database and Data Setup

The database is a Postgres DB. Confirm that you have postgres installed on your
environment. To do so, create a `.env` file for Prisma to connect to your DB.

Inside the `.env` file, add the following line:
```
DATABASE_URL="postgresql://<USERNAME>:<PASSWORD>@localhost:5432/amzn_product_db"
```

Afterwards, migrate the schemas into the DB:
```
(697_venv) $ prisma db push
```

Lastly, migrate all of the product metadata into the DB:
```
(697_venv) $ python db/main.py
```

### 4. Run the backend

This will run the backend APIs to call our search engines:

```
(697_venv) $ uvicorn src.backend.main:api --reload
```

### (Optional) Reproduce the pipeline

You can reproduce the pipeline by running the following:

```
(697_venv) $ dvc repro
```

### Documentations

Step 1. Cleaning Data:
`clean_metadata` is a script that takes the input meta data file and returns the cleaned one.

```
python3 src/clean_metadata.py
        --input_bucket <INPUT BUCKET>
        --input_file <Input file from S3 within that bucket>
        --output_file <>
        
Example run:
python3 src/clean_metadata.py
        --input_bucket 697-datasets
        --input_file meta_Electronics-0.1-percent.csv
        --output_file data/cleaned_output.csv
```

