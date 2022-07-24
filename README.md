# 697 Capstone: Amazon Product Search Engine 

## Overview

This is the repository that includes the code, data, and model components for the engine.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/downloads)
* [Python3.8+](https://www.python.org/downloads/)

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

### 2. Reproduce the pipeline

Afterwards, you can reproduce the pipeline by running the following:

```
(697_venv) $ dvc repro
```

### 3. Run the backend

This will run the backend APIs to call our search engines:

```
(697_venv) $ uvicorn src.backend.main:api --reload
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

