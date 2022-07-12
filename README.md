# 697 Capstone: Amazon Product Search Engine 

## Overview

This is the repository that includes the code, data, and model components for the engine.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/downloads)
* [Python3.8+](https://www.python.org/downloads/)

### 1. Environment Setup

To get started, Let's setup the environment.
create a Virtual Environment (`venv`) with `python3`:
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

### 2. Reproduce the output

Afterwards, you can reproduce the pipeline by running the following:

```
(697_venv) $ dvc repro
```