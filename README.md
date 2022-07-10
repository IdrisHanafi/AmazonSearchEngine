# 697 Capstone: Amazon Product Search Engine 

## Overview

This is the repository that includes the code, data, and model components for the engine.

## Prerequisites

You will need the following things properly installed on your computer.

* [Git](https://git-scm.com/downloads)
* [Python3.6+](https://www.python.org/downloads/)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 1. Environment Setup

To get started, setup the environment:

```
$ conda env create -f environment.yml
$ conda activate 697-env
```

### 2. Reproduce the output

Afterwards, you can reproduce the pipeline by running the following:

```
$ dvc repro
```