#!/usr/bin/env bash

echo "setting up virtual environment"
source 697_venv/bin/activate

./697_venv/bin/pip install -r requirements.txt
echo "completed..."
