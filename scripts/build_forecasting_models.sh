#!/bin/bash

BASE_DIR="$(dirname $(pwd))"

# score metric: mean absolute percentage error, criteria: min
$BASE_DIR/python_code/tools/build_forecasting_models.py

# score metric: root mean square error, criteria: min
$BASE_DIR/python_code/tools/build_forecasting_models.py -sm rmse -c min
