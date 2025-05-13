#!/bin/bash

set -e

for i in {2..9}; do
    config_path="tuning_config/config_PDE_KS_${i}.yaml"
    echo "Running optimize_parameters.py with ${config_path}..."
    python optimize_parameters.py --config-path "$config_path"
done
echo "All tuning jobs completed."