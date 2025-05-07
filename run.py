import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import yaml
import numpy as np
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_prediction_timesteps, get_training_timesteps, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from ctf_koopman import KoopmanModel


def main(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset name and parse pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = "Koopman"
    # Generate a unique batch_id for this run
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize visualization object
    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        
        # Load sub-dataset
        train_data, init_data = load_dataset(dataset_name, pair_id)

        # Load initialization matrix if it exists
        if init_data is None:
            # Stack all training matrices to get a single training matrix
            train_data = np.concatenate(train_data, axis=1)
        
        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        training_timesteps = get_training_timesteps(dataset_name, pair_id)

        # Initialize the model with the config and train_data
        model = KoopmanModel(config, train_data, init_data, training_timesteps, prediction_timesteps, pair_id)
                
        # Train the model
        model.train()
        
        # Generate predictions
        pred_data = model.predict()

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
    #main("models/ctf_koopman/config/config0_Lorenz.yaml")
    #main("models/ctf_koopman/config/config0_KS.yaml")
    # main("models/ctf_koopman/config/config_ODE_Lorenz_9_optimized.yaml")
