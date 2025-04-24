import argparse
import yaml
import numpy as np
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_prediction_timesteps, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from ctf_koopman import KoopmanModel


def main(config_path):
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

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        train_data, init_data = load_dataset(dataset_name, pair_id)

        # Load initialization matrix if it exists
        if init_data is None:
            # Stack all training matrices to get a single training matrix
            train_data = np.concatenate(train_data, axis=1)
        else:
            # If we are given a burn-in matrix, use it as the training matrix
            train_data = init_data
        
        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        # prediction_time_steps = prediction_timesteps.shape[0]

        # Initialize the model with the config and train_data
        model = KoopmanModel(config, train_data, prediction_timesteps, pair_id)
        
        
        # Generate predictions
        pred_data = model.predict()
        
        # Plot the first component of pred_data
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(pred_data[0], label=f'pair_id {pair_id} - pred_data[0]')
        plt.title(f'First Component of pred_data for pair_id {pair_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        K = model.A

        # Let's have a look at the eigenvalues of the Koopman matrix
        evals, evecs = np.linalg.eig(K)
        evals_cont = np.log(evals)#/delta_t

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.plot(evals_cont.real, evals_cont.imag, 'bo', label='estimated',markersize=5)
        plt.show()



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

    # Print batch results in a nice table
    print("\nBatch Results Summary:")
    if batch_results['pairs']:
        # Get all metric names from the first result
        for entry in batch_results['pairs']:
            metric_names = list(entry['metrics'].keys())
            # Print header
            header = ["pair_id"] + metric_names
            print(" | ".join(f"{h:>15}" for h in header))
            print("-" * (18 * len(header)))
            # Print each row
            row = [str(entry['pair_id'])] + [f"{entry['metrics'][m]:.6f}" if isinstance(entry['metrics'][m], float) else str(entry['metrics'][m]) for m in metric_names]
            print(" | ".join(f"{v:>15}" for v in row))
            print("-" * (18 * len(header)))
    else:
        print("No results available.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=str, help="Path to the configuration file")
    # args = parser.parse_args()
    main("models/ctf_koopman/config/config0_Lorenz.yaml")
    # main("models/ctf_koopman/config/config0_KS.yaml")
