import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
import yaml
import numpy as np
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_prediction_timesteps, get_training_timesteps, _load_test_data
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from ctf_koopman import KoopmanModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm





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

        # For testing purposes only
        test_data = _load_test_data(dataset_name, pair_id)

        # Load initialization matrix if it exists
        if init_data is None:
            # Stack all training matrices to get a single training matrix
            train_data = np.concatenate(train_data, axis=1)
        # else:
        #     # If we are given a burn-in matrix, use it as the training matrix
        #     train_data = init_data
        
        # Load metadata (to provide forecast length)
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
        training_timesteps = get_training_timesteps(dataset_name, pair_id)
        if pair_id == 0:
            print("Prediction timesteps:", prediction_timesteps)
            print("Training timesteps:", training_timesteps)
        # prediction_time_steps = prediction_timesteps.shape[0]

        # Initialize the model with the config and train_data
        model = KoopmanModel(config, train_data, init_data, training_timesteps, prediction_timesteps, pair_id)
        
        model.train()
        
        # Generate predictions
        pred_data = model.predict()
        
        # Plot the first component of pred_data
        if config['dataset']['name'] == "ODE_Lorenz":
            plt.figure()
            plt.plot(pred_data[:,0], label=f'pair_id {pair_id} - pred_data[0]')
            plt.title(f'First Component of pred_data for pair_id {pair_id}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()

        elif config['dataset']['name'] == "PDE_KS":
            if pair_id in [8,9]:
                train_data = init_data
            # Plotting for Kuramoto-Sivashinsky (KS) equation
            levels = np.linspace(train_data.min(), train_data.max(), 40)

            fig = plt.figure(figsize=(20, 5))
            fig.suptitle(f"Pair ID: {pair_id} - Model: {model_name} - Dataset: {dataset_name}", fontsize=16)
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)
            axs = [fig.add_subplot(gs[i]) for i in range(3)]

            # Get mesh and times from the dataset
            times = prediction_timesteps
            mesh = np.linspace(0, 1, train_data.shape[0])

            # Plot original data in the first subplot of the figure
            im = axs[0].imshow(train_data, aspect='auto', extent=[times[0], times[-1], mesh[0], mesh[-1]], cmap='viridis')
            axs[0].set_ylabel('Mesh')
            axs[0].set_xlabel('Time')
            axs[0].set_title('Train data')
            fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)


            im = axs[1].imshow(test_data, aspect='auto', extent=[times[0], times[-1], mesh[0], mesh[-1]], cmap='viridis')
            axs[1].set_xlabel('Time')
            axs[1].set_title('Test data')
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)


            im = axs[2].imshow(pred_data, aspect='auto', extent=[times[0], times[-1], mesh[0], mesh[-1]], cmap='viridis')
            axs[2].set_xlabel('Time')
            axs[2].set_title('Prediction')
            fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label='train_data value')


            # input("Press Enter to continue...")

            # cont = plot_ks(axs[0], train_data, mesh['KS'], times['KS'], show_ticks=True, levels=levels)
            # axs[0].set_title('Original data')

            # # Plot Koopman reconstruction (prediction)
            # plot_ks(axs[1], koop_rec, mesh['KS'], times['KS'], show_ticks=True, levels=levels)
            # axs[1].set_title('Koopman')

            # # Plot DMD reconstruction (if available)
            # plot_ks(axs[2], dmd_rec, mesh['KS'], times['KS'], show_ticks=True, levels=levels)
            # axs[2].set_title('DMD')

            # cbar_ax = fig.add_subplot(gs[3])
            # fig.colorbar(cont, cax=cbar_ax)

            # # Mark training and prediction regions
            # for ax in axs:
            #     ax.axvline(times['KS'][cut_train], color='black', linestyle='--')
            #     ax.axvline(times['KS'][2*cut_train], color='black', linestyle='--')

        # K = model.A

        # # Let's have a look at the eigenvalues of the Koopman matrix
        # evals, evecs = np.linalg.eig(K)
        # evals_cont = np.log(evals)#/delta_t

        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # ax.plot(evals_cont.real, evals_cont.imag, 'bo', label='estimated',markersize=5)



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
    

    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=str, help="Path to the configuration file")
    # args = parser.parse_args()
    # main("models/ctf_koopman/config/config1_Lorenz.yaml")
    # main("models/ctf_koopman/config/config1_KS.yaml")
    main("models/ctf_koopman/config/config1_seismo.yaml")
