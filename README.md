# CTF_Koopman Model

This directory contains an implementation of the [PyKoopman](https://github.com/dynamicslab/pykoopman) package for testing on the CTF for Science Framework.

## Files
- `ctf_koopman.py`: Contains the `KoopmanModel` class adapting the PyKoopman logic to the CTF.
- `run.py`: Batch runner script for running the model across multiple sub-datasets.
- `config_XX.yaml`: Example configuration file for running the model.

## Usage

To run test and replicate the results presented in the paper, use the `run.py` script from the **project root** followed by the path to a configuration file. For example:

```bash
python models/ctf_koopman/run.py models/ctf_koopman/optimal_params_ODE_Lorenz_1.yaml
python models/ctf_koopman/run.py models/ctf_koopman/optimal_params_PDE_KS_1.yaml
```

## Hyperparameter tuning

For hyperparameter tuning use `optimize_parameters.py --config_path tuning_config/config_ODE_Lorenz_1.yaml`. The computational budget can be controlled with the `n_trials` variable as well as with the `--time-budget-hours` flag.

## Description

PyKoopman learns the Koopman operator and corresponding eigenbasis from a given set of observables and using a chosen regressor. The details of this can be found in the [PyKoopman paper](https://doi.org/10.21105/joss.05881) and in the following [SIAM review article](https://doi.org/10.1137/21M1401243).

### Configuration Structure

Each configuration file must include the following. The options also highlight which of the PyKoopman Observables and Regressors are implemented in the current version and were hence used for hyperparameter tuning:
- **`dataset`** (required):
  - `name`: The dataset name (e.g., `ODE_Lorenz`, `PDE_KS`).
  - `pair_id`: Specifies sub-datasets to run on. Formats:
    - Single integer: `pair_id: 3`
    - List: `pair_id: [1, 2, 3, 4, 5, 6]`
    - Range string: `pair_id: '1-6'`
    - Omitted or `'all'`: Runs on all sub-datasets.
- **`model`**:
  - `name`: `Koopman`
  - `observables`: `'Identity'`, `'Polynomial'`, `'TimeDelay'`, `'RadialBasisFunctions'`, `'RandomFourierFeatures'`, `'ConcatObservables'` - Determines the type of observables used;
  - `observables_cat_identity`: `Bool` - Determines if the above observables are to be concatenated with the identiy;
  - `observables_int_param`: `Int` - Determines the polynomial degree if `observables='Polynomial'`, the number of timedelay steps if `observables='TimeDelay'` and the parameter `D` if `observables='RandomFourierFeatures'`
  - `observables_float_param`: `Float` - Determines the RBF kernel width if `observables='RadialBasisFunctions'` and the parameter `gamma` if `observables='RandomFourierFeatures'`;
  - `observables_rbf_centers_number`: `Int` - Number of RBF kernels if `observables='RadialBasisFunctions'`;
  - `observables_include_state`: `Bool` - Including the original state space coordinates in the space of observables;
  
  - `regressor`: `'DMD'`, `'EDMD'`, `'HAVOK'`, `'KDMD'`, `'NNDMD'`
  - `regressor_dmd_rank`: `Int` - the truncation rank used in the regressor;
  - `regressor_tlsq_rank`: `Int` - Rank used in the Total Least Squares (TLSQ) pre-processing step (for de-noising), this is only available when `regressor='EDMD', 'KDMD'`.

Example (`models/ctf_koopman/config/config1_Lorenz.yaml`):
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: 1-9  # Example: run on sub-datasets 1 to 3
model:
  name: Koopman
  # Observables parameters
  observables: "Polynomial" # Options: Identity, Polynomial, TimeDelay, RadialBasisFunctions, RandomFourierFeatures, ConcatObservables
  observables_cat_identity: False # Bool: Only called if observables is not Identity


  observables_int_param: 1 # Int: (Corresponds to time-delay and polydegree and D, RFF) Only called if observables is TimeDelay
  observables_rbf_centers_number: 100 # Int: Only called if observables is RadialBasisFunctions
  observables_float_param: 1.0 # Float: (rbf_kernel_width if RBF and gamma if RFF) Only called if observables is RadialBasisFunctions
  observables_include_state: True # Bool: Only called if observables is RandomFourierFeatures

  # Regressor parameters
  regressor: "DMD" # Options: DMD, EDMD, HAVOK, KDMD, NNDMD
  regressor_dmd_rank: 10 # Int: Only called if regressor is DMD, EDMD, HAVOK, KDMD
  regressor_tlsq_rank: 10 # Int: Only called if regressor is EDMD, KDMD
```

## Requirements

PyKoopman for CTF relies on Python@3.10 and the following packages listed in `requirements.txt`:
- numpy
- pydmd @ 0.4.1
- pykoopman @ 1.1.0