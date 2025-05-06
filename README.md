# CTF_Koopman Model

This directory contains an implementation of the [PyKoopman](https://github.com/dynamicslab/pykoopman) Ppackage for testing on the CTF for Science Framework.

## Files
- `ctf_koopman.py`: Contains the `KoopmanModel` class adapting the PyKoopman logic to the CTF.
- `run.py`: Batch runner script for running the model across multiple sub-datasets.
- `config_XX.yaml`: Example configuration file for running the model.

## Usage

To run test, use the `run.py` script from the **project root** followed by the path to a configuration file. For example:

```bash
python models/ctf_koopman/run.py models/ctf_koopman/config0_Lorenz.yaml
python models/ctf_koopman/run.py models/ctf_koopman/config0_KS.yaml
```

## Description
- Add a detailed description of your Koopman model, its parameters, and usage instructions here.


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
  - `observables`: `Identity`, `Polynomial`, `TimeDelay`, `RadialBasisFunctions`, `RandomFourierFeatures`, `ConcatObservables`
  - `observables_cat_identity`: `Bool`
  - `observables_poly_degree`: `Int`
  - `observables_time_delay`: `Int`
  - `observables_rbf_centers_number`: `Int`
  - `observables_rbf_kernel_width`: `Float`
  - `observables_include_state`: `Bool`
  - `observables_gamma`: `Float`
  - `observables_D`: `Int`

  - `regressor`: `DMD`, `EDMD`, `HAVOK`, `KDMD`, `NNDMD`
  - `regressor_dmd_rank`: `Int`
  - `regressor_tlsq_rank`: `Int`

Example (`models/ctf_koopman/config/config0_Lorenz.yaml`):
```yaml
dataset:
  name: PDE_KS
  pair_id: 1-9  # Example: run on sub-datasets 1 to 3
model:
  name: Koopman
  # Observables parameters
  observables: "Polynomial" # Options: Identity, Polynomial, TimeDelay, RadialBasisFunctions, RandomFourierFeatures, ConcatObservables
  observables_cat_identity: False # Bool: Only called if observables is not Identity


  observables_poly_degree: 1 # Int: Only called if observables is Polynomial
  observables_time_delay: 1 # Int: Only called if observables is TimeDelay
  observables_rbf_centers_number: 100 # Int: Only called if observables is RadialBasisFunctions
  observables_rbf_kernel_width: 1.0 # Float: Only called if observables is RadialBasisFunctions
  observables_include_state: True # Bool: Only called if observables is RandomFourierFeatures
  observables_gamma: 1.0 # Float: Only called if observables is RandomFourierFeatures
  observables_D: 3 # Int: Only called if observables is RandomFourierFeatures

  
  # Regressor parameters
  regressor: "DMD" # Options: DMD, EDMD, HAVOK, KDMD, NNDMD
  regressor_dmd_rank: 10 # Int: Only called if regressor is DMD, EDMD, HAVOK, KDMD
  regressor_tlsq_rank: 10 # Int: Only called if regressor is EDMD, KDMD
```

## Requirements

PyKoopman for CTF relies on the following packages lists in `requirements.txt`:
- numpy
- pydmd > 0.4, <= 0.4.1
- pykoopman