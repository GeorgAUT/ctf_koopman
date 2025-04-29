import numpy as np
from typing import Dict, Optional
import pykoopman as pk
from pydmd import DMD
import copy

class KoopmanModel:
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None, training_timesteps: Optional[np.ndarray] = None, prediction_timesteps: Optional[np.ndarray] = None, pair_id: Optional[int] = None):
        """
        Initialize the Koopman model with the given configuration and training data.
        :param config: Configuration dictionary containing model parameters.
        :param
        train_data: Training data for the model.
        :param prediction_time_steps: Number of steps to predict into the future.
        :param pair_id: Identifier for the data pair.
        """
        # Load configuration parameters
        self.config = config
        self.pair_id = pair_id
        self.dataset_name = config['dataset']['name']
        self.dmd_rank = config['model']['dmd_rank']
        self.observables = config['model']['observables']
        self.observables_degree = config['model']['observables_degree']

        self.pair_id = pair_id
        self.train_data = np.transpose(train_data)
        self.train_data = self.train_data.squeeze()
        self.init_data = np.transpose(init_data)
        self.init_data = self.init_data.squeeze()
        self.prediction_timesteps = prediction_timesteps
        self.training_timesteps = training_timesteps[0]
        print("Prediction timesteps:", self.prediction_timesteps)
        self.spatial_dimension = self.train_data.shape[0]


        self.dt = self.prediction_timesteps[1] - self.prediction_timesteps[0]

        if pair_id == 8:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,4]),
                'test_params': np.array([3])
            }
        elif pair_id == 9:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,3]),
                'test_params': np.array([4])
            }
        else:
            self.parametric = None

    def train(self):
        """
        Train the Koopman model using the provided training data.
        """
        dmd = DMD(svd_rank=self.dmd_rank)

        ## TODO: Fix parameter passing to self.config['model']...
        if self.observables == "polynomial":
            pkobservables=pk.observables.Polynomial(degree=self.observables_degree)
        elif self.observables == "time_delay":
            pkobservables=pk.observables.TimeDelay(delay=self.delay, n_delays=self.n_delays)
        elif self.observables == "random_fourier":
            pkobservables=pk.observables.RandomFourierFeatures(include_state=True,gamma=self.gamma,D=self.D)
        elif self.observables == "RBF":
            pkobservables=pk.observables.RadialBasisFunction(
                    rbf_type="thinplate",
                    n_centers=centers.shape[1],
                    centers=centers,
                    kernel_width=1,
                    polyharmonic_coeff=1.0,
                    include_state=True,
                )

        # TODO: Include regressor options
        # Can concatanate observables obs = ob1 + ob2 + ob3 + ob4 + ob5....
        # pykoopman.regression import KDMD
        # EDMD = pk.regression.EDMD()
        
        if self.parametric is None:

            self.model = pk.Koopman(regressor=dmd, observables=pkobservables)
            # observables = [lambda x, y: x * y, lambda x: x**2]
            # model = pk.Koopman(regressor=dmd, observables=pk.observables.CustomObservables(observables))

            print(self.model)

            # Fitting the model to the available training data
            self.model.fit(self.train_data, dt=self.training_timesteps[1]-self.training_timesteps[0])
        else:
            dmd0 = copy.deepcopy(dmd)
            dmd1 = copy.deepcopy(dmd)
            dmd2 = copy.deepcopy(dmd)
            dmd3 = copy.deepcopy(dmd)

            pkobservables0 = copy.deepcopy(pkobservables)
            pkobservables1 = copy.deepcopy(pkobservables)
            pkobservables2 = copy.deepcopy(pkobservables)
            pkobservables3 = copy.deepcopy(pkobservables)

            self.model0 = pk.Koopman(regressor=dmd0, observables=pkobservables0)
            self.model0.fit(self.train_data[0], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model1 = pk.Koopman(regressor=dmd1, observables=pkobservables1)
            self.model1.fit(self.train_data[1], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model2 = pk.Koopman(regressor=dmd2, observables=pkobservables2)
            self.model2.fit(self.train_data[2], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model3 = pk.Koopman(regressor=dmd3, observables=pkobservables3)
            self.model3.fit(self.init_data, dt=self.training_timesteps[1]-self.training_timesteps[0])
            # print(self.model0)
            # 'train_params': np.array([1,2,3]),
            # 'test_params': np.array([4])


    def predict(self):
        if self.parametric is None:
            if abs(self.prediction_timesteps[0])<1e-6:
                init=self.train_data[0]
                # concatante the first time step of the training data with the prediction time steps
                pred_data = self.model.simulate(init, n_steps=self.prediction_timesteps.shape[0]-1)
                pred_data = np.transpose(pred_data)
                pred_data = np.concatenate([np.expand_dims(init,axis=1),pred_data],axis=1)
            else:
                init=self.train_data[-1]
                pred_data = self.model.simulate(init, n_steps=self.prediction_timesteps.shape[0]) # This assumes that train_data[-1] is the time step before the test set
                pred_data = np.transpose(pred_data)
                # Use the last time step of the training data as the initial condition for prediction
        else:
            pred_data = self.predict_parametric()    
        return pred_data# Transpose to match the original data shape
    

    def predict_parametric(self):
        """
        Predict the future states of the system using the trained model.
        :return: Predicted data.
        """
        # Manual set-up for the prediction
        x0=self.init_data[-1]
        # x0=np.transpose(x0)
        n_steps=self.prediction_timesteps.shape[0]

        # Parametric inference
        if x0.ndim == 1:  # handle non-time delay input but 1D accidently
            x0 = x0.reshape(-1, 1)
        elif x0.ndim == 2 and x0.shape[0] > 1:  # handle time delay input
            x0 = x0.T
        else:
            raise TypeError("Check your initial condition shape!")

        y = np.empty((n_steps, self.model0.A.shape[0]), dtype=self.model0.W.dtype)

        # Define lifted initial condition in eigenspace
        self.model0.psi(x0).flatten()


        # # lifted eigen space and move 1 step forward
        # y[0] = self.lamda @ self.psi(x0).flatten()

        # # iterate in the lifted space
        # for k in range(n_steps - 1):
        #     # tmp = self.W @ self.lamda**(k+1) @ y[0].reshape(-1,1)
        #     y[k + 1] = self.lamda @ y[k]
        # x = np.transpose(self.W @ y.T)
        # x = x.astype(self.A.dtype)
