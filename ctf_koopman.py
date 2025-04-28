import numpy as np
from typing import Dict, Optional
import pykoopman as pk
from pydmd import DMD

class KoopmanModel:
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, prediction_timesteps: Optional[np.ndarray] = None, pair_id: Optional[int] = None):
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
        self.prediction_timesteps = prediction_timesteps
        self.spatial_dimension = self.train_data.shape[0]


        self.dt = self.prediction_timesteps[1] - self.prediction_timesteps[0]

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
        
        self.model = pk.Koopman(regressor=dmd, observables=pkobservables)
        # observables = [lambda x, y: x * y, lambda x: x**2]
        # model = pk.Koopman(regressor=dmd, observables=pk.observables.CustomObservables(observables))

        print(self.model)

        # cut_train = 1000
        self.model.fit(self.train_data, dt=self.dt)

        self.A = self.model.A


        # # Print the eigenvalues and modes for debugging
        # print("Eigenvalues:", self.model.eigenvalues)
        # print("Modes:", self.model.modes)

        # Try prediction
        #pred=model.simulate(self.train_data[-1], n_steps=self.prediction_timesteps.shape[0])


    def predict(self):
        if self.pair_id in [1]:
            return self.predict_pure_forecast()
        


        pred_data = self.model.simulate(self.train_data[-1], n_steps=self.prediction_timesteps.shape[0]) # This assumes that train_data[-1] is the time step before the test set


        return np.transpose(pred_data) # Transpose to match the original data shape

    def predict_pure_forecast(self):
        """
        Predict the next time steps using the trained model.
        :return: Predicted data for the next time steps.
        """
        # Use the last time step of the training data as the initial condition for prediction
        initial_condition = self.train_data[-1]
        pred_data = self.model.simulate(initial_condition, n_steps=self.prediction_timesteps.shape[0])
        return np.transpose(pred_data)  # Transpose to match the original data shape