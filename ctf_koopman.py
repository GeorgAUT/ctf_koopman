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
        self.dataset_name = config['dataset']['name']
        self.pair_id = pair_id
        self.train_data = np.transpose(train_data)
        self.prediction_timesteps = prediction_timesteps
        self.spatial_dimension = self.train_data.shape[0]

    def predict(self):
        # TODO: Implement prediction logic for Koopman model
        # pred_data = np.mean(self.train_data, axis=1, keepdims=True)
        # pred_data = np.tile(pred_data, (1, self.prediction_timesteps))


        dmd = DMD(svd_rank=10)
        model = pk.Koopman(regressor=dmd, observables=pk.observables.Polynomial(degree=2))
        # observables = [lambda x, y: x * y, lambda x: x**2]
        # model = pk.Koopman(regressor=dmd, observables=pk.observables.CustomObservables(observables))

        print(model)

        # cut_train = 1000
        dt = self.prediction_timesteps[1] - self.prediction_timesteps[0]
        model.fit(self.train_data, dt=dt)

        # Try prediction
        #pred=model.simulate(self.train_data[-1], n_steps=self.prediction_timesteps.shape[0])



        pred_data = model.simulate(self.train_data[-1], n_steps=self.prediction_timesteps.shape[0]) # This assumes that train_data[-1] is the time step before the test set


        return np.transpose(pred_data) # Transpose to match the original data shape
