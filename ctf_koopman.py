import numpy as np
from typing import Dict, Optional
import pykoopman as pk
from pydmd import DMD
import copy
from numpy.polynomial.polynomial import polyfit, polyval # For polynomial fitting in parametric case

class KoopmanModel:
    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None, training_timesteps: Optional[np.ndarray] = None, prediction_timesteps: Optional[np.ndarray] = None, pair_id: Optional[int] = None):
        """
        Initialize the Koopman model with the given configuration and training data.
        :param config: Configuration dictionary containing model parameters.
        :param
        train_data: Training data for the model.
        :param
        init_data: warm-start data for parametric regimes.
        :param prediction_time_steps: time instances at which to predict.
        :param training_time_steps: time instances at which to train.
        :param pair_id: Identifier for the data pair.
        """
        # Load configuration parameters
        self.config = config
        self.pair_id = pair_id
        self.dataset_name = config['dataset']['name']
        self.pair_id = pair_id

        # Load training data (need to reshape it for PyKoopman)
        self.train_data = train_data
        # self.train_data = self.train_data.squeeze()
        self.init_data = init_data
        # if self.init_data is not None:
        #     self.init_data = self.init_data.squeeze()
        
        # Load auxiliary dataparameters
        self.prediction_timesteps = prediction_timesteps
        self.training_timesteps = training_timesteps[0]
        self.dt = self.prediction_timesteps[1] - self.prediction_timesteps[0]

        # Set up parametric regimes (interpolation and extrapolation)
        if pair_id == 8:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,4]),
                'test_params': np.array([3])
            }
            self.spatial_dimension = self.train_data[0].shape[1]
        elif pair_id == 9:
            self.parametric = {
                'mode': config['model']['parametric'] if 'parametric' in config['model'] else 'monolithic',
                'train_params': np.array([1,2,3]),
                'test_params': np.array([4])
            }
            self.spatial_dimension = self.train_data[0].shape[1]
        else:
            self.parametric = None
            self.spatial_dimension = self.train_data.shape[1]
        

        np.random.seed(self.config['model']['seed'])  # set random seed for reproducibility

    def train(self):
        """
        Train the Koopman model using the provided training data.
        """


        # Firstly choose the Koopman observables

        if self.config['model']['observables'] == "Identity":
            pkobservables=pk.observables.Identity()
        if self.config['model']['observables'] == "Polynomial":
            pkobservables=pk.observables.Polynomial(degree=self.config['model']['observables_int_param'])
        elif self.config['model']['observables'] == "TimeDelay":
            pkobservables=pk.observables.TimeDelay(delay=self.dt, n_delays=self.config['model']['observables_int_param'])
        elif self.config['model']['observables'] == "RandomFourierFeatures":
            pkobservables=pk.observables.RandomFourierFeatures(include_state=self.config['model']['observables_include_state'],gamma=self.config['model']['observables_float_param'],D=self.config['model']['observables_int_param'],random_state=self.config['model']['seed'])
        elif self.config['model']['observables'] == "RadialBasisFunctions":
            centers = np.random.uniform(-1,1,(self.spatial_dimension,self.config['model']['observables_rbf_centers_number']))
            pkobservables=pk.observables.RadialBasisFunction(
                    rbf_type="thinplate",
                    n_centers=centers.shape[1],
                    centers=centers,
                    kernel_width=self.config['model']['observables_float_param'],
                    polyharmonic_coeff=1.0,
                    include_state=True,
                )
            
        if self.config['model']['observables_cat_identity'] == "True" and self.config['model']['observables'] != "Identity":
            # Concatenate the identity observables with the other observables
            pkobservables = pkobservables + pk.observables.Identity()

        # Define regressor
        if self.config['model']['regressor'] == "DMD":
            pkregressor = DMD(svd_rank=self.config['model']['regressor_dmd_rank'])
        elif self.config['model']['regressor'] == "EDMD":
            pkregressor = pk.regression.EDMD(svd_rank=self.config['model']['regressor_dmd_rank'],tlsq_rank=self.config['model']['regressor_tlsq_rank'])
        elif self.config['model']['regressor'] == "HAVOK":
            pkregressor = pk.regression.HAVOK(svd_rank=self.config['model']['regressor_dmd_rank'])
        elif self.config['model']['regressor'] == "KDMD":
            pkregressor = pk.regression.KDMD(svd_rank=self.config['model']['regressor_dmd_rank'],tlsq_rank=self.config['model']['regressor_tlsq_rank'])
        elif self.config['model']['regressor'] == "NNDMD":
            pkregressor = pk.regression.NNDMD(config_encoder=dict(
                input_size=self.spatial_dimension+1, hidden_sizes=[self.config['model']['regressor_dmd_rank']] * 2, output_size=6, activations="tanh"),
                config_decoder=dict(
                    input_size=6, hidden_sizes=[self.config['model']['regressor_dmd_rank']] * 2, output_size=self.spatial_dimension+1, activations="linear"),
                    trainer_kwargs=dict(max_epochs=10,accelerator="gpu", devices=1))
        
        # Train the model (two cases for parametric and non-parametric)
        if self.parametric is None:
            self.model = pk.Koopman(regressor=pkregressor, observables=pkobservables)
            # Fit the model to the training data
            self.model.fit(self.train_data, dt=self.training_timesteps[1]-self.training_timesteps[0])
        else:
            pkregressor0 = copy.deepcopy(pkregressor)
            pkregressor1 = copy.deepcopy(pkregressor)
            pkregressor2 = copy.deepcopy(pkregressor)
            pkregressor3 = copy.deepcopy(pkregressor)

            pkobservables0 = copy.deepcopy(pkobservables)
            pkobservables1 = copy.deepcopy(pkobservables)
            pkobservables2 = copy.deepcopy(pkobservables)
            pkobservables3 = copy.deepcopy(pkobservables)

            self.model0 = pk.Koopman(regressor=pkregressor0, observables=pkobservables0)
            self.model0.fit(self.train_data[0], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model1 = pk.Koopman(regressor=pkregressor1, observables=pkobservables1)
            self.model1.fit(self.train_data[1], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model2 = pk.Koopman(regressor=pkregressor2, observables=pkobservables2)
            self.model2.fit(self.train_data[2], dt=self.training_timesteps[1]-self.training_timesteps[0])
            self.model3 = pk.Koopman(regressor=pkregressor3, observables=pkobservables3)
            self.model3.fit(self.init_data, dt=self.training_timesteps[1]-self.training_timesteps[0])

    def predict(self):
        if self.parametric is None:
            if abs(self.prediction_timesteps[0])<1e-6:
                init=self.train_data[0]
                # concatante the first time step of the training data with the prediction time steps
                pred_data = self.model.simulate(init, n_steps=self.prediction_timesteps.shape[0]-1)
                #pred_data = np.transpose(pred_data)
                pred_data = np.concatenate([np.expand_dims(init,axis=0),pred_data],axis=0)
            else:
                init=self.train_data[-1]
                pred_data = self.model.simulate(init, n_steps=self.prediction_timesteps.shape[0]) # This assumes that train_data[-1] is the time step before the test set
                #pred_data = pred_data
                # Use the last time step of the training data as the initial condition for prediction
        else:
            pred_data = self.predict_parametric()    
        return pred_data

    def predict_parametric(self):
        """
        Predict the future states of the system using the trained model.
        :return: Predicted data.
        """
        # Manual set-up for the prediction
        x0=self.init_data[-1]
        x0=np.transpose(x0)
        n_steps=self.prediction_timesteps.shape[0]

        # Parametric inference
        if x0.ndim == 1:  # handle non-time delay input but 1D accidently
            x0 = x0.reshape(-1, 1)
        elif x0.ndim == 2 and x0.shape[0] > 1:  # handle time delay input
            x0 = x0.T
        else:
            raise TypeError("Check your initial condition shape!")


        y = np.empty((n_steps, self.model0.A.shape[0]), dtype=self.model0.W.dtype)

        # Define parameter regimes
        p=np.concatenate([self.parametric['train_params'],self.parametric['test_params']],axis=0)

        # Define lifted initial condition in eigenspace
        x00=self.model0.psi(x0).flatten()
        x01=self.model1.psi(x0).flatten()
        x02=self.model2.psi(x0).flatten()
        x03=self.model3.psi(x0).flatten()
        
        x0_data = np.array([x00,x01,x02,x03])
    
        degree = 1
        m=x0_data.shape[1]
        
        # Create coefficient array: shape (degree+1, m)
        xcoeffs = np.zeros((degree + 1, m))

        # Fit each entry A[i,j] across p-values
        for i in range(m):
            aux = x0_data[:, i]
            xcoeffs[:, i] = polyfit(p, aux, degree)

        psix0=np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], xcoeffs, axes=1)  # shape (m, n)

        # Now fit the eigenvalues
        lambda0 = np.diag(self.model0.lamda)
        lambda1 = np.diag(self.model1.lamda)
        lambda2 = np.diag(self.model2.lamda)
        lambda3 = np.diag(self.model3.lamda)

        # Create coefficient array: shape (degree+1, m)
        lambdacoeffs = np.zeros((degree + 1, lambda0.shape[0]))

        # Fit each entry A[i,j] across p-values
        for i in range(lambda0.shape[0]):
            aux = np.array([lambda0[i], lambda1[i], lambda2[i], lambda3[i]])
            lambdacoeffs[:, i] = polyfit(p, aux, degree)
        
        lambdanew=np.diag(np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], lambdacoeffs, axes=1))  # shape (m, n)

        # Now fit the return transform W
        W0 = self.model0.W
        W1 = self.model1.W
        W2 = self.model2.W
        W3 = self.model3.W

        # Create coefficient array: shape (degree+1, m)
        Wcoeffs = np.zeros((degree + 1, W0.shape[0], W0.shape[1]))
        # Fit each entry A[i,j] across p-values
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                aux = np.array([W0[i, j], W1[i, j], W2[i, j], W3[i, j]])
                Wcoeffs[:, i, j] = polyfit(p, aux, degree)

        W=np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], Wcoeffs, axes=1)  # shape (m, n)

        # Now predict
        y[0] = lambdanew @ psix0
        # # iterate in the lifted space
        for k in range(n_steps - 1):
            y[k + 1] = lambdanew @ y[k]
        x = W @ y.T
        x = x.astype(W.dtype)
        return x.T
    
    # def predict_parametric(self):
    #     """
    #     Predict the future states of the system using the trained model.
    #     :return: Predicted data.
    #     """
    #     # Manual set-up for the prediction
    #     x0=self.init_data[-1]
    #     # x0=np.transpose(x0)
    #     n_steps=self.prediction_timesteps.shape[0]

    #     # Parametric inference
    #     if x0.ndim == 1:  # handle non-time delay input but 1D accidently
    #         x0 = x0.reshape(-1, 1)
    #     elif x0.ndim == 2 and x0.shape[0] > 1:  # handle time delay input
    #         x0 = x0.T
    #     else:
    #         raise TypeError("Check your initial condition shape!")

    #     y = np.empty((n_steps, self.model0.A.shape[0]), dtype=self.model0.W.dtype)

    #     # Define lifted initial condition in eigenspace
    #     x00=self.model0.psi(x0).flatten()
    #     x01=self.model1.psi(x0).flatten()
    #     x02=self.model2.psi(x0).flatten()
    #     x03=self.model3.psi(x0).flatten()
        
    #     x0_data = np.array([x00,x01,x02,x03])
    
    #     degree = 2
    #     m=x0_data.shape[1]
        
    #     # Create coefficient array: shape (degree+1, m)
    #     xcoeffs = np.zeros((degree + 1, m))

    #     # Fit each entry A[i,j] across p-values
    #     for i in range(m):
    #         aux = x0_data[:, i]
    #         xcoeffs[:, i] = polyfit(self.parametric['train_params'], aux, degree)

    #     psix0=np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], xcoeffs, axes=1)  # shape (m, n)

    #     # Now fit the eigenvalues
    #     lambda0 = np.diag(self.model0.lamda)
    #     lambda1 = np.diag(self.model1.lamda)
    #     lambda2 = np.diag(self.model2.lamda)
    #     lambda3 = np.diag(self.model3.lamda)

    #     # Create coefficient array: shape (degree+1, m)
    #     lambdacoeffs = np.zeros((degree + 1, lambda0.shape[0]))

    #     # Fit each entry A[i,j] across p-values
    #     for i in range(lambda0.shape[0]):
    #         aux = np.array([lambda0[i], lambda1[i], lambda2[i]])
    #         lambdacoeffs[:, i] = polyfit(self.parametric['train_params'], aux, degree)
        
    #     lambdanew=np.diag(np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], lambdacoeffs, axes=1))  # shape (m, n)

    #     # Now fit the return transform W
    #     W0 = self.model0.W
    #     W1 = self.model1.W
    #     W2 = self.model2.W

    #     # Create coefficient array: shape (degree+1, m)
    #     Wcoeffs = np.zeros((degree + 1, W0.shape[0], W0.shape[1]))
    #     # Fit each entry A[i,j] across p-values
    #     for i in range(W0.shape[0]):
    #         for j in range(W0.shape[1]):
    #             aux = np.array([W0[i, j], W1[i, j], W2[i, j]])
    #             Wcoeffs[:, i, j] = polyfit(self.parametric['train_params'], aux, degree)

    #     W=np.tensordot([self.parametric['test_params'][0]**d for d in range(degree + 1)], Wcoeffs, axes=1)  # shape (m, n)

    #     # Now predict
    #     y[0] = lambdanew @ psix0
    #     # # iterate in the lifted space
    #     for k in range(n_steps - 1):
    #         y[k + 1] = lambdanew @ y[k]
    #     x = W @ y.T
    #     x = x.astype(W.dtype)
    #     return x
