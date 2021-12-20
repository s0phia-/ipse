import numpy as np
from numba import njit
import scipy.optimize as optim
import torch
from stew.utils import create_diff_matrix
import itertools


@njit
def stew_reg(X, y, D, lam):
    return np.linalg.inv(X.T @ X + lam * D) @ X.T @ y


def stew_loss(beta, X, y, D, lam):
    residuals = y - X @ beta
    l = residuals.T.dot(residuals) + lam * beta.T.dot(D).dot(beta)
    return l


def stew_grad(beta, X, y, D, lam):
    return 2 * np.dot(beta, X.T).dot(X) - 2 * y.T.dot(X) + 2 * lam * beta.dot(D)


def stew_hessian(beta, X, y, D, lam):
    return 2 * X.T.dot(X) + 2 * lam * D


def stew_reg_iter(X, y, D, lam, method='Newton-CG'):
    op = optim.minimize(fun=stew_loss, x0=np.zeros(X.shape[1]), args=(X, y, D, lam),
                        jac=stew_grad, hess=stew_hessian, method=method)
    return op.x


class LinearRegressionTorch:
    def __init__(self,
                 num_features,
                 learning_rate=0.1,
                 regularization="none",
                 positivity_constraint=False,
                 lam=0,
                 verbose=False):
        self.num_features = num_features
        self.model = LinearRegressionModel(self.num_features)
        self.loss = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.regularization = regularization
        assert(self.regularization in ["none",
                                       "stem_opt",
                                       "ridge", "lasso",
                                       "stew1", "stew2",
                                       "stem1", "stem2",
                                       "stow", "stnw",
                                       "sted", "sthd",
                                       "peno", "penlex"])
        self.positivity_constraint = positivity_constraint
        self.positivity_weight_clipper = PositivityClipper()
        self.lam = lam
        self.D = torch.from_numpy(create_diff_matrix(num_features=num_features)).float()
        self.verbose = verbose

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            self.optimizer.zero_grad()

            if self.lam > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = np.minimum(self.learning_rate, 1 / (self.lam))

            # get output from the model, given the inputs
            outputs = self.model(torch.from_numpy(X).float())

            # get loss for the predicted output
            loss = self.loss(outputs, torch.from_numpy(y).float().reshape((-1, 1)))
            if self.regularization == "stew2":
                stew2_reg = torch.tensor(0.)
                for i in range(self.num_features-1):
                    for j in range(i+1, self.num_features):
                        stew2_reg += torch.pow(self.model.input_linear.weight[0][i] - self.model.input_linear.weight[0][j], 2)
                stew2_reg /= self.num_features * (self.num_features + 1) / 2
                loss += self.lam * stew2_reg
            elif self.regularization == "stew1":
                stew1_reg = torch.tensor(0.)
                for i in range(self.num_features-1):
                    for j in range(i+1, self.num_features):
                        stew1_reg += torch.abs(self.model.input_linear.weight[0][i] - self.model.input_linear.weight[0][j])
                stew1_reg /= self.num_features * (self.num_features + 1) / 2
                loss += self.lam * stew1_reg
            elif self.regularization == "ridge":
                ridge_reg = torch.tensor(0.)
                for i in range(self.num_features):
                    ridge_reg += torch.pow(self.model.input_linear.weight[0][i], 2)
                ridge_reg /= self.num_features
                loss += self.lam * ridge_reg
            elif self.regularization == "lasso":
                lasso_reg = torch.tensor(0.)
                for i in range(self.num_features):
                    lasso_reg += torch.abs(self.model.input_linear.weight[0][i])
                lasso_reg /= self.num_features
                loss += self.lam * lasso_reg
            elif self.regularization == "stem2":
                stem2_reg = torch.tensor(0.)
                for i in range(self.num_features - 1):
                    for j in range(i + 1, self.num_features):
                        stem2_reg += torch.pow(torch.abs(self.model.input_linear.weight[0][i]) - torch.abs(self.model.input_linear.weight[0][j]), 2)
                stem2_reg /= self.num_features * (self.num_features + 1) / 2
                loss += self.lam * stem2_reg
            elif self.regularization == "stem1":
                stem1_reg = torch.tensor(0.)
                for i in range(self.num_features - 1):
                    for j in range(i + 1, self.num_features):
                        stem1_reg += torch.abs(torch.abs(self.model.input_linear.weight[0][i]) - torch.abs(self.model.input_linear.weight[0][j]))
                stem1_reg /= self.num_features * (self.num_features + 1) / 2
                loss += self.lam * stem1_reg
            elif self.regularization == "stow":
                stow_reg = torch.tensor(0.)
                for i in range(1, self.num_features):
                    stow_reg += torch.clamp(self.model.input_linear.weight[0][i] - self.model.input_linear.weight[0][i - 1], min=0)
                stow_reg /= (self.num_features - 1)
                loss += self.lam * stow_reg
            elif self.regularization == "peno":
                peno = torch.tensor(0.)
                for i in range(1, self.num_features):
                    peno += self.model.input_linear.weight[0][i] < self.model.input_linear.weight[0][i - 1]
                    # peno += torch.clamp(self.model.input_linear.weight[0][i] - self.model.input_linear.weight[0][i - 1], min=0)
                peno /= (self.num_features - 1)
                loss += self.lam * peno
            elif self.regularization == "stnw":
                stnw_reg = torch.tensor(0.)
                for i in range(self.num_features-1):
                    stnw_reg += torch.clamp(self.model.input_linear.weight[0][(i+1):].sum() - self.model.input_linear.weight[0][i], min=0)
                stnw_reg /= (self.num_features - 1)
                loss += self.lam * stnw_reg
            elif self.regularization == "sted":
                sted_reg = torch.tensor(0.)
                for i in range(self.num_features - 1):
                    sted_reg += torch.pow(self.model.input_linear.weight[0][i] - 2 * self.model.input_linear.weight[0][i+1], 2)
                sted_reg /= (self.num_features - 1)
                loss += self.lam * sted_reg
            elif self.regularization == "sthd":
                sthd_reg = torch.tensor(0.)
                for i in range(self.num_features - 1):
                    sthd_reg += torch.pow(self.model.input_linear.weight[0][i] - (i + 2)/(i + 1) * self.model.input_linear.weight[0][i+1], 2)
                sthd_reg /= (self.num_features - 1)
                loss += self.lam * sthd_reg

            if self.verbose:
                print('epoch {}, loss {}'.format(epoch, loss.item()))
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            self.optimizer.step()

            if self.positivity_constraint:
                self.model.input_linear.apply(self.positivity_weight_clipper)

        return self.model.input_linear.weight[0].detach().numpy()  # , loss.item()

    def predict(self, X):
        """
        Predict on (new) data
        :param X: features matrix as numpy array (is converted to torch.tensor within this method.
        :return predictions:
        """
        with torch.no_grad():
            return self.model(torch.from_numpy(X).float()).numpy().flatten()


class STEMopt:
    def __init__(self,
                 train_fraction,
                 num_features,
                 learning_rate=0.1,
                 regularization="none",
                 positivity_constraint=False,
                 lam=0,
                 verbose=False
                 ):
        self.train_fraction = train_fraction
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.positivity_constraint = positivity_constraint
        self.lam = lam
        self.verbose = verbose
        self.train_fraction = train_fraction
        if regularization == "stem_opt":
            self.regularization = "stew2"
        # self.model = LinearRegressionTorch(self.num_features, self.learning_rate,
        #                                    self.regularization, self.positivity_constraint,
        #                                    self.lam, self.verbose)
        self.D = create_diff_matrix(self.num_features)
        self.beta = None

    def fit(self, X, y, epochs):
        num_samples, num_features = X.shape
        # permuted_indices = np.random.permutation(np.arange(num_samples))
        # train_set_size = int(np.floor(self.train_fraction * num_samples))
        # train_indices = permuted_indices[:train_set_size]
        # test_indices = permuted_indices[train_set_size:]
        # X_train = X[train_indices, :]
        # X_test = X[test_indices, :]
        # y_train = y[train_indices]
        # y_test = y[test_indices]
        configurations = np.array(list(itertools.product([-1, 1], repeat=num_features)))
        num_configurations = len(configurations)
        betas = np.zeros((num_configurations, num_features))
        losses = np.zeros(num_configurations)
        for configuration_ix, configuration in enumerate(configurations):  # configuration_ix = 0; configuration = configurations[configuration_ix]
            # print(configuration_ix, configuration)
            confd_X = X * configuration[np.newaxis, :]
            # confd_X_test = X_test * configuration[np.newaxis, :]
            # self.model = LinearRegressionTorch(self.num_features, self.learning_rate,
            #                                    self.regularization, self.positivity_constraint,
            #                                    self.lam, self.verbose)
            # beta, loss = self.model.fit(confd_X, y, epochs)
            beta = stew_reg(confd_X, y, self.D, self.lam)
            loss = stew_loss(beta, confd_X, y, self.D, self.lam)
            betas[configuration_ix, :] = beta * configuration
            losses[configuration_ix] = loss

        argmin_loss = np.argmin(losses)
        argmin_beta = betas[argmin_loss, :]
        self.beta = argmin_beta
        return argmin_beta

    def predict(self, X):
        return X @ self.beta


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, num_features):
        super(LinearRegressionModel, self).__init__()
        self.num_features = num_features
        self.input_linear = torch.nn.Linear(in_features=self.num_features, out_features=1, bias=False)

    def forward(self, choice_sets):
        y_pred = self.input_linear(choice_sets)
        return y_pred


class PositivityClipper:
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            module.weight.data = torch.clamp(module.weight.data, min=0)


