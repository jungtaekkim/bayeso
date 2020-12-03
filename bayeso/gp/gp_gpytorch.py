#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is Gaussian process regression implementations with GPyTorch."""

import time
import numpy as np
import torch
import gpytorch

from bayeso import constants
from bayeso.gp import gp_common
from bayeso.utils import utils_gp
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_gpytorch')


class ExactGPModel(gpytorch.models.ExactGP):
    """ExactGPModel"""
    def __init__(self, str_cov, prior_mu, X_train, Y_train, likelihood):
        super().__init__(X_train, Y_train, likelihood)

        self.dim_X = X_train.shape[1]

        if prior_mu is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            raise NotImplementedError()

        if str_cov in ('eq', 'se'):
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.dim_X))
        elif str_cov == 'matern32':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.dim_X))
        elif str_cov == 'matern52':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_X))
        else:
            raise NotImplementedError('allowed str_cov conditions, but it is not implemented.')

    def forward(self, X):
        mean = self.mean_module(X)
        cov = self.covar_module(X)

        return gpytorch.distributions.MultivariateNormal(mean, cov)

@utils_common.validate_types
def get_optimized_kernel(X_train: np.ndarray, Y_train: np.ndarray,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE, str_cov: str,
    fix_noise: bool=constants.FIX_GP_NOISE,
    num_iters: int=1000,
    debug: bool=False
) -> constants.TYPING_TUPLE_TWO_ARRAYS_DICT:
    """
    This function computes the kernel matrix optimized by optimization
    method specified, its inverse matrix, and the optimized hyperparameters,
    using GPyTorch.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param prior_mu: prior mean function or None.
    :type prior_mu: function or NoneType
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param num_iters: the number of iterations for optimizing negative log likelihood.
    :type num_iters: int., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix
        inverse, and dictionary of hyperparameters.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, dict.)

    :raises: AssertionError, ValueError

    """

    # TODO: check to input same fix_noise to convert_hyps and restore_hyps
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(str_cov, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(num_iters, int)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    utils_gp.check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert num_iters >= 10 or num_iters == 0

    # TODO: prior_mu and fix_noise are not working now.
    prior_mu = None
    fix_noise = False

    time_start = time.time()

    if str_cov in constants.ALLOWED_GP_COV_BASE:
#        num_dim = X_train.shape[1]
        pass
    elif str_cov in constants.ALLOWED_GP_COV_SET:
#        num_dim = X_train.shape[2]
        raise NotImplementedError('It is not implemented yet.')
    else:
        raise NotImplementedError

    X_train_ = torch.from_numpy(X_train).double()
    Y_train_ = torch.from_numpy(Y_train.flatten()).double()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(str_cov, prior_mu, X_train_, Y_train_, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=1e-2)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    list_neg_log_likelihoods = []
    ind_iter = 0

    while num_iters >= 10:
        optimizer.zero_grad()
        outputs = model(X_train_)
        loss = -1.0 * mll(outputs, Y_train_)
        loss.backward()
        optimizer.step()
        list_neg_log_likelihoods.append(loss.item())

        if ind_iter > num_iters and np.abs(np.mean(
            list_neg_log_likelihoods[-6:-1]) - loss.item()) < 5e-2:
            break
        if ind_iter > 10 * num_iters: # pragma: no cover
            break

        ind_iter += 1

    model.eval()
    likelihood.eval()

    hyps = {
        'signal': np.sqrt(model.covar_module.outputscale.item()),
        'lengthscales': model.covar_module.base_kernel.lengthscale.detach().numpy()[0],
        'noise': np.sqrt(model.likelihood.noise.item())
    }

    cov_X_X, inv_cov_X_X, _ = gp_common.get_kernel_inverse(X_train, hyps,
        str_cov, fix_noise=fix_noise, debug=debug)

    time_end = time.time()

    if debug:
        logger.debug('iterations to be converged: %d', ind_iter)
        logger.debug('hyps optimized: %s', utils_logger.get_str_hyps(hyps))
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)

    return cov_X_X, inv_cov_X_X, hyps
