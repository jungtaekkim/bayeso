#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is Gaussian process regression implementations with TensorFlow."""

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayeso import constants
from bayeso.gp import gp_common
from bayeso.utils import utils_gp
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_tensorflow')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: # pragma: no cover
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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
    using TensorFlow and TensorFlow probability.

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
    :param num_iters: the number of iterations for optimizing negative log
        likelihood.
    :type num_iters: int., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix inverse,
        and dictionary of hyperparameters.
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
        num_dim = X_train.shape[1]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        num_dim = X_train.shape[2]
        raise NotImplementedError('It is not implemented yet.')

    constraint_positive = tfp.bijectors.Shift(np.finfo(np.float64).tiny)(tfp.bijectors.Exp())

    var_amplitude = tfp.util.TransformedVariable(
        initial_value=1.0,
        bijector=constraint_positive,
        dtype=np.float64
    )

    var_length_scale = tfp.util.TransformedVariable(
        initial_value=[1.0] * num_dim,
        bijector=constraint_positive,
        dtype=np.float64
    )

    var_observation_noise_variance = tfp.util.TransformedVariable(
        initial_value=1.0,
        bijector=constraint_positive,
        dtype=np.float64
    )

    def create_kernel(str_cov):
        if str_cov in ('eq', 'se'):
            kernel_main = tfp.math.psd_kernels.ExponentiatedQuadratic(
                amplitude=var_amplitude, length_scale=None)
        elif str_cov == 'matern32':
            kernel_main = tfp.math.psd_kernels.MaternThreeHalves(
                amplitude=var_amplitude, length_scale=None)
        elif str_cov == 'matern52':
            kernel_main = tfp.math.psd_kernels.MaternFiveHalves(
                amplitude=var_amplitude, length_scale=None)
        else:
            raise NotImplementedError('allowed str_cov conditions, but it is not implemented.')

        kernel = tfp.math.psd_kernels.FeatureScaled(
            kernel_main,
            var_length_scale
        )

        return kernel

    model_gp = tfp.distributions.GaussianProcess(
        kernel=create_kernel(str_cov),
        index_points=X_train,
        observation_noise_variance=var_observation_noise_variance,
        mean_fn=prior_mu
    )

    @tf.function()
    def log_prob_outputs(): # pragma: no cover
        return model_gp.log_prob(np.ravel(Y_train))

    optimizer = tf.optimizers.Adam(learning_rate=1e-2)
    trainable_variables = [
        var_.trainable_variables[0] for var_ in [
            var_amplitude,
            var_length_scale,
            var_observation_noise_variance
        ]
    ]

    list_neg_log_likelihoods = []
    ind_iter = 0

    while num_iters >= 10:
        with tf.GradientTape() as tape:
            loss = -1.0 * log_prob_outputs()

        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        list_neg_log_likelihoods.append(loss)

        if ind_iter > num_iters and np.abs(np.mean(list_neg_log_likelihoods[-6:-1]) - loss) < 5e-2:
            break
        if ind_iter > 10 * num_iters: # pragma: no cover
            break

        ind_iter += 1

    hyps = {
        'signal': tf.convert_to_tensor(var_amplitude).numpy(),
        'lengthscales': tf.convert_to_tensor(var_length_scale).numpy(),
        'noise': np.sqrt(tf.convert_to_tensor(var_observation_noise_variance).numpy())
#        'signal': var_amplitude._value().numpy(),
#        'lengthscales': var_length_scale._value().numpy(),
#        'noise': np.sqrt(var_observation_noise_variance._value().numpy())
    }

    cov_X_X, inv_cov_X_X, _ = gp_common.get_kernel_inverse(X_train, hyps,
        str_cov, fix_noise=fix_noise, debug=debug)

    time_end = time.time()

    if debug:
        logger.debug('iterations to be converged: %d', ind_iter)
        logger.debug('hyps optimized: %s', utils_logger.get_str_hyps(hyps))
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)

    return cov_X_X, inv_cov_X_X, hyps
