#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 21, 2021
#
"""It defines thompson sampling, which is employed
to determine where next to evaluate."""

import numpy as np

from bayeso import bo
from bayeso import constants
from bayeso.gp import gp
from bayeso.utils import utils_bo
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('thompson_sampling')


@utils_common.validate_types
def thompson_sampling_gp_iteration(range_X: np.ndarray,
    X: np.ndarray, Y: np.ndarray,
    normalize_Y: bool=constants.NORMALIZE_RESPONSE,
    str_sampling_method: str='sobol',
    num_samples: int=200,
    debug: bool=False,
) -> np.ndarray:
    """
    It chooses the next query point via Thompson sampling.

    :param range_X: bounds for a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Y: outputs. Shape: (n, 1).
    :type Y: numpy.ndarray
    :param normalize_Y: flag for normalizing responses.
    :type normalize_Y: bool., optional
    :param str_sampling_method: the name of sampling method.
    :type str_sampling_method: str., optional
    :param num_samples: the number of samples.
    :type num_samples: int., optional
    :param debug: flag for a debug option.
    :type debug: bool., optional

    :returns: the next point. Shape: (d, ).
    :rtype: numpy.ndarray

    :raises: AssertionError, ValueError

    """

    assert isinstance(range_X, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(normalize_Y, bool)
    assert isinstance(str_sampling_method, str)
    assert isinstance(num_samples, int)
    assert isinstance(debug, bool)
    assert len(range_X.shape) == 2
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert range_X.shape[0] == X.shape[1]
    assert range_X.shape[1] == 2

    str_cov = 'matern52'
    prior_mu = None
    str_optimizer_method_gp = 'BFGS'
#    use_ard = True

    if normalize_Y:
        if debug:
            logger.debug('Responses are normalized.')

        Y = utils_bo.normalize_min_max(Y)

    model_bo = bo.BOwGP(range_X)
    X_test = model_bo.get_samples(str_sampling_method, num_samples=num_samples)

    mu_Xs, _, Sigma_Xs = gp.predict_with_optimized_hyps(X, Y, X_test,
        str_cov=str_cov, str_optimizer_method=str_optimizer_method_gp,
        prior_mu=prior_mu, debug=debug)
    mu_Xs = np.squeeze(mu_Xs, axis=1)

    Y_sampled = None
    list_jitters = [0.0, 1e-4, 1e-2, 1e0, 1e1, 1e2, 1e3, 1e4]

    for jitter_cov in list_jitters:
        try:
            Sigma_Xs_ = Sigma_Xs + jitter_cov * np.eye(Sigma_Xs.shape[0])
            Y_sampled = gp.sample_functions(mu_Xs, Sigma_Xs_, num_samples=1)

            break
        except ValueError: # pragma: no cover
            pass

    if Y_sampled is None: # pragma: no cover
        raise ValueError('jitter_cov is not large enough.')

    ind_min = np.argmin(Y_sampled[:, 0])
    next_point = X_test[ind_min]

    return next_point

def thompson_sampling_gp(range_X: np.ndarray,
    fun_target: callable,
    num_init: int, num_iter: int,
    normalize_Y: bool=constants.NORMALIZE_RESPONSE,
    str_sampling_method: str='uniform',
    num_samples: int=200,
    debug: bool=False,
) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    """
    It chooses `num_iter` query points via Thompson sampling
    with `num_init` initial points.

    :param range_X: bounds for a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param fun_target: target function.
    :type fun_target: typing.Callable
    :param num_init: the number of initial points.
    :type num_init: int.
    :param num_iter: the number of iterations.
    :type num_iter: int.
    :param normalize_Y: flag for normalizing responses.
    :type normalize_Y: bool., optional
    :param str_sampling_method: the name of sampling method.
    :type str_sampling_method: str., optional
    :param num_samples: the number of samples.
    :type num_samples: int., optional
    :param debug: flag for a debug option.
    :type debug: bool., optional

    :returns: a tuple of query points and their evaluations.
        Shape: (`num_init` + `num_iter`, d), (`num_init` + `num_iter`, 1).
    :rtype: (numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(range_X, np.ndarray)
    assert callable(fun_target)
    assert isinstance(num_init, int)
    assert isinstance(num_iter, int)
    assert isinstance(normalize_Y, bool)
    assert isinstance(str_sampling_method, str)
    assert isinstance(num_samples, int)
    assert isinstance(debug, bool)
    assert len(range_X.shape) == 2
    assert range_X.shape[1] == 2

    model_bo = bo.BOwGP(range_X)
    X = model_bo.get_initials(str_sampling_method, num_init)
    Y = []

    for bx in X:
        Y.append(fun_target(bx))
    Y = np.reshape(Y, (X.shape[0], 1))

    for ind_iter in range(0, num_iter):
        print(f'{ind_iter+1} iteration')

        next_point = thompson_sampling_gp_iteration(range_X, X, Y,
            normalize_Y, str_sampling_method, num_samples, debug)

        X = np.concatenate((X, [next_point]), axis=0)
        Y = np.vstack((Y, fun_target(next_point)))

    return X, Y
