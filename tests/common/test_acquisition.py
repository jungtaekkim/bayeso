# test_acquisition
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 14, 2020

import pytest
import numpy as np
import typing

from bayeso import acquisition


TEST_EPSILON = 1e-5


def test_pi_typing():
    annos = acquisition.pi.__annotations__

    assert annos['pred_mean'] == np.ndarray
    assert annos['pred_std'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['jitter'] == float
    assert annos['return'] == np.ndarray

def test_pi():
    with pytest.raises(AssertionError) as error:
        acquisition.pi('abc', np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), 'abc', np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

    val_acq = acquisition.pi(np.arange(0, 10), np.ones(10), np.zeros((5, 1)))
    truth_val_acq = np.array([5.00000000e-01, 1.58657674e-01, 2.27512118e-02, 1.35003099e-03, 3.16765954e-05, 2.86725916e-07, 9.86952260e-10, 1.28045212e-12, 6.22500364e-16, 1.12951395e-19])
    print(val_acq)

    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

def test_ei_typing():
    annos = acquisition.ei.__annotations__

    assert annos['pred_mean'] == np.ndarray
    assert annos['pred_std'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['jitter'] == float
    assert annos['return'] == np.ndarray

def test_ei():
    with pytest.raises(AssertionError) as error:
        acquisition.ei('abc', np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), 'abc', np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

    val_acq = acquisition.ei(np.arange(0, 10), np.ones(10), np.zeros((5, 1)))
    truth_val_acq = np.array([3.98942280e-01, 8.33154706e-02, 8.49070261e-03, 3.82154315e-04, 7.14525833e-06, 5.34616535e-08, 1.56356969e-10, 1.76032579e-13, 7.55026079e-17, 1.22477876e-20])
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

def test_ucb_typing():
    annos = acquisition.ucb.__annotations__

    assert annos['pred_mean'] == np.ndarray
    assert annos['pred_std'] == np.ndarray
    assert annos['Y_train'] == typing.Union[np.ndarray, type(None)]
    assert annos['kappa'] == float
    assert annos['increase_kappa'] == bool
    assert annos['return'] == np.ndarray

def test_ucb():
    with pytest.raises(AssertionError) as error:
        acquisition.ucb('abc', np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), kappa='abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(5), np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), Y_train='abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), Y_train=np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones((10, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones((10, 1)), np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), increase_kappa='abc')

    val_acq = acquisition.ucb(np.arange(0, 10), np.ones(10), Y_train=np.zeros((5, 1)))
    truth_val_acq = np.array([3.21887582, 2.21887582, 1.21887582, 0.21887582, -0.78112418, -1.78112418, -2.78112418, -3.78112418, -4.78112418, -5.78112418])
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

    val_acq = acquisition.ucb(np.arange(0, 10), np.ones(10))
    truth_val_acq = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

def test_aei_typing():
    annos = acquisition.aei.__annotations__

    assert annos['pred_mean'] == np.ndarray
    assert annos['pred_std'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['jitter'] == float
    assert annos['return'] == np.ndarray

def test_aei():
    with pytest.raises(AssertionError) as error:
        acquisition.aei('abc', np.ones(10), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), 'abc', np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), np.ones(10), 'abc', 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), np.ones(10), np.zeros((5, 1)), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), np.ones(10), np.zeros((5, 1)), 1.0, jitter=1)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(5), np.ones(10), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), np.ones(10), np.zeros(5), 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        acquisition.aei(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)), 1.0)

    val_acq = acquisition.aei(np.arange(0, 10), np.ones(10), np.zeros((5, 1)), 1.0)
    truth_val_acq = np.array([1.16847489e-01, 2.44025364e-02, 2.48686922e-03, 1.11930407e-04, 2.09279771e-06, 1.56585558e-08, 4.57958958e-11, 5.15587486e-14, 2.21142019e-17, 3.58729395e-21])
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

def test_pure_exploit_typing():
    annos = acquisition.pure_exploit.__annotations__

    assert annos['pred_mean'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_pure_exploit():
    with pytest.raises(AssertionError) as error:
        acquisition.pure_exploit('abc')
    with pytest.raises(AssertionError) as error:
        acquisition.pure_exploit(np.ones((10, 1)))

    val_acq = acquisition.pure_exploit(np.arange(0, 10))
    truth_val_acq = -np.arange(0, 10)
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()

def test_pure_explore_typing():
    annos = acquisition.pure_explore.__annotations__

    assert annos['pred_std'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_pure_explore():
    with pytest.raises(AssertionError) as error:
        acquisition.pure_explore('abc')
    with pytest.raises(AssertionError) as error:
        acquisition.pure_explore(np.ones((10, 1)))

    val_acq = acquisition.pure_explore(np.arange(0, 10))
    truth_val_acq = np.arange(0, 10)
    assert (np.abs(val_acq - truth_val_acq) < TEST_EPSILON).all()
