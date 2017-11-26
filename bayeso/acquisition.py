import numpy as np
import tensorflow as tf

def pi(pred_mean, pred_std, Y_train):
#    pred_first = pred_mean.shape[0]
    normal = tf.contrib.distributions.Normal(tf.constant(0.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64))
    normal = tf.contrib.distributions.Normal(pred_mean, pred_std)
    return normal.cdf((np.min(Y_train) - pred_mean) / pred_std)

def ei(pred_mean, pred_std, Y_train):
#    normal = tf.contrib.distributions.Normal(pred_mean, pred_std)
    pred_first = pred_mean.shape[0]
    normal = tf.contrib.distributions.Normal(tf.constant(0.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64))
    val_z = (np.min(Y_train) - pred_mean) / pred_std
    return (np.min(Y_train) - pred_mean) * normal.cdf(val_z) + pred_std * normal.prob(val_z)

def ucb(pred_mean, pred_std):
    kappa = 2.0
    return tf.add(-pred_mean, kappa * pred_std)


if __name__ == '__main__':
    import gp
    import matplotlib.pyplot as plt
    X_train = np.array([[-2.0], [0.0], [0.2], [0.1], [0.05], [0.15], [1.0], [1.5], [2.05], [1.9], [2.0], [2.1], [3.0], [-1.0]])
    Y_train = np.sin(X_train)
    X_test = np.linspace(-3, 3, 100)
    X_test = X_test.reshape((100, 1))
    pred_mean, pred_std = gp.predict_test(X_train, Y_train, X_test)
    res_pi = pi(pred_mean, pred_std, Y_train)
    res_ei = ei(pred_mean, pred_std, Y_train)
    res_ucb = ucb(pred_mean, pred_std)
    sess = tf.Session()
    val_pi = sess.run(res_pi)
    val_ei = sess.run(res_ei)
    val_ucb = sess.run(res_ucb)

    plt.plot(X_train, Y_train, 'o')
    plt.plot(X_test, pred_mean)
    plt.fill_between(X_test.flatten(), pred_mean.flatten() - 2*pred_std.flatten(), pred_mean.flatten() + 2*pred_std.flatten(), color='blue', alpha=0.2)
    plt.plot(X_test, val_pi)
    plt.plot(X_test, val_ei)
    plt.plot(X_test, val_ucb)
    plt.show()
    

