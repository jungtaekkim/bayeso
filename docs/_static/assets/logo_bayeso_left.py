import bayeso
from bayeso import gp
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X_test = np.expand_dims(np.linspace(0, 1, 100), axis=1)
    X_train = np.array([
        [2.],
        [3.],
        [5.],
        [7.],
        [9.],
    ]) / 10.0
    Y_train = np.array([
        [-0.2],
        [-0.25],
        [0.05],
        [-0.1],
        [0.1],
    ]) * 1.0

    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov='matern52')
    mu = np.squeeze(mu)
    sigma = np.squeeze(sigma)

    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca()

    ax.plot(X_test.flatten(), mu, color=(255./255., 179./255., 0./255.), linewidth=8)

    '''
    range_shade = 4.0
    ax.fill_between(
        X_test.flatten(),
        mu - range_shade * sigma,
        mu + range_shade * sigma,
        color=(102./255., 102./255., 92./255.),
        alpha=0.3
    )
    '''
    ax.plot([0.254], [-0.235], color=(200./255., 1./255., 80./255.), linestyle='none', marker='+', markersize=18, markeredgewidth=6)

    plt.axis('off')

    plt.savefig(
        './logo_bayeso_left.pdf',
        format='pdf',
        transparent=True,
        bbox_inches='tight',
    )
    plt.show()

