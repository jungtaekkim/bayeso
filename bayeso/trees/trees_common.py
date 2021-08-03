import numpy as np


def get_inputs_from_leaf(leaf):
    assert isinstance(leaf, list)

    X = [bx for bx, by in leaf]
    return np.array(X)

def get_outputs_from_leaf(leaf):
    assert isinstance(leaf, list)

    Y = [by for bx, by in leaf]
    return np.array(Y)

def _mse(Y):
    if Y.shape[0] > 0:
        mean = np.mean(Y, axis=0)
        mse_ = np.mean((Y - mean)**2)
    else:
        mse_ = 1e8
    return mse_

def mse(left_right):
    assert isinstance(left_right, tuple)

    left, right = left_right

    Y_left = get_outputs_from_leaf(left)
    Y_right = get_outputs_from_leaf(right)

    mse_left = _mse(Y_left)
    mse_right = _mse(Y_right)

    return mse_left + mse_right

def subsample(X, Y, ratio_sample, replace_samples):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(ratio_sample, float)
    assert isinstance(replace_samples, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    num_X = X.shape[0]
    num_samples = int(num_X * ratio_sample)

    indices = np.random.choice(num_X, num_samples, replace=replace_samples)

    X_ = X[indices]
    Y_ = Y[indices]

    assert X_.shape[0] == Y_.shape[0] == num_samples
    return X_, Y_

def _split_left_right(X, Y, dim_to_split, val_to_split):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(dim_to_split, int)
    assert isinstance(val_to_split, float)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    left = []
    right = []

    for bx, by in zip(X, Y):
        if bx[dim_to_split] < val_to_split:
            left.append((bx, by))
        else:
            right.append((bx, by))
    return left, right

def _split(X, Y, num_features, split_random_location):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    # TODO: when X.shape[0] < 2, it needs to be handled.

    cur_ind = np.inf
    cur_val = np.inf
    cur_score = np.inf
    cur_left_right = None

    features = np.random.choice(X.shape[1], num_features, replace=False)

    for ind in features:
        dim_to_split = int(ind)

        if X.shape[0] > 1 and split_random_location:
            min_bx = np.min(X[:, dim_to_split])
            max_bx = np.max(X[:, dim_to_split])

        for bx, by in zip(X, Y):
            if X.shape[0] > 1 and split_random_location:
                val_to_split = np.random.uniform(low=min_bx, high=max_bx)
            else:
                val_to_split = bx[dim_to_split]


            left_right = _split_left_right(X, Y, dim_to_split, val_to_split)
            score = mse(left_right)

            if score < cur_score:
                cur_ind = dim_to_split
                cur_val = val_to_split
                cur_score = score
                cur_left_right = left_right

    return {'index': cur_ind, 'value': cur_val, 'left_right': cur_left_right}

def split(node, depth_max, size_min_leaf, num_features, split_random_location, cur_depth):
    assert isinstance(node, dict)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(num_features, int)
    assert isinstance(cur_depth, int)
    assert isinstance(split_random_location, bool)

    left, right = node['left_right']
    del(node['left_right'])

    if not left or not right:
        node['left'] = node['right'] = left + right
        return

    if cur_depth >= depth_max:
        node['left'], node['right'] = left, right
        return

    if len(left) <= size_min_leaf:
        node['left'] = left
    else:
        node['left'] = _split(get_inputs_from_leaf(left), get_outputs_from_leaf(left), num_features, split_random_location)
        split(node['left'], depth_max, size_min_leaf, num_features, split_random_location, cur_depth + 1)

    if len(right) <= size_min_leaf:
        node['right'] = right
    else:
        node['right'] = _split(get_inputs_from_leaf(right), get_outputs_from_leaf(right), num_features, split_random_location)
        split(node['right'], depth_max, size_min_leaf, num_features, split_random_location, cur_depth + 1)

def _predict_by_tree(bx, tree):
    assert isinstance(bx, np.ndarray)
    assert isinstance(tree, dict)

    assert len(bx.shape) == 1

    if bx[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return _predict_by_tree(bx, tree['left'])
        else:
            cur_Y = get_outputs_from_leaf(tree['left'])
            return np.mean(cur_Y), np.std(cur_Y)
    else:
        if isinstance(tree['right'], dict):
            return _predict_by_tree(bx, tree['right'])
        else:
            cur_Y = get_outputs_from_leaf(tree['right'])
            return np.mean(cur_Y), np.std(cur_Y)

def _predict_by_trees(bx, list_trees):
    assert isinstance(bx, np.ndarray)
    assert isinstance(list_trees, list)

    assert len(bx.shape) == 1

    list_mu_leaf = []
    list_sigma_leaf = []

    for tree in list_trees:
        mu_leaf, sigma_leaf = _predict_by_tree(bx, tree)

        list_mu_leaf.append(mu_leaf)
        list_sigma_leaf.append(sigma_leaf)

    mu = np.mean(list_mu_leaf)
    sigma = compute_sigma(np.array(list_mu_leaf), np.array(list_sigma_leaf))

    return mu, sigma

def predict_by_trees(X, list_trees):
    assert isinstance(X, np.ndarray)
    assert isinstance(list_trees, list)

    assert len(X.shape) == 2

    preds_mu = []
    preds_sigma = []

    for bx in X:
        mu_bx, sigma_bx = _predict_by_trees(bx, list_trees)

        preds_mu.append(mu_bx)
        preds_sigma.append(sigma_bx)

    preds_mu = np.array(preds_mu)[..., np.newaxis]
    preds_sigma = np.array(preds_sigma)[..., np.newaxis]

    return preds_mu, preds_sigma

def compute_sigma(preds_mu_leaf, preds_sigma_leaf, min_sigma=0.0):
    preds_sigma_leaf_ = np.maximum(preds_sigma_leaf, np.zeros(preds_sigma_leaf.shape) + min_sigma)

    sigma = np.mean(preds_mu_leaf**2 + preds_sigma_leaf_**2)
    sigma -= np.mean(preds_mu_leaf)**2

    if sigma < 0.0:
        sigma = 0.0
    sigma = np.sqrt(sigma)

    return sigma
