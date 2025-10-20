import transforms as tfs
import tools
import numpy as np
import tensorflow as tf
import datasets as ds
import tensorflow_probability as tfp
import os
import sys
import scipy
from numpy import linalg as la
from argparse import Namespace
from tqdm import tqdm
from pyentropy import systems
from random import sample


def dmd(X, tau=1, mean_center=True):
    if mean_center:
        X1 = X[:,:-tau] - np.mean(X[:,:-tau], axis=1, keepdims=True)
        X2 = X[:,tau:] - np.mean(X[:,tau:], axis=1, keepdims=True)
    else:
        X1 = X[:,:-tau]
        X2 = X[:,tau:]
    U = X2 @ X1.T @ la.inv(X1 @ X1.T)
    lam, V = la.eig(U)
    sort_modes = np.argsort(-np.abs(lam))
    lam = lam[sort_modes]
    V = V[:,sort_modes]
    return lam, V


def autocorrelation(X, dt=0):
    if len(X.shape) == 1:
        X = X[None,:]
    N, P = X.shape
    T = P - dt
    t1 = np.arange(T)
    t2 = np.arange(T) + dt
    m1 = np.sum(X[:,t1], axis=1) / (T - 1)
    m2 = np.sum(X[:,t2], axis=1) / (T - 1)
    m12 = np.sum(X[:,t1] * X[:,t2], axis=1) / (T - 1)
    return m12 - m1 * m2
    

def map_clustering(X, dim=1, return_residuals=False):
    """
    MAP = maximal alignment projections.

    Made this up to figure out how the meta neuron filters are composed but may just be inferior to PCA.
    """
    N, P = X.shape
    indices = np.zeros(dim, dtype=int)
    basis_vectors = np.zeros((N, dim))
    Y = X.copy()
    for n in range(dim):
        overlaps = Y.T @ Y
        i = np.argsort(-np.sum(overlaps**2, axis=0))[0]
        vec = Y[:, i]
        q = vec @ Y
        Y = Y - q[None, :] * vec[:, None]
        indices[n] = i
        basis_vectors[:, n] = vec

    if return_residuals:
        return indices, basis_vectors, Y

    return indices, basis_vectors


def validate_joint_input(X, Y):
    N, P = X.shape
    M, PY = Y.shape
    if P != PY:
        raise ValueError(
            f"X and Y must have same number of samples but have {P} and {PY} respectively"
        )
    return N, M, P

def equal_bin_edges(X, nbins):
    N, P = X.shape
    bin_edges = list()
    for n in range(N):
        bin_edges.append(tools.equal_bin_edges(X[n, :], nbins))
    return bin_edges


def pca(X):
    sig = np.cov(X)
    lam, V = la.eigh(sig)
    return lam[::-1], V[:, ::-1]


def cca(X, Y):
    """
    Perform Canonical Correlation Analysis (CCA) on input arrays X and Y.

    Args:
        X (ndarray): Input array X with shape (N, P), where N is the number of features and P is the number of samples.
        Y (ndarray): Input array Y with shape (M, P), where M is the number of features and P is the number of samples.

    Returns:
        ndarray: Eigenvalues of the matrix \Sigma_X^{-1} \Sigma_{X|Y}, representing the correlation strengths between the canonical variates of X and Y.
        ndarray: Eigenvectors of the matrix \Sigma_X^{-1} \Sigma_{X|Y}, representing the canonical weights for X.

    Raises:
        ValueError: If the number of features in X and Y does not match.

    Notes:
        - CCA finds linear combinations of X and Y that are maximally correlated.
        - This implementation performs CCA in an asymmetric manner.
        - Eigenvalues in the range [0, 1] represent how significant the correlation is between the corresponding X mode and Y.
        - Eigenvalues closer to 0 indicate modes of X that are highly correlated with Y.
    """
    N, M, P = validate_joint_input(X, Y)

    XY = np.concatenate((X, Y))
    sig = np.cov(XY)
    sig_XX = sig[:N, :N]
    sig_YY = sig[N:, N:]
    sig_XY = sig[:N, N:]
    sig_X_Y = sig_XX - sig_XY @ la.pinv(sig_YY) @ sig_XY.T
    K = la.pinv(sig_XX) @ sig_X_Y
    lam_x, V_x = la.eig(K)
    sort_inds = np.argsort(lam_x)
    return lam_x[sort_inds], V_x[:, sort_inds]


def multi_outer(X, order=1):
    """
    Want out[i1,i2,...,in,a] = X[i1,a] * X[i2,a] * ... * X[in,a]
    """
    N, P = np.shape(X)
    if order > 0:
        cur_shape = [N] + [1] * (order - 1) + [P]
        X_cur = X.reshape(cur_shape)
        return X_cur * multi_outer(X, order - 1)
    else:
        return 1


def nth_moment(X, order, approx_buffer_size=10e6):
    """"""
    N, P = np.shape(X)

    # preallocate output array
    output_shape = [N] * order
    M = np.zeros(output_shape)

    # determine number of buffer steps needed
    total_operations = N**order * P
    steps = int(total_operations / approx_buffer_size) + 1
    buffer_size = int(P / steps)

    # step through buffer steps
    for step in range(steps):
        X_cur = X[:, step * buffer_size : (step + 1) * buffer_size]
        M_cur = np.sum(multi_outer(X_cur, order), axis=-1)
        M += M_cur

    # get the remaining elements
    if buffer_size * steps < P:
        X_cur = X[:, steps * buffer_size :]
        M_cur = np.sum(multi_outer(X_cur, order), axis=-1)
        M += M_cur  # compute moment for remaining

    M = M / P
    return M


def normalized_nth_moment(X, order, center=True):
    N, P = X.shape
    if center:
        X = X - np.mean(X, axis=1)[:, None]
    mn = np.sum(X**order, axis=1) / P
    m2 = np.sum(X**2, axis=1) / P
    return mn / m2 ** int(order / 2)


def PCA_scaling(X, cutoffs, order=4):
    N, P = X.shape
    excess = np.zeros((len(cutoffs), N))
    _, V = pca(X)
    for i, k in enumerate(cutoffs):
        y_cur = V[:, :k] @ V.T[:k, :] @ X
        excess[i, :] = normalized_nth_moment(y_cur, order)
    return excess


def nth_cumulant(X, order):
    pass


def nth_moment_elementwise(X, order, indices):
    pass


def linear_extrapolation_estimate_joint(
    X, Y, analysis_function, fractions=[1], N_subsamples=1, quiet=True
):
    N, M, P = validate_joint_input(X, Y)
    N_fractions = len(fractions)
    fractions = np.array(fractions)

    individual_trial_outputs = np.zeros((N_fractions, N_subsamples))
    for n, fraction in enumerate(fractions):
        individual_trial_outputs[n, :] = bootstrap_joint(
            X, Y, analysis_function, N_subsamples, fraction, quiet
        )

    L = linear_regress(1 / fractions, individual_trial_outputs)
    mean_estimate = L["fit"](0)
    reference_fraction = fractions[0]  # chosen to be 0.5 in PNAS 2015 paper
    stdev_estimate = np.sqrt(
        np.cov(individual_trial_outputs[0, :]) * reference_fraction
        + L["intercept_err"] ** 2
    )

    return mean_estimate, stdev_estimate


def linear_extrapolation_estimate(
    X, analysis_function, fractions=[0.5], N_subsamples=50, quiet=True
):
    N_fractions = len(fractions)
    fractions = np.array(fractions)

    individual_trial_outputs = np.zeros((N_fractions, N_subsamples))
    for n, fraction in enumerate(fractions):
        individual_trial_outputs[n, :] = bootstrap(
            X, analysis_function, N_subsamples, fraction, quiet
        )

    L = linear_regress(1 / fractions, individual_trial_outputs)
    mean_estimate = L["fit"](0)
    reference_fraction = fractions[0]  # chosen to be 0.5 in PNAS 2015 paper

    # TODO: add in linear regression error to this stdev estimate,
    # Not sure if the approach below is correct.
    # I think we should actually use an estimate of the intercept variance here instead of the confidence interval.
    stdev_estimate = np.sqrt(
        np.cov(individual_trial_outputs[0, :]) * reference_fraction
        + L["intercept_err"] ** 2
    )
    return mean_estimate, stdev_estimate


def bootstrap_joint(X, Y, analysis, N_subsamples=1, subsample_fraction=1, quiet=True):
    N, M, P = validate_joint_input(X, Y)
    total_data_poicleaningnts = P
    output_array = np.zeros((N_subsamples,))

    if quiet:
        counter = range(N_subsamples)
    else:
        counter = tqdm(range(N_subsamples))

    for n in counter:
        subsample_inds = sample(range(P), k=int(subsample_fraction * P))
        subsample_X = X[:, subsample_inds]
        subsample_Y = Y[:, subsample_inds]
        output_array[n] = analysis(subsample_X, subsample_Y)

    return output_array


def bootstrap(X, analysis, N_subsamples=1, subsample_fraction=1, quiet=True):
    """
    User needs to provide an analysis function `analysis` which takes in a single array `X` in standard format.

    """
    N, P = X.shape
    total_data_points = P
    output_array = np.zeros((N_subsamples,))

    if quiet:
        counter = range(N_subsamples)
    else:
        counter = tqdm(range(N_subsamples))

    for n in counter:
        subsample_inds = sample(range(P), k=int(subsample_fraction * P))
        subsample = X[:, subsample_inds]
        output_array[n] = analysis(subsample)

    return output_array


def plugin_MI(X, Y, log_base=np.e):
    N, M, P = validate_joint_input(X, Y)
    X = X.astype(int)
    Y = Y.astype(int)
    number_of_features_X = N
    number_of_unique_states_X = len(np.unique(np.ravel(X)))
    number_of_features_Y = M
    number_of_unique_states_Y = len(np.unique(np.ravel(Y)))
    s = systems.DiscreteSystem(
        X,
        (number_of_features_X, number_of_unique_states_X),
        Y,
        (number_of_features_Y, number_of_unique_states_Y),
    )  # for this package, it is necessary to have a joint distribution dataset.
    s.calculate_entropies("plugin", "naive", ["HX", "HY", "HXY"])
    mutual_information_in_bits = s.I()
    return mutual_information_in_bits * np.log(2) / np.log(log_base)


def plugin_entropy(X, log_base=np.e):
    """
    This is just a wrapper for the pyentropy package.
    """
    N, P = X.shape
    number_of_features = N
    number_of_unique_states = len(np.unique(np.ravel(X)))
    s = systems.DiscreteSystem(
        X,
        (number_of_features, number_of_unique_states),
        X,
        (number_of_features, number_of_unique_states),
    )  # for this package, it is necessary to have a joint distribution dataset.
    s.calculate_entropies("plugin", "naive", "HX")
    entropy_in_bits = s.H["HX"]
    return entropy_in_bits * np.log(2) / np.log(log_base)


def cae(X, log_base=np.e):
    """
    Returns the CAE entropy estimate for X.

    X states need to be discrete, but do not need to have been labelled yet. For example, X can be neural codewords.

    Arguments:
    X (ndarray) -- Data array. Should have shape (N, P) where N is the number of features and P is the number of points
    log_base (float) -- Base of the logarithm. Defaults to natural logarithm.
    """
    _, counts = np.unique(
        X, axis=1, return_counts=True
    )  # first returned argument is the words, we don't need that.
    H = tools.cae_entropy(counts) / np.log(log_base)
    return H


def linear_regress(x, Y, Y_err=None):
    """
    error estimation: https://online.stat.psu.edu/stat415/lesson/7/7.5

    Error estimates are 95% confidence intervals.

    Update 5.13.24: I added a missing factor of N in the error estimates. See https://en.wikipedia.org/wiki/Simple_linear_regression 
    """
    if len(x.shape) != 1:
        raise TypeError(
            f"Independent variable data x has shape {x.shape} but needs to be 1-dimensional."
        )

    if Y_err is None:
        Y_err = np.zeros(Y.shape)
    elif Y_err.shape != Y.shape:
        raise TypeError(
            f"`Y` and `Y_err` need to have the same shape but have shapes {Y.shape} and {Y_err.shape}"
        )

    # If one trial only,
    if len(Y.shape) == 1:
        # return linear_regress_single(x, Y, Y_err=Y_err)
        Y = Y[:, None]

    # Now we expand x and Y_mean to have as many total entries as Y, then unfold both.
    # Y_mean = np.average(Y, axis=1)
    x_mean = np.average(x)
    # Y_mean = Y_mean[:,None] * np.ones(Y.shape)
    x = x[:, None] * np.ones(Y.shape)
    N = np.size(Y)
    x_flat = x.reshape((N,))
    Y_flat = Y.reshape((N,))
    # Y_mean_flat = Y_mean.reshape((N,))
    Y_err_flat = Y_err.reshape((N,))

    # perform linear regression, accounting for heteroscedasticity if possible
    if (Y_err == 0).any():
        weights = np.ones((N,))
    else:
        weights = 1 / Y_err_flat
    params = np.polynomial.polynomial.polyfit(x_flat, Y_flat, deg=1, w=weights)
    intercept, slope = tuple(params)
    fit = np.polynomial.polynomial.Polynomial(params)
    residuals = Y_flat - fit(x_flat)

    # compute confidence intervals for regression parameters
    MSE = np.sum(residuals**2) / (N - 2)
    precision_error = np.sum(Y_err**2) / N
    alpha = 0.05
    confidence = 1 - alpha / 2
    dof = N - 2
    interval = scipy.stats.t.interval(confidence, dof)
    intercept_err = interval[1] * np.sqrt((MSE + precision_error) / N)
    slope_err = interval[1] * np.sqrt((MSE + precision_error) / np.sum((x_flat - x_mean) ** 2)) 

    # Make a fit function for the errors
    # fit = np.polynomial.polynomial.Polynomial(params)
    fit_err = np.polynomial.polynomial.Polynomial(np.array([intercept_err, slope_err]))

    return_vars = ["fit", "fit_err", "intercept", "slope", "intercept_err", "slope_err"]
    cur_locals = locals()
    return {k: cur_locals[k] for k in return_vars}


def IB(X, Y, coarse_grainer, critic_function, **opt_params):
    """
    An implementation of IB-like coarse-graining based on the RSMI-NE project.

    The convention for the critic_function inputs is critic_function(y, t) where t is coarse_grainer(x)

    Arguments:
    X (ndarray or Tensor) -- "Input" variable of IB (will be coarse-grained). Shape is (N, P) with interpretation (features, samples)
    Y (ndarray or Tensor) -- "Relevance" variable of IB. Shape is (M, P) with interpretation (features, samples)
    coarse_grainer (tensorflow Layer or Model) -- Coarse-graining map for X
    critic_function (tensorflow Layer or Model) -- functional form describing log(P(y,t)) - log(P(y)P(t)).
    """
    N, M, P = validate_joint_input(X, Y)

    opt_params_default = {
        "batch_size": 800,  # was 800 for larger buffers
        "iterations": 1,  # was 400 for above BKT, 25 for below BKT
        "shuffle": 100000,
        "learning_rate": 9e-3,  # was 4e-3
        "quiet": False,  # turn off updates during training
        "track_quantities": False,  # makes a pretty big difference in speed
        "use_GPU": False,
        "optimizer": None,  # options in tf.keras.optimizers. Default is Adam
        # 'bound': 'infonce', # currently doesn't do anything, only infonce is available
    }
    opt_params_default.update(opt_params)
    params = opt_params_default.copy()

    # convert X and Y to tensorflow arrays. Make into tensorflow datasets.
    X_tf = tf.cast(X.T, dtype=tf.as_dtype(X.dtype))
    Y_tf = tf.cast(Y.T, dtype=tf.as_dtype(Y.dtype))
    dat = tf.data.Dataset.from_tensor_slices((X_tf, Y_tf))
    dat = (
        dat.shuffle(params["shuffle"])
        .batch(params["batch_size"])
        .repeat(params["iterations"])
    )

    # bound = params['bound']
    objective_function = infonce_lower_bound

    # set optimizer
    if params["optimizer"] == None:
        # set optimiser as adam with given learning rate
        # opt = tf.keras.optimizers.Adam(params["learning_rate"])
        opt = tf.keras.optimizers.legacy.Adam(params["learning_rate"])
    else:
        opt = params["optimizer"]

    @tf.function
    def train_step(x, y):
        """Single training step: performs gradient descent
        on the coarse-graining network and vbmi net simultaneously.
        Returns the most recent value of the RSMI estimate and the
        corresponding set of coarse-grained random variables H.

        Keyword arguments:
        x, y -- samples for random variables E and V, respectively.
        """

        with tf.GradientTape() as tape:
            h = coarse_grainer(x)

            if params["use_GPU"]:
                with tf.device("/GPU:" + str(0)):
                    mi = objective_function(y, h, critic_function, training=True)
            else:
                mi = objective_function(y, h, critic_function, training=True)
            loss = -mi

            trainable_vars = []
            # train VBMI critic and coarse-graining filters simulatenously
            trainable_vars += coarse_grainer.trainable_variables
            trainable_vars += critic_function.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            opt.apply_gradients(zip(grads, trainable_vars))
        return mi, h

    mi_during_training = []
    coarse_grainer_weights = []
    critic_weights = []
    printing = not params["quiet"]
    track_quantities = params["track_quantities"]
    if printing:
        pbar = tqdm(total=len(dat), desc="")

    for V, E in dat:
        mi, _ = train_step(V, E)
        if track_quantities:
            mi_during_training.append(mi.numpy())
            coarse_grainer_weights.append(coarse_grainer.get_weights())
            critic_weights.append(critic_function.get_weights())

        if printing:
            pbar.set_description(f"I={mi:.2f} nats")
            pbar.update(1)  # update progress bar for each iteration step

        # # TODO: Update coarse-grainer layers to include normalized option to replace omitted code below
        # # # update weights in coarsegrainer
        # if coarse_grainer_params['normalize_weights']:
        #     temp_filter = coarse_grainer.coarse_grainer.get_weights()[0]  # necessary to use get_weights and not coarse_grainer.coarse_grainer.ws because FixableUnstructured1D requires non-trainable variables. See notes 8.26.23
        #     filters.append(temp_filter / (np.sqrt(np.sum(temp_filter ** 2, axis=0)) + 1e-15))
        # filters.append(coarse_grainer.coarse_grainer.get_weights()[0])
        # train coarse-graining filters and vbmi critic parameters simultaneously

    if printing:
        print("Training complete.")
        pbar.close()

    # return mi, filters, coarse_grainer, critic_function
    # return mi, filters
    if track_quantities:
        return mi_during_training, coarse_grainer_weights, critic_weights


def MI_NCE(
    X, Y, critic_function, shuffle=True, batch_size=100, N_batches=50, quiet=False
):
    """
    Note that in RSMI X is supposed to be the coarse-grained representation H and Y is supposed to be the environment E
    """

    N, M, P = validate_joint_input(X, Y)

    @tf.function
    def single_batch_MI(E, H):
        return infonce_lower_bound(E, H, critic_function)

    N_repeats = max(int(N_batches * batch_size / (X.shape[1])), 1)
    shuffle_size = 2 * (X.shape[1])

    if batch_size == 0:
        if not quiet:
            print("batch size of zero")
        return None

    dat = tf.data.Dataset.from_tensor_slices((X.T, Y.T))
    if shuffle:
        dat = dat.repeat(N_repeats).shuffle(shuffle_size).batch(batch_size)
    else:
        dat = dat.batch(batch_size).repeat(N_repeats)

    mi = []
    total_iterations = len(dat)
    if not quiet:
        pbar = tqdm(total=total_iterations)

    for V, E in dat:
        mi.append(single_batch_MI(E, V))
        if not quiet:
            pbar.update(1)

    mi_array = np.array(mi)
    mi = sum(mi_array) / total_iterations

    resample_fraction = batch_size / (
        X.shape[1]
    )  # number of repeats does not affect variance
    # resample_fraction = 1
    mi_stdev = np.sqrt(sum((mi_array - mi) ** 2) / total_iterations * resample_fraction)
    if not quiet:
        pbar.close()

    output = {"mi": mi, "mi_stdev": mi_stdev, "mi_array": mi_array}
    return output


def train_NCE(X, Y, critic_function, **opt_params):
    N, M, P = validate_joint_input(X, Y)

    opt_params_default = {
        "batch_size": 800,  # was 800 for larger buffers
        "iterations": 1,  # was 400 for above BKT, 25 for below BKT
        "shuffle": 100000,
        "learning_rate": 9e-3,  # was 4e-3
        "quiet": False,  # turn off updates during training
        "track_quantities": False,  # makes a pretty big difference in speed
        "use_GPU": False,
        "optimizer": None,  # options in tf.keras.optimizers. Default is Adam
        # 'bound': 'infonce', # currently doesn't do anything, only infonce is available
    }
    opt_params_default.update(opt_params)
    params = opt_params_default.copy()

    # convert X and Y to tensorflow arrays. Make into tensorflow datasets.
    X_tf = tf.cast(X.T, dtype=tf.as_dtype(X.dtype))
    Y_tf = tf.cast(Y.T, dtype=tf.as_dtype(Y.dtype))
    dat = tf.data.Dataset.from_tensor_slices((X_tf, Y_tf))
    dat = (
        dat.shuffle(params["shuffle"])
        .batch(params["batch_size"])
        .repeat(params["iterations"])
    )

    # bound = params['bound']
    objective_function = infonce_lower_bound

    # set optimizer
    if params["optimizer"] == None:
        # set optimiser as adam with given learning rate
        # opt = tf.keras.optimizers.Adam(params["learning_rate"])
        opt = tf.keras.optimizers.legacy.Adam(params["learning_rate"])
        
    else:
        opt = params["optimizer"]

    @tf.function
    def train_step(x, y):
        """Single training step: performs gradient descent
        on the coarse-graining network and vbmi net simultaneously.
        Returns the most recent value of the RSMI estimate and the
        corresponding set of coarse-grained random variables H.

        Keyword arguments:
        x, y -- samples for random variables E and V, respectively.
        """

        with tf.GradientTape() as tape:
            if params["use_GPU"]:
                with tf.device("/GPU:" + str(0)):
                    mi = objective_function(y, x, critic_function, training=True)
            else:
                mi = objective_function(y, x, critic_function, training=True)
            loss = -mi

            trainable_vars = []
            # train VBMI critic and coarse-graining filters simulatenously
            trainable_vars += critic_function.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            opt.apply_gradients(zip(grads, trainable_vars))
        return mi

    mi_during_training = []
    critic_weights = []
    printing = not params["quiet"]
    track_quantities = params["track_quantities"]
    if printing:
        pbar = tqdm(total=len(dat), desc="")

    for V, E in dat:
        mi = train_step(V, E)
        if track_quantities:
            mi_during_training.append(mi.numpy())
            critic_weights.append(critic_function.get_weights())

        if printing:
            pbar.set_description(f"I={mi:.2f} nats")
            pbar.update(1)  # update progress bar for each iteration step

    if printing:
        print("Training complete.")
        pbar.close()

    if track_quantities:
        return mi_during_training, critic_weights


def infonce_lower_bound(X, Y, critic_function, training=False):
    """InfoNCE replica lower-bound (van den Oord et al. 2018)
    for estimating I(X:Y).

    Keyword arguments:
    x -- full sample dataset for random variable X
    y -- full sample dataset for random variable Y
    f_ansatz (str) -- type of ansatz function

    Author: Doruk Efe Gökmen
    Date: 10/01/2021
    """
    scores = critic_function(X, Y, training=training)
    num_samples = scores.shape[0]
    positive_mask = tf.eye(num_samples, dtype=bool)

    return (tfp.vi.mutual_information).lower_bound_info_nce(
        logu=scores, joint_sample_mask=positive_mask
    )


def IB_test_train(X, Y, frac=0.5):
    N, M, P = validate_joint_input(X, Y)
    pass


# def MI_NCE(X, Y, f_ansatz, shuffle=True, batch_size=1000, N_repeats=100):
#     '''Note that in RSMI X is supposed to be the coarse-grained representation H and Y is supposed to be the environment E'''

#     @tf.function
#     def single_batch_MI(E, V):
#         return lowerbounds['infonce'](E, V, f_ansatz)

#     shuffle_size = 100000
#     dat = tf.data.Dataset.from_tensor_slices((X.T, Y.T))
#     if shuffle:
#         dat = dat.repeat(N_repeats).shuffle(shuffle_size).batch(batch_size)
#     else:
#         dat = dat.batch(batch_size).repeat(N_repeats)

#     mi = []
#     total_iterations = np.size([1 for a in dat])
#     pbar = tqdm(total=total_iterations)
#     for V, E in dat:
#         mi.append(single_batch_MI(E, V))
#         pbar.update(1)
#     mi_array = np.array(mi)
#     mi = sum(mi_array) / total_iterations
#     mi_stdev = np.sqrt(sum((mi_array - mi)**2)/total_iterations)
#     pbar.close()

#     output = {'mi':mi,
#         'mi_stdev':mi_stdev,
#         'mi_array':mi_array
#         }

#     return output
