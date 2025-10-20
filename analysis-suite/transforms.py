import numpy as np
import analyses as als
import tools


def discretize_uniform(X, nbins):
    """
    Discretize each feature of X uniformly into n bins such that marginals are uniform.
    """
    bin_edges = als.equal_bin_edges(X, nbins)
    return encode_from_bins(X, bin_edges)


def encode_from_bins(X, bin_edges):
    """
    bin_edges needs to be a list with N entries. bin_edges[n] is a list or array with the bin edges for X[n, :].
    """
    N, P = X.shape
    if len(bin_edges) != N:
        raise ValueError(
            f"bin_edges needs to be a list with {N} entries but instead has {len(bin_edges)}"
        )

    X_out = np.zeros((N, P), dtype=int)
    for n in range(N):
        X_out[n, :] = tools.encode_from_bins(X[n, :], bin_edges[n])

    return X_out


def trim(X, T):
    if T >= 0:
        return X[:, T:]
    else:
        return X[:, :T]


def time_delay_embed(X, T):
    """
    Perform time-delay embedding on input data X with a maximum shift of T.

    Time-delay embedding represents time-series data in a higher-dimensional space by creating delayed copies of the original data. The maximum shift allowed is determined by T.

    Args:
        X (ndarray): Input data array of shape (N, P), where N is the number of features and P is the number of samples.
        T (int): Maximum time shift for the embedding.

    Returns:
        ndarray: The time-delay embedded data array of shape (N, (abs(T) + 1) * P), where each column represents a delayed copy of the input data. The time dimension progresses from left to right.

    Raises:
        ValueError: If the number of features in X is less than the absolute value of T.

    Notes:
        - If T is 0, the input data X is returned without embedding.
        - When the columns of the output array are reshaped to have N entries along the first shape dimension, time progresses from left to right.
    """

    N, P = X.shape
    # if P < abs(T):
    #     print('Error (tde): maximal time shift exceeds number of data points')
    #     return None
    ## No longer necessary: new behavior handled in time_delay_embed_explicit by returning ndarray which has a second dimension shape of zero.

    # T = abs(T)
    embed_inds = np.arange(abs(T) + 1)
    return time_delay_embed_explicit(X, embed_inds)


def time_delay_embed_explicit(X, embed_inds):
    """
    Generate a new array with time delay embeddings defined by the given indices.

    For the input ndarray X with shape (N, P), this method returns an array Y time delay embeddings specified by the provided embed_inds.

    Args:
        X (ndarray): Input data array of shape (N, P), where N is the number of samples and P is the number of features.
        embed_inds (ndarray or list): Indices representing the time delay embeddings.

    Returns:
        ndarray: The output array Y of shape (N * len(embed_inds), P_out), where P_out is P - index_width. Each row of Y contains a time delay embedding of the input data.

    Raises:
        ValueError: If the number of features in X is less than the difference between the maximum and minimum values in embed_inds.

    Notes:
        - The embed_inds should be provided as an ndarray or a list.
        - The embed_inds are sorted in ascending order.
        - The number of columns in the output array Y is reduced by index_width compared to the input array X.
        - Each row of the output array Y represents a time delay embedding of the corresponding row in the input array X, following the indices specified by embed_inds.
    """

    # TODO: change so that datatype of X entries is preserved in output
    N, P = X.shape
    embed_inds = np.sort(embed_inds)
    index_width = embed_inds[-1] - embed_inds[0]

    if P < index_width:
        return np.zeros((N * len(embed_inds), 0))

    P_out = P - index_width
    X_out = np.zeros((N * len(embed_inds), P_out))
    t_new = np.arange(P_out)
    shifted_inds = embed_inds - embed_inds[0]
    for n in range(len(embed_inds)):
        X_out[n * N : (n + 1) * N, :] = X[:, t_new + shifted_inds[n]]
    return X_out


def pca(X, k):
    """
    Project X onto first k PCA modes, write in original basis. Will still have N-dimensional features

    """
    N = np.shape(X)[0]
    _, V = als.pca(X)
    theta_k = np.diag(np.concatenate([np.ones((k,)), np.zeros((N - k,))]))
    P_k = V @ theta_k @ V.T
    return P_k @ X
