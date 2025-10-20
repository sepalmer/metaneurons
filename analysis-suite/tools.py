import numpy as np
import math
from random import sample
from scipy.spatial import distance_matrix
import matplotlib as mpl

fac = math.factorial

def sample_cmap(x, cmap='bwr', alp=1):
    x = unitize_range(x)
    cmap = mpl.colormaps[cmap]
    colors = np.array([cmap(y) for y in x])
    colors[:,-1] = alp
    return colors
    

def unitize_range(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def linear_color(x, color_1, color_2):
    x = unitize_range(x)
    color_1 = np.array(color_1)
    color_2 = np.array(color_2)
    return [tuple((1 - y) * color_1 + y * color_2) for y in x]

def view_model_layers(layer, level=0):
    # Print the current layer
    print('  ' * level + f'{layer.name} ({layer.__class__.__name__})')

    # Check if this layer has sub-layers and if so, iterate through them recursively
    if hasattr(layer, 'layers') and layer.layers:
        for sub_layer in layer.layers:
            view_model_layers(sub_layer, level + 1)

def distance_threshold_clustering(points, threshold, distance_function=None):
    """
    Identify clusters from a list of N-dimensional points based on a threshold distance.

    Example usage:
        points = np.random.randn(100, 3) # Generate 100 random 3-dimensional points
        clusters = distance_threshold_clustering(points, threshold=0.1)
        print(clusters)

    """
    # Compute the distance matrix between the points
    if distance_function is None:
        distances = distance_matrix(points, points)
    else:
        distances = distance_function(points, points)

    # Initialize the list of clusters
    clusters = []

    # Assign each point to a cluster based on the threshold distance
    for i in range(len(points)):
        assigned = False
        for j in range(len(clusters)):
            if any([distances[i, k] <= threshold for k in clusters[j]]):
                clusters[j].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    # Convert the cluster indices to NumPy arrays of points
    cluster_points = [np.array([points[i] for i in c]) for c in clusters]

    return clusters, cluster_points


def cov_entrywise(X, sum_axes):
    """
    Computes the element-wise covariance of an ndarray X along the specified sum axes.

    :param X: The input ndarray.
    :param sum_axes: A list of axes along which to sum for the covariance calculation.
    :return: Element-wise covariance matrix.
    """
    # If no axes are provided, return None or raise an error
    if not sum_axes:
        raise ValueError(
            "At least one axis must be specified for the covariance calculation."
        )

    # Ensuring that the axes are a list
    if isinstance(sum_axes, int):
        sum_axes = [sum_axes]

    # Normalize X by subtracting the mean along the specified axes
    mean_X = np.mean(X, axis=tuple(sum_axes), keepdims=True)
    X_centered = X - mean_X

    # Compute the element-wise covariance
    n_elements = np.prod(
        [X.shape[axis] for axis in sum_axes]
    )  # Total number of elements over sum_axes
    cov = np.sum(X_centered**2, axis=tuple(sum_axes)) / (n_elements - 1)

    return cov

def x_fraction_max(x, y, frac):
    y_hm = np.max(y) * frac
    i = np.arange(len(y))
    greater_than = y > y_hm
    i_switches = np.diff(greater_than * 1)

    # could generalize this to return a list of solutions
    i_left = np.where(i_switches != 0)[0][0]
    i_right = i_left + 1
    t = x[i_left] - (x[i_left] - x[i_right]) * (y[i_left] - y_hm) / (y[i_left] - y[i_right])
    return t

def x_half_max(x, y):
    return x_fraction_max(x, y, 1/2)

# def x_half_max(x, y):
#     y_hm = np.max(y) / 2
#     i = np.arange(len(y))
#     greater_than = y > y_hm
#     i_switches = np.diff(greater_than * 1)

#     # could generalize this to return a list of solutions
#     i_left = np.where(i_switches != 0)[0][0]
#     i_right = i_left + 1
#     t = x[i_left] - (x[i_left] - x[i_right]) * (y[i_left] - y_hm) / (y[i_left] - y[i_right])
#     return t


def arg_fraction_max(y, frac):
    i = np.argsort(y)
    y_max = y[i[-1]]
    cost = np.abs(y - frac * y_max)
    return np.argmin(cost)

def arg_half_max(y):
    return arg_fraction_max(y, 1/2)

# def arg_half_max(y):
#     i = np.argsort(y)
#     y_max = y[i[-1]]
#     cost = np.abs(y - y_max / 2)
#     return np.argmin(cost)


def CCA_information_deterministic(lam, log_base=np.e):
    return -1 / 2 * np.cumsum(np.log(lam)) / np.log(log_base)


def generate_train_test_indices(N_points, train_fraction):
    N_train_points = int(N_points * train_fraction)
    N_test_points = N_points - N_train_points
    train_points = sample(range(N_points), k=N_train_points)
    test_points = list(set(range(N_points)) - set(train_points))
    return train_points, test_points


def equal_bin_edges(x, n_bins):
    """calculate bin edges so that x can be equipartitioned"""
    N = len(x)
    xs = np.sort(x)
    width = (N - 1) / n_bins
    indices = (width * np.arange(n_bins + 1)).astype(int)
    edges = np.unique(xs[indices])
    edges[0] = xs[0] - 1
    edges[-1] = xs[-1] + 1
    return edges


def encode_from_bins(x, edges):
    """given a 1D series `x` and bin edges `edges`, return the sequence of bin indices"""
    comparison_mask = (edges[None, :-1] <= x[:, None]) * (x[:, None] < edges[None, 1:])
    return np.sum(comparison_mask * np.arange(len(edges) - 1), axis=1)


def broadcast(X, f):
    Y = list()
    for x in X:
        Y.append(f(x))
    return Y


def cae_entropy(counts):
    n = np.sum(counts)
    p = counts / n
    f1 = np.count_nonzero(counts == 1)
    if f1 == n:
        f1 = n - 1
    C = 1 - f1 / n
    pa = C * p
    ma = 1 - (1 - pa) ** n
    return -np.sum(pa * np.log(pa) / ma)


def count_partitions_blocks(b):
    return count_partitions(blocks_to_counts(b))


def count_partitions(m):
    """
    m is a list or numpy array with the form (m1,m2,...,mn) with m1 + 2*m2 + 3*m3 + ... + n*mn = n

    output number of ways of partitioning n numbers into blocks with sizes described by mj
    """
    m = np.array(m)
    n = len(m)
    j = np.arange(n) + 1
    return (fac(n) / np.prod(((broadcast(j, fac) ** m) * broadcast(m, fac)))).astype(
        int
    )


def blocks_to_counts(b):
    n = sum(np.array(b))
    return list(np.histogram(b, bins=np.arange(n + 1) + 1)[0])


def counts_to_blocks(m):
    n = len(m)
    b = list()
    for k in range(n):
        for l in range(m[k]):
            b.append(k + 1)
    return b


# def blocks_to_counts(b):
#     '''
#     convert from block description of partition to multiplicity description. For example

#     n = 3

#     input:
#     b = [[3], [1, 2], [1, 1, 1]]

#     output:
#     m = [[0,0,1], [1,1,0], [3,0,0]]
#     '''
#     n = max([max(a) for a in b])
#     k = len(b)
#     m = list()
#     for j in range(k):
#         m.append(list(np.histogram(b[j], bins=np.arange(n+1)+1)[0]))

#     return m

# def counts_to_blocks(m):
#     '''
#     convert from multiplicity description of partition to block description. Inverts blocs_to_counts(b)
#     '''
#     n = len(m[0])
#     b = list()
#     for j in range(len(m)):
#         b.append(list())
#         for k in range(n):
#             for l in range(m[j][k]):
#                 b[j].append(k+1)
#     return b


def partition(number):
    """adapted from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning

    return all partitions of number in block form
    """
    return [list(a) for a in list(partition_(number))]


def partition_(number):
    """adapted from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning"""
    answer = set()
    answer.add((number,))
    for x in range(1, number):
        for y in partition_(number - x):
            answer.add(tuple(sorted((x,) + y)))
    return answer
