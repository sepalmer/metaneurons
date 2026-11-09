import os
import re
import numpy as np
import math
from datetime import datetime, timezone
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


PARAMETER_FILE_NAME = "parameters.txt"
PARAMETER_FILE_HEADER = [
    "# Parameters used to generate each data file in this directory, one [file name] section per file.",
    "# Written by tools.record_parameters; results (e.g. float arrays) are omitted.",
]


def is_parameter(value, max_length=100):
    """decide whether a saved value is an experiment parameter rather than a result

    Scalars, strings, types and dicts of these count as parameters, as do ranges and lists,
    tuples or 1d arrays of at most `max_length` ints, bools or strings. Float sequences, longer
    sequences (e.g. train/test indices) and other arrays are treated as results.
    """
    if value is None or isinstance(value, (bool, int, float, str, type, range, np.generic)):
        return True
    if isinstance(value, dict):
        return all(is_parameter(v, max_length) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) <= max_length and all(
            isinstance(v, (bool, int, str, np.integer, np.bool_)) for v in value
        )
    if isinstance(value, np.ndarray):
        return value.ndim <= 1 and value.size <= max_length and value.dtype.kind in "biuU"
    return False


def format_parameter(value):
    """write a parameter value as a single readable line"""
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {format_parameter(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        items = [format_parameter(v) for v in value]
        if isinstance(value, tuple):
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        return "[" + ", ".join(items) + "]"
    return repr(value)


def read_parameter_sections(parameter_path):
    """read a parameters.txt file into a dict mapping data file name to its lines"""
    sections = {}
    if not os.path.exists(parameter_path):
        return sections
    current = None
    with open(parameter_path, "r") as f:
        for line in f.read().splitlines():
            header = re.fullmatch(r"\[(.+)\]", line)
            if header:
                current = header.group(1)
                sections[current] = []
            elif current is not None and line.strip():
                sections[current].append(line)
    return sections


def current_notebook():
    """path of the running notebook relative to the project folder, or None outside a notebook

    VS Code sets `__vsc_ipynb_file__` to the absolute path. Jupyter sets `__session__` relative
    to the server root, so a relative path is resolved against the kernel's working directory,
    which is the notebook's own directory by default.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None:
        return None
    path = shell.user_ns.get("__vsc_ipynb_file__") or shell.user_ns.get("__session__")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), os.path.basename(path))
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.relpath(path, project_folder)


def record_parameters(data_path, params, comment=None, notebook=None):
    """record the parameters used to generate the file at `data_path`

    Entries are kept in parameters.txt in the same directory as `data_path`, one section per
    data file; an existing section for the same file is replaced. `params` may be the full
    `output_data` dict, since only values passing `is_parameter` are written. The generating
    notebook is detected automatically unless `notebook` is given.
    """
    directory, file_name = os.path.split(os.path.abspath(data_path))
    parameter_path = os.path.join(directory, PARAMETER_FILE_NAME)
    if comment is None:
        comment = "recorded " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if notebook is None:
        notebook = current_notebook()

    sections = read_parameter_sections(parameter_path)
    sections[file_name] = [f"# {comment}"] + ([f"# notebook: {notebook}"] if notebook else []) + [
        f"{k} = {format_parameter(params[k])}" for k in sorted(params) if is_parameter(params[k])
    ]

    lines = list(PARAMETER_FILE_HEADER)
    for name in sorted(sections):
        lines += ["", f"[{name}]"] + sections[name]
    os.makedirs(directory, exist_ok=True)
    with open(parameter_path, "w") as f:
        f.write("\n".join(lines) + "\n")
