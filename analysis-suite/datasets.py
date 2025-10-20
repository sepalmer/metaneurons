import numpy as np
import transforms as tfs
import analyses as als
import tools
import copy
from scipy import io as io
import os
import pickle
from random import sample, shuffle

# TODO: Make dataset transformation code less redundant?


def X_is_iterable_of_Y(X, Y):
    if isinstance(X, (list, tuple, np.ndarray)):
        for item in X:
            if not isinstance(item, Y):
                return False
        return True
    elif isinstance(X, np.ndarray):
        pass
    else:
        return False


def load_dataset(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


class SaveableObject(object):
    """
    Simple class that allows straightforward pickling by inheritance.
    """

    def __init__(self):
        pass

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path), "wb+") as f:
            pickle.dump(self, f)


class Dataset(SaveableObject):
    def __init__(self, data):
        if not (isinstance(data, np.ndarray) and data.ndim == 2):
            print(
                "Error: data needs to be in standard format of numpy ndarray with shape (N, P)"
            )
            return None

        # self.N, self.P = data.shape
        self.X = data
        self.reset_point = copy.deepcopy(data)

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def P(self):
        return self.X.shape[1]

    def reset(self):
        self.__init__(self.reset_point)
        return self

    def trim(self, T):
        self.X = tfs.trim(self.X, T)
        return self

    def tde(self, T):
        self.X = tfs.time_delay_embed(self.X, T)
        return self

    def project(self, P):
        self.X = P @ self.X
        return self

    def pca(self, k):
        self.X = tfs.pca(self.X, k)
        return self

    def subset_features(self, inds):
        self.X = self.X[inds, :]
        return self

    def subset_points(self, inds):
        self.X = self.X[:, inds]
        return self

    def discretize_uniform(self, nbins):
        self.X = tfs.discretize_uniform(self.X, nbins)
        return self

    def encode_from_bins(self, bin_edges):
        self.X = tfs.encode_from_bins(self.X, bin_edges)
        return self
    
    def create_V_and_E_datasets(self, V_width, V_E_buffer, E_width, samp_keep=1):
        """
        Return two datasets, V and E, with time widths V_width and E_width, separated by V_E_buffer timesteps.
        """

        # first create a copy of the current dataset. copy.deepcopy is necessary to prevent this operation from changning the internals of D.
        A = Dataset(self.X.copy())

        # now time delay embed to necessary width
        block_width = V_width + E_width + V_E_buffer
        A.tde(block_width - 1)

        # randomly select a fraction samp_keep of the data in each trial to keep
        subset_inds = sample(range(A.P), k=int(samp_keep * A.P))
        A.subset_points(subset_inds)

        # TODO: we could also have this return DataTrialed versions of V ane E with a little bit of work, but unclear if needed yet.
        V = Dataset(A.X[: V_width * self.N, :])
        E = Dataset(A.X[(V_width + V_E_buffer) * self.N :, :])
        return V, E

    def shuffle_time(self):
        new_order = sample(range(self.P), k=self.P)
        self.subset_points(new_order)
        return self



class DataTrialed(SaveableObject):
    """
    For instances where there are a collection of dataset objects which need to be analyzed individually. Designed essentially for data with time domains and multiple trials, where trials can't be mixed together.
    """

    def __init__(self, data):
        """
        Allowed input formats for data: tuple or list of
        """
        if X_is_iterable_of_Y(data, Dataset):
            self.data = np.array(data).copy()

        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self.data = np.array([Dataset(data)]).copy()

        elif X_is_iterable_of_Y(data, np.ndarray):
            self.data = np.array([Dataset(d) for d in data]).copy()

        else:
            print("Error: data is not an iterable of Dataset objects")
            return None

        self.N_trials = len(self.data)
        N = np.array([d.N for d in self.data])
        if len(np.unique(N)) > 1:
            print("Error: dimensionality N does not match across input Dataset objects")
            return None
        # self.N = N[0]
        self.update_X()
        # self.N = self.X.shape[0]
        # self.P = np.array([d.P for d in self.data])
        self.reset_point = copy.deepcopy(data)

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def P(self):
        return np.array([d.P for d in self.data])

    def reset(self):
        self.__init__(self.reset_point)
        return self

    def update_X(self):
        self.X = np.concatenate([d.X for d in self.data], axis=1)
        return self

    def trim(self, T):
        for d in self.data:
            d.trim(T)
        self.update_X()
        return self

    def tde(self, T):
        for d in self.data:
            d.tde(T)
        self.update_X()
        return self

    def project(self, P):
        for d in self.data:
            d.X = P @ d.X
        self.update_X()
        return self

    def pca_trialed(self, k):
        for d in self.data:
            d.pca(k)
        self.update_X()
        return self

    def pca_pooled(self, k):
        """
        PCA on entire dataset, with all trials first pooled into a single trial
        """
        _, V = als.pca(self.X)
        theta_k = np.diag(np.concatenate([np.ones((k,)), np.zeros((self.N - k,))]))
        P_k = V @ theta_k @ V.T
        for d in self.data:
            d.project(P_k)
        self.update_X()
        return self

    def subset_features(self, inds):
        for d in self.data:
            d.subset_features(inds)
        self.update_X()
        return self

    def subset_points_uniform(self, inds):
        """
        Perform the subset_points operation across all Datasets within this DatasetTrialed
        """
        for d in self.data:
            d.subset_points(inds)
        self.update_X()
        return self

    def subset_points_by_trial(self, inds_list):
        """
        Perform subset_points operation on all Datasets with indices specified for each trial
        """
        a = X_is_iterable_of_Y(inds_list, list)
        b = X_is_iterable_of_Y(inds_list, np.ndarray)
        if not (a or b):
            raise ValueError("inds_list is not an iterable of lists or numpy ndarrays.")

        if len(inds_list) != len(self.P):
            raise ValueError(
                f"inds_list has length {len(inds_list)} but should have length {len(self.P)}"
            )

        for i, d in enumerate(self.data):
            d.subset_points(inds_list[i])
        self.update_X()
        return self

    def create_V_and_E_datasets(self, V_width, V_E_buffer, E_width, samp_keep=1):
        """
        Return two datasets, V and E, with time widths V_width and E_width, separated by V_E_buffer timesteps.
        """

        # first create a copy of the current dataset. copy.deepcopy is necessary to prevent this operation from changning the internals of D.
        A = DataTrialed(copy.deepcopy(self.data))

        # now time delay embed to necessary width
        block_width = V_width + E_width + V_E_buffer
        A.tde(block_width - 1)

        # randomly select a fraction samp_keep of the data in each trial to keep
        subset_inds = []
        for i, P in enumerate(A.P):
            subset_inds.append(
                list(np.sort(sample(range(P), k=int(samp_keep * P)))))
            # subset_inds.append(sample(range(P), k=int(samp_keep * P)))
        A.subset_points_by_trial(subset_inds)

        # TODO: we could also have this return DataTrialed versions of V ane E with a little bit of work, but unclear if needed yet.
        V = Dataset(A.X[: V_width * self.N, :])
        E = Dataset(A.X[(V_width + V_E_buffer) * self.N :, :])
        return V, E

    # TODO: we want this to perform the historgram binning across trials, I think.
    def discretize_uniform(self, nbins):
        bin_edges = als.equal_bin_edges(self.X, nbins)
        return self.encode_from_bins(bin_edges)

    def encode_from_bins(self, bin_edges):
        for d in self.data:
            d.encode_from_bins(bin_edges)
        self.update_X()
        return self

    def shuffle_time(self):
        """
        Pick random time permutations for each trial and apply them without mixing between trials
        """
        for d in self.data:
            d.shuffle_time()
        self.update_X()
        return self
        # new_order = [sample(range(p), k=p) for p in D.P]
        # self.subset_points_by_trial(new_order)
        # self.update_X()

    def shuffle_time_and_trial(self):
        """
        Randomly permute time bins across all trials without removing trial structure
        """
        A = Dataset(self.X)
        A.shuffle_time()
        inds = np.cumsum(np.concatenate((np.array([0]), self.P)))
        new_data = []
        for i in range(len(self.P)):
            new_data.append(Dataset(A.X[:,inds[i]:inds[i+1]]))
        self.data = new_data
        self.update_X()
        return self

class DataBarCommonFutures(DataTrialed):
    """
    This handles the dataset used in the PNAS 2015 paper.

    There were 6000 planned stimulus trials, but only 3350 of them were collected before deterioration of responses.

    Each stimulus repeat is 100 time bins long and has one of 30 possible future segments, contained in the latter 50 bins. The identity of each repeat is stored in the "futures_order" entry.

    keys:
        "clips": shape (30, 200, 100) array, meaning 30 possible futures, 200 distinct pasts for each, and 100 timebins long
        "clips_future" (30, 50) array, meaning the 30 possible futures, each 50 timebins long.
        "clips_past" (30, 200, 50) array, meaning 30 possible futures with 200 distinct pasts, each with

    """

    def __init__(self, file_path, sort_by_activity=True):
        self.file_path = file_path
        self.all_data = io.loadmat(file_path, squeeze_me=True)
        self.binned = self.all_data[
            "alldata"
        ]  # (49, 3350, 100) cell number x repeat x timestep
        nreps = np.shape(self.binned)[1]
        self.binsize = self.all_data["dt"]  # 16.67
        self.raw_stimulus = self.all_data[
            "clips"
        ]  # (30, 200, 100) futures x pasts x timestep
        self.futures_order = (self.all_data["futures_order"] - 1)[:nreps].astype(
            int
        )  # (3350) future identities
        self.past_index_order = np.zeros(
            nreps, dtype=int
        )  # the index of the past entry on a given trial.

        self.futures = self.all_data["clips_future"]  # (30, 50) futures x timestep
        self.stim_data = []
        past_counter = np.zeros((30,), dtype=int)
        for trial in range(nreps):
            future = self.futures_order[trial]
            past = past_counter[future]
            past_counter[future] += 1
            self.past_index_order[trial] = past
            self.stim_data.append(self.raw_stimulus[future, past, :][None, :])

        self.response_data = [
            self.binned[:, trial, :].astype(int) for trial in range(nreps)
        ]

        super().__init__(self.response_data)

        if sort_by_activity:
            counts = np.sum(np.concatenate(self.response_data, axis=1), axis=1)
            i = np.argsort(-counts)
            self.subset_features(i)

    def reset(self):
        self.__init__(self.file_path)

    def trial_to_pair(self, trial):
        return (self.futures_order[trial], self.past_index_order[trial])

    def pair_to_trial(self, future, past):
        pass


class DataBar(DataTrialed):
    def __init__(self, file_path, sort_by_activity=True):
        # get and store movie data
        self.file_path = file_path
        self.all_data = io.loadmat(file_path, squeeze_me=True)
        self.binned = self.all_data[
            "nonrepbardata"
        ]  # Cell Number x trial index x time axis  shape: (53,62,5221)
        # many entries are nan, these need to be filtered out by hand during analysis. the valid bins are stored in the matrix valid_bins
        # keep_inds are the valid inds
        self.raw_stimulus = self.all_data["nonrepbarstim"]
        self.binsize = self.all_data["dt"]  # 16.67 ms

        # build dataset
        nreps = np.shape(self.binned)[1]
        valid_bins = self.all_data["nonrepbarstim"] > 0

        self.stim_data = [
            self.raw_stimulus[trial, valid_bins[trial]][None, :]
            for trial in range(nreps)
        ]

        # TODO: make entries integers
        self.response_data = [
            self.binned[:, trial, valid_bins[trial]].astype(int)
            for trial in range(nreps)
        ]
        super().__init__(self.response_data)

        if sort_by_activity:
            counts = np.sum(np.concatenate(self.response_data, axis=1), axis=1)
            i = np.argsort(-counts)
            self.subset_features(i)

    def reset(self):
        self.__init__(self.file_path)


class DataNatScenes(DataTrialed):
    """
    Notes:
        - shape of self.binned is different from DataBar to conform to design of older class. This does not affect the representation of self.X, which is always (N, P) = (features, points).
    """

    def __init__(self, file_path, mov_ind, sort_by_activity=True):
        # get and store movie data
        self.file_path = file_path
        self.all_data = io.loadmat(file_path, squeeze_me=True)

        self.mov_ind = mov_ind  # necessary for self.reset()
        nreps = self.all_data["nreps"][
            mov_ind
        ]  # array([83, 80, 84, 91, 85]), number of repititions of each move.
        self.binned = self.all_data["binned"][
            :nreps, :, :, mov_ind
        ]  # Trials x Time x Cell Number x Movie Number = (91,1203,93,5)
        # binned comes in as ragged. For example, move 1, 'bushes', only gets 83 repeats. In this case, binned[j,:,:,0] is zero for all indices j > 83
        # This needs to be trimmed, so best practice is to do a separate array for each movie.

        self.binsize = self.all_data["binsize"]  # 16.67 ms
        self.movname = self.all_data["movnames"][
            mov_ind
        ]  # movie name string, selected from array(['bushes', 'water', 'leaves', 'fish', 'opticflow'], dtype=object)
        self.ncell = self.all_data["ncell"]  # 93, same as len(clean)

        # build dataset
        response_data = [
            self.binned[trial, :, :].T.astype(int) for trial in range(nreps)
        ]
        super().__init__(response_data)

        if sort_by_activity:
            counts = np.sum(np.concatenate(response_data, axis=1), axis=1)
            i = np.argsort(-counts)
            self.subset_features(i)

    def reset(self):
        self.__init__(self.file_path, self.mov_ind)

    def shuffle_time_uniform(self):
        """
        Pick a random time permutation and apply it to all trials
        """
        N_points = self.P[0]
        new_order = sample(range(N_points), k=N_points)
        self.subset_points_uniform(new_order)

    def shuffle_time_and_trial(self):
        """
        Randomly permute all trial and time indices
        """
        X = np.stack([d.X for d in self.data], axis=-1)
        N, P, T = X.shape
        X = X.reshape((N, P * T))
        new_order = sample(range(P * T), k=P * T)
        X = X[:, new_order]
        X = X.reshape((N, P, T))
        self.data = [Dataset(X[:,:,t]) for t in range(T)]
        return self
    
    def shuffle_trial_per_time(self):
        """
        For each time bin, randomly permute trial index
        """
        X = np.stack([d.X for d in self.data], axis=-1)
        N, P, T = X.shape
        for p in range(P):
            new_order = sample(range(T), k=T)
            X[:,p,:] = X[:,p,new_order]
        self.data = [Dataset(X[:,:,t]) for t in range(T)]
        return self

    def shuffle_trial_per_cell(self):
        """
        For each cell, randomly permute trial index
        """
        X = np.stack([d.X for d in self.data], axis=-1)
        N, P, T = X.shape
        for n in range(N):
            new_order = sample(range(T), k=T)
            X[n,:,:] = X[n,:,new_order].T  # new_order puts last index first here. This doesn't happen in shuffle_trial_per_time. 
        self.data = [Dataset(X[:,:,t]) for t in range(T)]
        return self

    def shuffle_trial(self):
        "for each cell and time, shuffle trial"
        X = np.stack([d.X for d in self.data], axis=-1)
        N, P, T = X.shape
        for n in range(N):
            for p in range(P):
                new_order = sample(range(T), k=T)
                X[n,p,:] = X[n,p,new_order]
        self.data = [Dataset(X[:,:,t]) for t in range(T)]
        return self

