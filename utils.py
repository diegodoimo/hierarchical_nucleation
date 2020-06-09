import numpy as np

np.set_printoptions(precision=2)
import time
from scipy.optimize import curve_fit
from scipy.special import gammaln
import cython_clustering as cc

import multiprocessing

cores = multiprocessing.cpu_count()

np.set_printoptions(precision=2)
import os

os.getcwd()


def double_centering(M):
    N = M.shape[0]

    Mc = - 0.5 * (M - 1. / N * np.sum(M, axis=0)[None, :] - 1. / N * np.sum(M, axis=1)[:,
                                                                     None] + 1. / N ** 2 * np.sum(M))

    return Mc


def kernel_PCA(layer):
    # obtain distance matrix from density peaks and saddle points
    distances = max(layer.Rho) - layer.out_bord
    distances[np.diag_indices_from(distances)] = 0.

    if distances.shape[0] < 2:
        return np.array([[0, 0]])

    else:
        # compute average distance to set lengthscae of gaussian kernel
        md = np.mean(distances[np.triu_indices_from(distances)])
        sig = np.mean(md)

        # compute kernel matrix
        ker = np.exp(- distances ** 2 / (0.5 * sig ** 2))
        ker = double_centering(-ker)

        e, v = np.linalg.eigh(ker)
        s = np.argsort(e)[::-1]
        out = v[:, s[:2]]

        return out


class Data:

    '''
    Computes the overlap with the ground truth labels and the density peaks of a representation.

    Parameters:
    distances:  dictionary containing the k-nearest neighbor distances (distnces[0] of shape [n_samples, n_nearest neighbors])
                and their integer labels(distances[1] of shape [n_samples, n_nearest neighbors])
    maxk:       integer. the number k of nearest neighbors to use in the computation

    Methods:
    compute_id:
        Returns: float: the intrinsic dimension of the dataset.

    compute_density_kNN: computes the local density of all the data points using with k nearest neighbors.
                        (k=30 by default)

    compute_clustering: computes the density peaks. (Z=1.65 by default)

    return_label_overlap_mean:
        Returns: float: the average ground thruth overlap using the first k=30 neighbors

    '''

    def __init__(self, distances=None, maxk=None, verbose=True, njobs=cores, labels=None):
        self.maxk = maxk

        assert (isinstance(distances, tuple))
        if self.maxk is None:
            self.maxk = distances[0].shape[1] - 1

        #nu
        self.Nele = distances[0].shape[0]
        self.distances = distances[0][:, :self.maxk + 1]
        self.dist_indices = distances[1][:, :self.maxk + 1]

        self.verb = verbose
        self.id_selected = None
        self.njobs = njobs
        self.gt_labels = labels

    def remove_zero_dists(self):
        # TODO remove all the degenerate distances
        assert (self.distances is not None)

        # find all points with any zero distance
        indx = np.nonzero(self.distances[:, 1] < np.finfo(float).eps)

        # set nearest distance to eps:
        self.distances[indx, 1] = np.finfo(float).eps

        #print('{} couple of points where at 0 distance: their distance have been set to eps: {}'.format(len(indx[0]), np.finfo(float).eps))

    def compute_id(self, decimation=1, fraction=0.9, algorithm='base'):
        self.remove_zero_dists()
        assert (0. < decimation and decimation <= 1.)

        # self.id_estimated_ml, self.id_estimated_ml_std = return_id(self.distances, decimation)

        # Nele = len(distances)
        # remove highest mu values
        Nele_eff = int(self.Nele * fraction)
        mus = np.log(self.distances[:, 2] / self.distances[:, 1])
        mus = np.sort(mus)[:Nele_eff]

        Nele_eff_dec = int(np.around(decimation * Nele_eff, decimals=0))
        idxs = np.arange(Nele_eff)
        idxs = np.random.choice(idxs, size=Nele_eff_dec, replace=False, p=None)
        mus_reduced = mus[idxs]

        if algorithm == 'ml':
            id = Nele_eff_dec / np.sum(mus_reduced)

        elif algorithm == 'base':
            def func(x, m):
                return m * x

            y = np.array([-np.log(1 - i / self.Nele) for i in range(1, Nele_eff + 1)])
            y = y[idxs]
            id, _ = curve_fit(func, mus_reduced, y)

        self.id_selected = int(np.around(id, decimals=0))

        if self.verb:
            # print('ID estimated from ML is {:f} +- {:f}'.format(self.id_estimated_ml, self.id_estimated_ml_std))
            # print(f'Selecting ID of {self.id_selected}')
            print(f'ID estimation finished: selecting ID of {self.id_selected}')

    def compute_density_kNN(self, k=30):
        assert (self.id_selected is not None)

        if self.verb: print('k-NN density estimation started (k={})'.format(k))

        kstar = np.empty(self.Nele, dtype=int)
        dc = np.empty(self.Nele, dtype=float)
        Rho = np.empty(self.Nele, dtype=float)
        Rho_err = np.empty(self.Nele, dtype=float)
        prefactor = np.exp(
            self.id_selected / 2. * np.log(np.pi) - gammaln((self.id_selected + 2) / 2))
        Rho_min = 9.9E300

        for i in range(self.Nele):
            kstar[i] = k
            dc[i] = self.distances[i, k]
            Rho[i] = np.log(kstar[i]) - (
                    np.log(prefactor) + self.id_selected * np.log(self.distances[i, kstar[i]]))

            Rho_err[i] = 1. / np.sqrt(k)
            if (Rho[i] < Rho_min):
                Rho_min = Rho[i]

        # Normalise density
        Rho -= np.log(self.Nele)

        self.Rho = Rho
        self.Rho_err = Rho_err
        self.dc = dc
        self.kstar = kstar

        if self.verb: print('k-NN density estimation finished')

    def compute_clustering(self, Z=1.65, halo=False):
        assert (self.Rho is not None)
        if self.verb: print('Clustering started')

        # Make all values of Rho positives (this is important to help convergence)
        Rho_min = np.min(self.Rho)

        Rho_c = self.Rho + np.log(self.Nele)
        Rho_c = Rho_c - Rho_min + 1

        # Putative modes of the PDF as preliminar clusters

        Nele = self.distances.shape[0]
        g = Rho_c - self.Rho_err
        # centers are point of max density  (max(g) ) within their optimal neighborhood (defined by kstar)
        seci = time.time()

        out = cc._compute_clustering(Z, halo, self.kstar, self.dist_indices.astype(int), self.maxk,
                                     self.verb, self.Rho_err, Rho_min, Rho_c, g, Nele)

        secf = time.time()

        self.clstruct_m = out[0]
        self.Nclus_m = out[1]
        self.labels = out[2]
        self.centers_m = out[3]
        out_bord = out[4]
        Rho_min = out[5]
        self.Rho_bord_err_m = out[6]

        self.out_bord = out_bord + Rho_min - 1 - np.log(
            Nele)
        for i in range(len(self.centers_m)):
            self.out_bord[i,i] = self.Rho[self.centers_m[i]]

        if self.verb:
            print('Clustering finished, {} clusters found'.format(self.Nclus_m))
            print('total time is, {}'.format(secf - seci))

    def return_rank_distribution(self):
        neigh_identities = self.dist_indices[:, 1:] == self.gt_labels[:, None]

        same_label = np.sum(neigh_identities, axis=1) / self.Nele
        different_label = np.sum(~neigh_identities, axis=1) / self.Nele

        return same_label, different_label

    def return_label_overlap_all(self, k=30):
        assert (self.distances is not None)
        assert (self.gt_labels is not None)

        overlaps = []
        for i in range(self.Nele):
            neigh_idx_i = self.dist_indices[i, 1:k + 1]
            overlaps.append(sum(self.gt_labels[neigh_idx_i] == self.gt_labels[i]) / k)

        return overlaps

    def return_label_overlap_mean(self, k=30):
        assert (self.distances is not None)
        assert (self.gt_labels is not None)

        overlap = 0.
        for i in range(self.Nele):
            neigh_idx_i = self.dist_indices[i, 1:k + 1]
            overlap += sum(self.gt_labels[neigh_idx_i] == self.gt_labels[i]) / k

        overlap = overlap / self.Nele

        return overlap


class Data_sets:

    '''
    Computes the overlaps and density peaks of a set of datasets e.g. the layers of a deep neural network;
    Parameters:
        distance_list: list of distances (as defined in Data).

        maxk_list: list of integers. The number k of nearest neighbors of each dataset to use in the computations.

        labels_list: ...

    Methods:

    '''

    def __init__(self, distances_list=(), labels_list=(),
                 maxk_list=[None], verbose=False, njobs=1):

        self.Nsets = len(distances_list)

        if len(maxk_list) == 1:
            maxk_list = [maxk_list[0]] * self.Nsets

        assert (len(labels_list) == 0 or len(labels_list) == self.Nsets)

        self.data_sets = []

        for i in range(self.Nsets):
            dists = distances_list[i]
            maxk = maxk_list[i]

            labels = labels_list[i]
            data = Data(distances=dists, labels=labels, maxk=maxk, verbose=verbose, njobs=njobs)

            self.data_sets.append(data)

        self.verbose = verbose
        self.njobs = njobs
        self.ids = None  # ids
        self.ov_gt = None  # overlap ground truth (classes)
        self.ov_out = None  # overlap output neighborhoods
        self.ov_ll = None  # overlap ll neighbourhoods

    def add_one_dataset(self, distances=None, labels=None, maxk=None):

        data = Data(distances=distances, labels=labels,
                    maxk=maxk, verbose=self.verbose, njobs=self.njobs)

        self.data_sets.append(data)
        self.Nsets += 1

    def compute_id(self, decimation=1, fraction=0.9):

        for i, d in enumerate(self.data_sets):
            print('computing id of layer ', i)
            d.compute_id(decimation=decimation, fraction=fraction)

        self.ids = [d.id_selected for d in self.data_sets]

    def compute_density_kNN(self, k):
        for i, d in enumerate(self.data_sets):
            print('computing kNN density for dataset ', i)
            d.compute_density_kNN(k=k)

    def compute_clustering(self, Z=1.65, halo=True):

        print('Z = {}, halo = {}'.format(Z, halo))
        for i, d in enumerate(self.data_sets):
            print('computing clustering for dataset ', i)
            d.compute_clustering(Z=Z, halo=halo)
        print('Z = {}, halo = {}'.format(Z, halo))

    def return_label_overlap_mean(self, k=30):

        overlap_means = []

        for i, d in enumerate(self.data_sets):
            print(f'computing overlap for dataset {i}/{len(self.data_sets)}')
            overlap_means.append(d.return_label_overlap_mean(k=k))

        return overlap_means

    def return_label_overlap_all(self, k=30):

        overlap_alls = []

        for i, d in enumerate(self.data_sets):
            print('computing overlap for dataset ', i)
            overlap_alls.append(d.return_label_overlap_all(k=k))

        return overlap_alls

    def return_overlap_mean_btw_layers(self, i1, i2, k=30):

        ranks_layer1 = self.data_sets[i1].dist_indices
        ranks_layer2 = self.data_sets[i2].dist_indices

        Nimg_tot = ranks_layer1.shape[0]
        cs = []
        for i in range(Nimg_tot):
            cs.append(len(set(ranks_layer1[i, 1:k + 1]) & set(ranks_layer2[i, 1:k + 1])))

        return np.mean(cs) / k

    def return_overlap_mean_ll(self, step=1, k=30):
        ov_ll = []
        for i in range(self.Nsets - step):
            print('computing overlap for datasets ', i, i + step)
            ov_ll.append(self.return_overlap_mean_btw_layers(i, i + step, k))
        return ov_ll
