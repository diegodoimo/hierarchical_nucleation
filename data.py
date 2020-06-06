import numpy as np

np.set_printoptions(precision=2)
from sklearn.neighbors import NearestNeighbors
import time
from scipy.optimize import curve_fit
from scipy.special import gammaln
import cython_clustering as cc

import multiprocessing

cores = multiprocessing.cpu_count()

np.set_printoptions(precision=2)
import os

os.getcwd()


class Data:

    def __init__(self, distances=None, maxk=None, verbose=True, njobs=cores, labels=None):
        self.maxk = maxk

        assert (isinstance(distances, tuple))
        if self.maxk is None:
            self.maxk = distances[0].shape[1] - 1

        self.Nele = distances[0].shape[0]
        self.distances = distances[0][:, :self.maxk + 1]
        self.dist_indices = distances[1][:, :self.maxk + 1]

        self.verb = verbose
        self.id_selected = None
        self.njobs = njobs
        self.gt_labels = labels


    def compute_id(self, decimation=1, fraction=0.9, algorithm='base'):

        if self.distances is None: self.compute_distances()

        assert (0. < decimation and decimation <= 1.)

        #self.id_estimated_ml, self.id_estimated_ml_std = return_id(self.distances, decimation)

        #Nele = len(distances)
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
            #print('ID estimated from ML is {:f} +- {:f}'.format(self.id_estimated_ml, self.id_estimated_ml_std))
            #print(f'Selecting ID of {self.id_selected}')
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
