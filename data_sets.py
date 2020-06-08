import numpy as np

from data import Data

class Data_sets:

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

        #print(f'computing id: fraction = {fraction}, decimation = {decimation}, range = 2'.format)
        for i, d in enumerate(self.data_sets):
            print('computing id of layer ', i)
            d.compute_id(decimation=decimation, fraction=fraction)
            #print('id computation finished')

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
            print('computing overlap for dataset ', i)
            overlap_means.append(d.return_label_overlap_mean(k=k))

        return overlap_means

    def return_label_overlap_all(self, k=30):

        overlap_alls = []

        for i, d in enumerate(self.data_sets):
            print('computing overlap for dataset ', i)
            overlap_alls.append(d.return_label_overlap_all(k=k))

        return overlap_alls

    def overlap_classes_btw_layers(self, ranks_layer1, ranks_layer2, k, nimg_class, nclasses):
        Nimg_tot = ranks_layer1.shape[0]
        c = np.empty((Nimg_tot))
        for j in range(Nimg_tot):
            ranks_classes_imgj_layer1 = ranks_layer1[j, 1:k + 1] // nimg_class
            ranks_classes_imgj_layer2 = ranks_layer2[j, 1:k + 1] // nimg_class
            n = 0
            for i in range(nclasses):
                n += min(np.sum(ranks_classes_imgj_layer1 == i),
                         np.sum(ranks_classes_imgj_layer2 == i))
            c[j] = n / k
        return c

    def overlap_classes_ground_truth(self, ranks_images, k, nimg_cat, ncat, jsubset):
        N_images = ranks_images.shape[0]
        c = np.empty((N_images))
        for j in range(N_images):
            ranks_categories = ranks_images[j][1:k + 1] // nimg_cat
            true_label = (j + jsubset) // nimg_cat
            n = np.sum(ranks_categories == true_label)
            c[j] = n / k
        return c

    def return_overlap_classes_output(self, ncat, nimg_cat, k=10, jsubset=0):
        ov_out_cl = np.empty((self.Nsets, self.data_sets[0].Nele))
        for i, d in enumerate(self.data_sets):
            ov_out_cl[i] = self.overlap_classes_ground_truth(d.dist_indices, k, nimg_cat, ncat,
                                                             jsubset)
        self.ov_gt = ov_out_cl
        return ov_out_cl

    def return_overlap_points_ll(self, step=1, k=10):
        ov_ll = np.empty((self.Nsets - 1, self.data_sets[0].Nele))
        l = 1
        for i in range(self.Nsets - step):
            ov_ll[i] = self.overlap_neighbourhoods_btw_layers(self.data_sets[i].dist_indices,
                                                              self.data_sets[i + step].dist_indices,
                                                              k)
        self.ov_ll = ov_ll
        return ov_ll

    def return_overlap_points_output(self, k=10):
        ov_out_pt = np.empty((self.Nsets, self.data_sets[0].Nele))
        for i, d in enumerate(self.data_sets):
            ov_out_pt[i] = self.overlap_neighbourhoods_btw_layers(d.dist_indices,
                                                                  self.data_sets[-1].dist_indices,
                                                                  k)
        self.ov_out = ov_out_pt
        return ov_out_pt
