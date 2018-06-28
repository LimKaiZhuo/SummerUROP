import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)


class Features_labels:
    def __init__(self, features_c, features_d, labels, save_name='fl', save_mode=False):
        """

        :param features_c: Continuous features. Np array, no. of examples x continous features
        :param features_d: Discrete or categorical features. Np array, no. of examples x dense categorical features
        :param labels: Labels as np array, no. of examples x 1
        :param save_name:
        :param save_mode:
        """
        # Setting up features
        self.count = features_c.shape[0]
        if self.count != features_d.shape[0]:
            raise ValueError('Number of examples in continuous features not same as discrete features.')

        def features_to_listedtuple(features, targets):
            dic = {}
            for feature, target in zip(np.ndarray.tolist(features), targets):
                if target in dic:  # Create new class tuple in the dic
                    dic[target].append(feature)
                else:  # If class already exists, append new features into that class
                    dic[target] = [feature]
            for target in dic:
                dic[target] = np.array(dic[target])
            # Convert from dictionary to list of tuple
            return sorted(dic.items())

        # _a at the back means it is a ndarray type
        self.features_c_a = features_c
        self.features_d_a = features_d
        # Without _a at the back means it is the listed tuple data type.
        self.features_c = features_to_listedtuple(features_c, labels)
        self.features_d = features_to_listedtuple(features_d, labels)
        # Normalizing continuous features
        self.scaler = MinMaxScaler()
        self.scaler.fit(features_c)  # Setting up scaler
        self.features_c_norm = self.scaler.transform(features_c)  # Normalizing features_c
        self.features_c_norm = features_to_listedtuple(self.features_c_norm, labels)
        # One-hot encoding for discrete features
        self.features_d_hot = to_categorical(features_d)
        self.features_d_hot = features_to_listedtuple(self.features_d_hot, labels)
        # Setting up labels
        self.labels = labels
        _, count = np.unique(labels, return_counts=True)
        self.n_classes = len(count)
        # Storing dimensions
        self.features_c_dim = features_c.shape[1]
        self.features_d_dense_dim = features_d.shape[1]
        self.features_d_hot_dim = self.features_d_hot[0][1].shape[1]

        # Saving
        if save_mode:
            file_path = open('./save/features_labels/' + save_name + '.obj', 'wb')
            pickle.dump(self, file_path)

    def generate_random_examples(self, numel):
        gen_features_c_norm_a = rng.random_sample((numel, self.features_c_dim))
        gen_features_c_a = self.scaler.inverse_transform(gen_features_c_norm_a)
        # Note: feature d selection currently only works if there is only 1 discrete feature.
        # Need to modify if there is more than 1 discrete feature!! Work in progress...
        gen_features_d_dense_a = rng.choice(self.features_d_a.flatten(), size=(numel,), replace=True)
        gen_features_d_hot_a = to_categorical(gen_features_d_dense_a)

        # Creating dic for SNN prediction
        gen_dic = {}
        gen_dic = dict(
            zip(('gen_features_c_a', 'gen_features_c_norm_a', 'gen_features_d_dense_a', 'gen_features_d_hot_a'),
                (gen_features_c_a, gen_features_c_norm_a, gen_features_d_dense_a, gen_features_d_hot_a)))
        return gen_dic

    def print_all(self):
        print('features_c : \n {} \n'
              'features_d : \n {} \n'
              'features_c_norm : \n {} \n'
              'features_d_hot : \n {} \n'
              'labels : \n {} \n'
              'dim of c, d_dense, d_hot = {} , {} , {}'.format(self.features_c, self.features_d,
                                                               self.features_c_norm,
                                                               self.features_d_hot,
                                                               self.labels, self.features_c_dim,
                                                               self.features_d_dense_dim, self.features_d_hot_dim))
