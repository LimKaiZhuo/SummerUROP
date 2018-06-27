import numpy as np
import pandas as pd
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

# trans contain information for the transformation of concentration, time, and temperature respectively
trans = np.array([[0.5, 10], [50, 100], [250, 450]])


def feature_splitter(features):
    features_dim=features.shape[1]
    return features[:, 0:features_dim - 2], features[:, [features_dim - 2]], features[:, [features_dim - 1]]


class Features:
    def __init__(self, input_conc, input_t, input_T, mode, trans=np.array([[0.5, 10], [50, 100], [250, 450]]), single_example=False,save_name='features', save_mode=False):
        """
        To keep class of features which has both normalized and actual values as attributes
        All inputs must be numpy array. Ensure that conc, t, T have same number of rows!
        :param input_conc: n x species numpy array, where each columns represents values for Ca, Cb, and so on
        :param input_t: n x 1 numpy array, column represent t values
        :param input_T: n x 1 numpy array, column represent T values
        :param mode: String value, either 'a' or 'n' representing that the input is either in actual or normalized form
        :param trans: Transformation matrix, 3 x 2 numpy array, col 1 and 2 represents min, max values.
        :param single_example: If True, means input is 1 single example. Must convert single example into a 1x1 np array
        Row 1,2,3 represents conc, t, T values
        """
        if single_example:
            input_conc = np.array([[input_conc]])
            input_t = np.array([[input_t]])
            input_T = np.array([[input_T]])
        if mode == 'n':
            self.count = input_conc.shape[0]
            self.n_conc = input_conc
            self.n_t = input_t
            self.n_T = input_T
            self.n_features = np.concatenate((input_conc, input_t, input_T), axis=1)
            # Transforming Normalized to Actual Values
            self.a_conc = input_conc * (trans[0, 1] - trans[0, 0]) + trans[0, 0]
            self.a_t = input_t * (trans[1, 1] - trans[1, 0]) + trans[1, 0]
            self.a_T = input_T * (trans[2, 1] - trans[2, 0]) + trans[2, 0]
            self.a_features = np.concatenate((self.a_conc, self.a_t, self.a_T), axis=1)
        if mode == 'a':
            self.count = input_conc.shape[0]
            self.a_conc = input_conc
            self.a_t = input_t
            self.a_T = input_T
            self.a_features = np.concatenate((input_conc, input_t, input_T), axis=1)
            # Transforming Actual to Normalized Values
            self.n_conc = (input_conc - trans[0, 0]) / (trans[0, 1] - trans[0, 0])
            self.n_t = (input_t - trans[1, 0]) / (trans[1, 1] - trans[1, 0])
            self.n_T = (input_T - trans[2, 0]) / (trans[2, 1] - trans[2, 0])
            self.n_features = np.concatenate((self.n_conc, self.n_t, self.n_T), axis=1)
        if save_mode:
            file_path=open('./save/features_labels/'+save_name+'.obj','wb')
            pickle.dump(self,file_path)

    def a_df(self, conc_names=None, pre_string=None):
        """
        Converts actual features values from numpy matrix to dataframe
        :param conc_names: List of names in order of appearance in numpy matrix. If None, names will be 0,1,...
        :param pre_string: String infront of concentration.
        :return: Dataframe of actual features
        """
        columns = []
        for i in range(self.n_conc.shape[1]):
            if conc_names is None:
                if pre_string is not None:
                    columns.append(pre_string + 'C' + str(i))
                else:
                    columns.append('C' + str(i))
            else:
                if pre_string is not None:
                    columns.append(pre_string + conc_names[i])
                else:
                    columns.append(conc_names[i])
        columns.extend(['t', 'T'])
        return pd.DataFrame(data=self.a_features, columns=columns)


class Labels:
    def __init__(self, input_conc, mode, trans=np.array([[0.5, 10], [50, 100], [250, 450]])):
        """
        To keep class of features which has both normalized and actual values as attributes
        All inputs must be numpy array. Ensure that conc, t, T have same number of rows!
        :param input_conc: n x species numpy array, where each columns represents values for Ca, Cb, and so on
        :param mode: String value, either 'a' or 'n' representing that the input is either in actual or normalized form
        :param trans: Transformation matrix, 3 x 2 numpy array, col 1 and 2 represents min, max values.
        Row 1,2,3 represents conc, t, T values
        """
        if mode == 'n':
            self.n_labels = input_conc
            # Transforming Normalized to Actual Values
            self.a_labels = input_conc * (trans[0, 1] - trans[0, 0]) + trans[0, 0]
        if mode == 'a':
            self.a_labels = input_conc
            # Transforming Actual to Normalized Values
            self.n_labels = (input_conc - trans[0, 0]) / (trans[0, 1] - trans[0, 0])

    def a_df(self, conc_names=None, pre_string=None):
        """
        Converts actual labels values from numpy matrix to dataframe
        :param conc_names: List of names in order of appearance in numpy matrix. If None, names will be 0,1,...
        :param pre_string: String infront of concentration.
        :return: Dataframe of actual labels
        """
        columns = []
        for i in range(self.n_labels.shape[1]):
            if conc_names is None:
                if pre_string is not None:
                    columns.append(pre_string + 'C_out' + str(i))
                else:
                    columns.append('C_out' + str(i))
            else:
                if pre_string is not None:
                    columns.append(pre_string + conc_names[i])
                else:
                    columns.append(conc_names[i])
        return pd.DataFrame(data=self.a_labels, columns=columns)

    def binning(self,bins):
        targets = np.digitize(self.a_labels[:, 1], bins) - 1
        return targets
