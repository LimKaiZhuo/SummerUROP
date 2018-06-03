import keras as K
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

# trans contain information for the transformation of concentration, time, and temperature respectively
trans = np.array([[0.5, 10], [50, 100], [250, 450]])


class Features:
    def __init__(self, input_conc, input_t, input_T, mode, trans=trans, single_example=False):
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
    def __init__(self, input_conc, mode, trans=trans):
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


def create_random_features(total_number, species):
    """
    Creates random values for con, time, and temperature
    :param total_number: Total number of feature examples.
    :param species: Total number of input chemical species. E.g: Ca, Cb, Cc ==> Species is 3
    :return: class Feature with required number of species and time and temperature.
    """
    return Features(np.random.random_sample((total_number, species)), np.random.random_sample((total_number, 1)),
                    np.random.random_sample((total_number, 1)), 'n')


def denbigh_rxn(input_features, A=[.2, 0.01, 0.005, 0.005], E=[7000, 3000, 3000, 100], plot_mode=False):
    '''

    :param input_features: Features class object containing features information. Last 2 column must be time and temperature.
    First n columns represent n species initial concentration
    :param A: Pre-Exponent factor for Arrhenius Equation
    :param E: Activation Energy
    :param plot_mode: Plot conc vs time graph for last set of features if = True
    :return: Label class containing output concentrations of A,R,T,S,U
    '''

    input_features = input_features.a_features
    numel_row = input_features.shape[0]
    numel_col = input_features.shape[1]
    conc = input_features[:, :numel_col - 2]
    c_out = []

    def reaction(c, t, T, A,
                 E):  # Rate equations for Denbigh reaction from Chemical Reaction Engineering, Levenspiel 3ed Chapter 8, page 194
        [Ca, Cr, Ct, Cs, Cu] = c
        [k1, k2, k3, k4] = A * np.exp(-np.array(E) / (8.314 * T))
        [dCadt, dCrdt, dCtdt, dCsdt, dCudt] = [-(k1 + k2) * Ca,
                                               k1 * Ca - (k3 + k4) * Cr,
                                               k2 * Ca,
                                               k3 * Cr,
                                               k4 * Cr]
        return [dCadt, dCrdt, dCtdt, dCsdt, dCudt]

    if numel_col < 7:  # If input_features has less than 7 columns, means not all species have non zero inital conc. Must add zero cols
        zeros = np.zeros((numel_row, 7 - numel_col))
        input_features = np.concatenate((conc, zeros, input_features[:, -2:]), axis=1)

    for i in range(numel_row):
        c0 = input_features[i, :-2]
        t = np.linspace(0, input_features[i, -2], 2000)
        T = input_features[i, -1]
        c = odeint(reaction, c0, t, args=(T, A, E))
        c_out.append(c[-1, :])

    if plot_mode:
        c = np.array(c)
        Ca, = plt.plot(t, c[:, 0], label='Ca')
        Cr, = plt.plot(t, c[:, 1], label='Cr')
        Ct, = plt.plot(t, c[:, 2], label='Ct')
        Cs, = plt.plot(t, c[:, 3], label='Cs')
        Cu, = plt.plot(t, c[:, 4], label='Cu')
        plt.legend(handles=[Ca, Cr, Ct, Cs, Cu])
        print('k Values:')
        print(A * np.exp(-np.array(E) / (8.314 * T)))
        print('Final Concentrations = ')
        print(c_out)
        plt.show()

    c_out = np.array(c_out)  # To convert list of numpy array to n x m numpy array
    c_out = Labels(c_out, 'a')
    return c_out


"""
training_features = create_random_features(1, 1)
print(training_features.a_df())
denbigh_rxn(training_features,plot_mode=True)
x=denbigh_rxn(Features(8,100,350,'a',single_example=True),plot_mode=True)
"""


def create_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, activation='relu',
                   optimizer='Adam', loss='mse', reg_term=0):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer', 'loss', 'reg_term']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, loss, reg_term]
    hparams = dict(zip(names, values))
    return hparams


class New_DNN:
    def __init__(self, features_dim, labels_dim, hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = features_dim
        self.labels_dim = labels_dim
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss=hparams['loss'])

    def create_DNN_model(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if numel > 1:
            for i in range(numel - 1):
                model.add(Dense(hidden_layers[i + 1],
                                activation=self.hparams['activation'],
                                kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='linear'))
        model.summary()
        return model

    def train_model_epochs(self, reaction_name, training_size, testing_size, validation_size,
                           model_dir='./save/DNN_epochs.h5', initial_species=1,plot_mode=False):
        training_features = create_random_features(training_size, initial_species)
        training_labels = reaction_name(training_features)

        testing_features = create_random_features(testing_size, initial_species)
        testing_labels = reaction_name(testing_features)

        validation_features = create_random_features(validation_size, initial_species)
        validation_labels = reaction_name(validation_features)

        history = self.model.fit(training_features.n_features, training_labels.n_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 validation_data=(testing_features.n_features, testing_labels.n_labels))

        # Saving Model
        self.model.save(model_dir)

        # Prediction and Loss on Validation Set
        validation_predictions = Labels(self.model.predict(validation_features.n_features), 'n')
        validation_data = pd.concat(
            [validation_features.a_df(['Ca0']), validation_labels.a_df(['Ca', 'Cr', 'Ct', 'Cs', 'Cu']),
             validation_predictions.a_df(['Ca', 'Cr', 'Ct', 'Cs', 'Cu'], pre_string='P_')], axis=1, sort=False)
        print(validation_data)
        validation_loss = self.model.evaluate(validation_features.n_features, validation_labels.n_labels, verbose=0)
        print(validation_loss)

        #Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

test = New_DNN(3, 5, create_hparams(hidden_layers=[30, 10]))
test.train_model_epochs(denbigh_rxn, 500, 50, 10,plot_mode=True)
