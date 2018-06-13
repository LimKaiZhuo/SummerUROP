import keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers, optimizers
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
import os,itertools

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
                   optimizer='Adam', loss='mse', patience=4, reg_term=0):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer', 'loss', 'patience',
             'reg_term']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, loss, patience, reg_term]
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
        print(pd.DataFrame(self.hparams))
        model.summary()
        return model

    def train_model_epochs(self, reaction_name, training_size, testing_size, validation_size,
                           model_dir='./save/DNN_epochs.h5', initial_species=1, plot_mode=False):
        training_features = create_random_features(training_size, initial_species)
        training_labels = reaction_name(training_features)

        testing_features = create_random_features(testing_size, initial_species)
        testing_labels = reaction_name(testing_features)

        validation_features = create_random_features(validation_size, initial_species)
        validation_labels = reaction_name(validation_features)

        # Training model
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

        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

        return validation_loss

    def train_model_earlystopping(self, reaction_name, training_size, testing_size, validation_size,
                                  model_dir='./save/DNN_earlystopping.h5', initial_species=1, plot_mode=False):
        training_features = create_random_features(training_size, initial_species)
        training_labels = reaction_name(training_features)

        testing_features = create_random_features(testing_size, initial_species)
        testing_labels = reaction_name(testing_features)

        validation_features = create_random_features(validation_size, initial_species)
        validation_labels = reaction_name(validation_features)

        # Setting Up Early Stopping. Saving is done by the ModelCheckPoint callback
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.hparams['patience']),
                     ModelCheckpoint(filepath=model_dir, monitor='val_loss', save_best_only=True)]

        history = self.model.fit(training_features.n_features, training_labels.n_labels,
                                 epochs=300,
                                 batch_size=self.hparams['batch_size'],
                                 callbacks=callbacks,
                                 validation_data=(testing_features.n_features, testing_labels.n_labels))
        # Earlystopping loss value and epoch number
        best_loss_value = history.history['val_loss'][-self.hparams['patience'] - 1]
        best_loss_epoch = len(history.history['val_loss']) - self.hparams['patience']
        print('######## best_loss =', best_loss_value, ',', 'best_epoch =', best_loss_epoch, ' #######')

        # Prediction and Loss on Validation Set
        validation_predictions = Labels(self.model.predict(validation_features.n_features), 'n')
        validation_data = pd.concat(
            [validation_features.a_df(['Ca0']), validation_labels.a_df(['Ca', 'Cr', 'Ct', 'Cs', 'Cu']),
             validation_predictions.a_df(['Ca', 'Cr', 'Ct', 'Cs', 'Cu'], pre_string='P_')], axis=1, sort=False)
        print(validation_data)
        validation_loss = self.model.evaluate(validation_features.n_features, validation_labels.n_labels, verbose=0)
        print('validation_loss', validation_loss)

        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

        return validation_loss


def test_DNN():
    test = New_DNN(3, 5, create_hparams(hidden_layers=[30, 10], activation='relu'))
    test.train_model_earlystopping(denbigh_rxn, 500, 50, 10, plot_mode=True)


# test_DNN()

def create_GAN_hparams(generator_hidden_layers=[30, 30], discriminator_hidden_layers=[30, 30], learning_rate=None,
                       epochs=100, batch_size=32, activation='relu',
                       optimizer='Adam', loss='mse', patience=4, reg_term=0, generator_dropout=0,
                       discriminator_dropout=0):
    """
    Creates hparam dict for input into GAN class. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['generator_hidden_layers', 'discriminator_hidden_layers', 'learning_rate', 'epochs', 'batch_size',
             'activation', 'optimizer', 'loss', 'patience',
             'reg_term', 'generator_dropout', 'discriminator_dropout']
    values = [generator_hidden_layers, discriminator_hidden_layers, learning_rate, epochs, batch_size, activation,
              optimizer, loss, patience, reg_term, generator_dropout, discriminator_dropout]
    hparams = dict(zip(names, values))
    return hparams


def create_g_x(numel, reaction_name=denbigh_rxn, inital_species=1):
    """
    Creates training x to input into GAN.train_GAN() class method.
    :param numel: Number of training examples
    :param inital_species: Number of inital species. If 1, means only Ca0 present, and classes = 8. Ca0, t, T, and 5 others from denbigh rxn
    :return: np array of numel x classes matrix.
    """
    x_features = create_random_features(numel, inital_species)
    x_labels = reaction_name(x_features)
    return np.concatenate((x_features.n_features, x_labels.n_labels), axis=1)


class GAN:
    def __init__(self, features_dim, labels_dim, GAN_hparams):
        self.features_dim = features_dim
        self.labels_dim = labels_dim
        self.hparams = GAN_hparams

        if GAN_hparams['learning_rate'] is None:
            self.G = self.generator()
            self.G.compile(loss=self.hparams['loss'], optimizer=self.hparams['optimizer'])

            self.D = self.discriminator()
            self.D.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'], metrics=['accuracy'])

            self.stacked_generator_discriminator = self.stacked_generator_discriminator()
            self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd=optimizers.Adam(lr=GAN_hparams['learning_rate'])
            self.G = self.generator()
            self.G.compile(loss=self.hparams['loss'], optimizer=sgd)

            self.D = self.discriminator()
            self.D.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

            self.stacked_generator_discriminator = self.stacked_generator_discriminator()
            self.stacked_generator_discriminator.compile(loss='binary_crossentropy',
                                                         optimizer=sgd)

    def generator(self):
        # Set up Generator model
        generator_input_dim = self.features_dim
        model = Sequential()
        generator_hidden_layers = self.hparams['generator_hidden_layers']
        generator_dropout = self.hparams['generator_dropout']

        model.add(Dense(generator_hidden_layers[0],
                        input_dim=generator_input_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))

        numel = len(generator_hidden_layers)
        if numel > 1:
            for i in range(numel - 1):
                model.add(Dense(generator_hidden_layers[i + 1],
                                activation=self.hparams['activation'],
                                kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='linear'))

        return model

    def discriminator(self):
        # Set up Discriminator model
        discriminator_input_dim = self.features_dim
        model = Sequential()
        discriminator_hidden_layers = self.hparams['discriminator_hidden_layers']
        discriminator_dropout = self.hparams['discriminator_dropout']

        model.add(Dense(discriminator_hidden_layers[0],
                        input_dim=discriminator_input_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        if discriminator_dropout != 0:
            model.add(Dropout(discriminator_dropout))

        numel = len(discriminator_hidden_layers)
        if numel > 1:
            for i in range(numel - 1):
                model.add(Dense(discriminator_hidden_layers[i + 1],
                                activation=self.hparams['activation'],
                                kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def stacked_generator_discriminator(self):
        # Freeze discriminator weights and biases when training generator.
        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train_GAN(self, training_x, plot_mode=False):
        epochs = self.hparams['epochs']
        batch_size = self.hparams['batch_size']
        numel_rows = training_x.shape[0]
        d_loss_store = []
        g_loss_store = []
        plt.figure()
        plt.title('model loss / acc , (G,D) = (' + str(self.hparams['generator_hidden_layers'][0]) + ',' + str(
            self.hparams['discriminator_hidden_layers'][0]) + ')')
        plt.ylabel('loss / acc')
        plt.xlabel('epoch')

        for cnt in range(epochs):  # Epochs is more like number of steps here. 1 step ==> 1 gradient update
            # Training Discriminator
            # Half batch size for discriminator, since half real half fake data =>combine
            d_batch_size = int(batch_size / 2)
            idx = np.random.randint(0, numel_rows - d_batch_size)  # Index to start drawing x batch_x from training_x
            batch_x = training_x[idx:(idx + d_batch_size), :]  # Correct x
            batch_z = np.random.normal(0, 1, (d_batch_size, self.features_dim))  # Random noise z to feed into G
            batch_v = self.G.predict(batch_z)  # v = f(z)

            combined_x_v = np.concatenate((batch_x, batch_v), axis=0)
            combined_y = np.concatenate((np.ones((d_batch_size, 1)), np.zeros((d_batch_size, 1))), axis=0)

            d_loss = self.D.train_on_batch(combined_x_v, combined_y)  # Returns loss and accuracy
            d_loss_store.append(d_loss)

            # Training Generator using stacked generator, discriminator model
            batch_z = np.random.normal(0, 1, (batch_size, self.features_dim))  # Now is full batch size, not halved
            mislabelled_y = np.ones((batch_size, 1))  # y output all labelled as 1 so that G will train towards that

            g_loss = self.stacked_generator_discriminator.train_on_batch(batch_z, mislabelled_y)
            g_loss_store.append(g_loss)
            print('epoch: %d, [Discriminator :: d_loss: %f , d_acc: %f], [ Generator :: loss: %f]' % (
            cnt, d_loss[0], d_loss[1], g_loss))

        # Plotting
        if plot_mode:
            d_loss_store = np.array(d_loss_store)
            g_loss_store = np.array(g_loss_store)
            plt.plot(d_loss_store[:, 0])
            plt.plot(d_loss_store[:, 1])
            plt.plot(g_loss_store)
            plt.legend(['d_loss', 'd_acc', 'g_loss'], loc='upper left')
            plt.savefig('./plots/'+str(self.hparams['generator_hidden_layers'][0])+'_'+str(self.hparams['discriminator_hidden_layers'][0]), bbox_inches='tight')
            plt.clf()

        plt.close()


def test_GAN():
    GAN_hparams = create_GAN_hparams(epochs=1000, batch_size=64, generator_hidden_layers=[50, 50],
                                     discriminator_hidden_layers=[20, 20], generator_dropout=0.2,
                                     discriminator_dropout=0.2)
    test = GAN(8, 8, GAN_hparams)
    test.train_GAN(create_g_x(1000), plot_mode=True)


#test_GAN()

def meshgrid_GAN():
    linspace = np.arange(10, 200, 10)
    meshgrid = list(itertools.product(linspace,linspace))

    for i in range(len(meshgrid)):
        g_hl=meshgrid[i][0]
        d_hl = meshgrid[i][1]
        GAN_hparams = create_GAN_hparams(epochs=1000, batch_size=64, generator_hidden_layers=[g_hl,g_hl ],
                                         discriminator_hidden_layers=[d_hl,d_hl], generator_dropout=0.2,
                                         discriminator_dropout=0.2)
        test = GAN(8, 8, GAN_hparams)
        test.train_GAN(create_g_x(1000), plot_mode=True)

meshgrid_GAN()