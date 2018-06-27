import keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# Own Scripts
from UROP_Package.features_labels_setup import Features, Labels
from UROP_Package.reactions_list import denbigh_rxn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)


def create_random_features(total_number, species, save_name='features',save_mode=False):
    """
    Creates random values for con, time, and temperature
    :param total_number: Total number of feature examples.
    :param species: Total number of input chemical species. E.g: Ca, Cb, Cc ==> Species is 3
    :return: class Feature with required number of species and time and temperature.
    """
    return Features(np.random.random_sample((total_number, species)), np.random.random_sample((total_number, 1)),
                    np.random.random_sample((total_number, 1)), 'n', save_name=save_name,save_mode=save_mode)


def create_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, activation='relu',
                   optimizer='Adam', loss='mse', patience=4, reg_term=0, verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer', 'loss', 'patience',
             'reg_term', 'verbose']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, loss, patience, reg_term,
              verbose]
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

    def train_model_epochs_training_data_only(self, training_features, training_labels,
                                  save_name='DNN_training_only.h5', save_dir='./save/models/',
                                  plot_mode=False, save_mode=False):
        # Training model
        history = self.model.fit(training_features.n_features, training_labels.n_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def train_model_epochs(self, training_size, testing_size, validation_size, reaction_name=denbigh_rxn,
                           save_name='DNN_epochs.h5', save_dir='./save/models/', initial_species=1,
                           normalized_input_loading_mode=False, plot_mode=False, save_mode=False):
        if normalized_input_loading_mode:
            # If true, means training/testing/validation_size input is the concatenated features labels
            total_dim = training_size.shape[1]
            training_features = Features(training_size[:, :total_dim - 7], training_size[:, [total_dim - 7]],
                                         training_size[:, [total_dim - 6]], mode='n')
            training_labels = Labels(training_size[:, total_dim - 5:], 'n')

            testing_features = Features(testing_size[:, :total_dim - 7], testing_size[:, [total_dim - 7]],
                                        testing_size[:, [total_dim - 6]], mode='n')
            testing_labels = Labels(testing_size[:, total_dim - 5:], 'n')

            validation_features = Features(validation_size[:, :total_dim - 7], validation_size[:, [total_dim - 7]],
                                           validation_size[:, [total_dim - 6]], mode='n')
            validation_labels = Labels(validation_size[:, total_dim - 5:], 'n')

        else:
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
                                 validation_data=(testing_features.n_features, testing_labels.n_labels),
                                 verbose=self.hparams['verbose'])

        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)

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

    def train_model_earlystopping(self, training_size, testing_size, validation_size, reaction_name=denbigh_rxn,
                                  save_name='DNN_earlystopping.h5', save_dir='./save/models/', initial_species=1,
                                  normalized_input_loading_mode=False, plot_mode=False, save_mode=False,
                                  max_epochs=300):
        if normalized_input_loading_mode:
            # If true, means training/testing/validation_size input is the concatenated features labels
            total_dim = training_size.shape[1]
            training_features = Features(training_size[:, :total_dim - 7], training_size[:, [total_dim - 7]],
                                         training_size[:, [total_dim - 6]], mode='n')
            training_labels = Labels(training_size[:, total_dim - 5:], 'n')

            testing_features = Features(testing_size[:, :total_dim - 7], testing_size[:, [total_dim - 7]],
                                        testing_size[:, [total_dim - 6]], mode='n')
            testing_labels = Labels(testing_size[:, total_dim - 5:], 'n')

            validation_features = Features(validation_size[:, :total_dim - 7], validation_size[:, [total_dim - 7]],
                                           validation_size[:, [total_dim - 6]], mode='n')
            validation_labels = Labels(validation_size[:, total_dim - 5:], 'n')

        else:
            training_features = create_random_features(training_size, initial_species)
            training_labels = reaction_name(training_features)

            testing_features = create_random_features(testing_size, initial_species)
            testing_labels = reaction_name(testing_features)

            validation_features = create_random_features(validation_size, initial_species)
            validation_labels = reaction_name(validation_features)

        # Setting Up Early Stopping. Saving is done by the ModelCheckPoint callback
        if save_mode:
            callbacks = [EarlyStopping(monitor='val_loss', patience=self.hparams['patience']),
                         ModelCheckpoint(filepath=save_dir + save_name, monitor='val_loss', save_best_only=True)]
        else:
            callbacks = [EarlyStopping(monitor='val_loss', patience=self.hparams['patience'])]

        history = self.model.fit(training_features.n_features, training_labels.n_labels,
                                 epochs=max_epochs,
                                 batch_size=self.hparams['batch_size'],
                                 callbacks=callbacks,
                                 validation_data=(testing_features.n_features, testing_labels.n_labels),
                                 verbose=self.hparams['verbose'])
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

class DNN_classifer:
    def __init__(self, features_dim, hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = features_dim
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss='binary_crossentropy')

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
        model.add(Dense(1, activation='sigmoid'))
        print(pd.DataFrame(self.hparams))
        model.summary()
        return model

    def train_model_epochs_training_data_only(self, training_features, training_labels,
                                  save_name='cDNN_training_only.h5', save_dir='./save/models/', bins=[10,6,0],
                                  plot_mode=False, save_mode=False):
        # Training model
        history = self.model.fit(training_features.n_features, training_labels.binning(bins),
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

if __name__ == '__main__':
    # Testing
    training_features = create_random_features(1, 1)
    print(training_features.a_df())
    denbigh_rxn(training_features, plot_mode=True)
    x = denbigh_rxn(Features(8, 100, 350, 'a', single_example=True), plot_mode=True)
