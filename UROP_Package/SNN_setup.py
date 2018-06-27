import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, Flatten, merge, Input, Lambda
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.metrics import confusion_matrix
import os, itertools
# Own Scripts
from UROP_Package.features_labels_setup import Features, Labels, feature_splitter
from UROP_Package.reactions_list import denbigh_rxn, denbigh_rxn2, create_class
from UROP_Package.DNN_setup import create_random_features, create_hparams, New_DNN


def create_SNN_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, activation='relu',
                       optimizer='Adam', loss='mse', patience=4, reg_term=0, verbose=1,
                       feature_vector_dim=10):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer', 'loss', 'patience',
             'reg_term', 'verbose',
             'feature_vector_dim']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, loss, patience, reg_term,
              verbose,
              feature_vector_dim]
    hparams = dict(zip(names, values))
    return hparams


class SNN:
    def __init__(self, SNN_hparams, features_dim=3):
        self.hparams = SNN_hparams
        self.features_dim = features_dim

        left_input = Input(shape=(features_dim,))
        right_input = Input(shape=(features_dim,))

        encoded_l = self.singlenet()(left_input)
        encoded_r = self.singlenet()(right_input)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(input=[left_input, right_input], output=prediction)

        if self.hparams.get('learning_rate', None) is None:
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd = optimizers.Adam(lr=self.hparams['learning_rate'])
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=sgd)

    def singlenet(self):
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        numel = len(hidden_layers)

        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        if numel > 1:
            for i in range(numel - 1):
                model.add(Dense(hidden_layers[i + 1],
                                activation=self.hparams['activation'],
                                kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.hparams.get('feature_vector_dim', 10), activation='sigmoid'))
        return model


class Siamese_loader:
    def __init__(self, model, x_train, x_val,bins=[10,6,0]):
        """
        :param model: SNN class model
        :param x_train: class list from create_class function
        :param x_val: same as above
        """
        self.model = model
        self.x_train = x_train
        self.features_dim = x_train[0][1].shape[1]  # First element in list, second element in tuple, no. of cols
        self.x_val = x_val
        self.n_classes = len(x_train)
        self.bins=bins
        if self.n_classes != len(x_val):
            # Ensure that both training and validation set has same number of classes
            raise ValueError(
                'WARNING! Number of classes in training set and validation set is not equal! Ensure they are equal.')

    def get_batch(self, batch_size, data='train'):
        if data == "train":
            x_train = self.x_train
        else:
            x_train = self.x_val
        n_classes = self.n_classes
        categories = rng.choice(n_classes, size=(batch_size,), replace=True)
        pairs = [np.zeros((batch_size, self.features_dim)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]  # Set the current category (aka class) for the loop
            n_examples = x_train[category][1].shape[0]  # Get no. of examples for current class in the loop
            idx_1 = rng.randint(0, n_examples)  # Choose one example out of the examples in the class
            pairs[0][i, :] = x_train[category][1][idx_1, :]  # Class, features, example, all input features
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1, n_classes)) % n_classes
                n_examples = x_train[category_2][1].shape[0]
            idx_2 = rng.randint(0, n_examples)
            pairs[1][i, :] = x_train[category_2][1][idx_2, :]
        return pairs, targets

    def generate(self, batch_size, data='train'):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size, data=data)
            yield (pairs, targets)

    def train(self, steps, batch_size, verbose=1, save_mode=False, save_dir='./save/models/',save_name='SNN.h5'):
        self.model.fit_generator(generator=self.generate(batch_size), steps_per_epoch=batch_size, epochs=steps,
                                 verbose=verbose)
        if save_mode:
            self.model.save(save_dir + save_name)

    def get_oneshot_predict_batch(self, category, x_predict):
        x_category = self.x_train[category][1]
        n_examples_category = x_category.shape[0]
        pairs=[]
        pairs.append(x_category)
        pairs.append(np.repeat(x_predict, n_examples_category, axis=0))
        return pairs

    def oneshot_predict(self, x_predict,compare_mode=True, print_mode=False, reaction=denbigh_rxn2):
        """

        :param x_predict: np array of no. examples x features_dim
        :param compare_mode: compare between predicted and actual labels
        :param reaction: reaction named used
        :return: Confusion matrix
        """
        n_examples_predict = x_predict.shape[0]
        support_set = self.x_train
        model = self.model
        predicted_class_store = []
        for j in range(n_examples_predict):
            predicted_labels_store = []
            for i in range(self.n_classes):
                pairs = self.get_oneshot_predict_batch(category=i, x_predict=x_predict[np.newaxis, j])
                predicted_labels = model.predict(pairs)
                predicted_labels_store.append(np.sum(predicted_labels) / support_set[i][1].shape[0])
            predicted_class_store.append(predicted_labels_store.index(max(predicted_labels_store)))
        if compare_mode:
            conc, t, T = feature_splitter(x_predict)
            labels = reaction(Features(conc, t, T, mode='n'))
            actual_class_store = labels.binning(self.bins)
            cm=confusion_matrix(actual_class_store,predicted_class_store)
            accuracy=np.count_nonzero(np.array(actual_class_store)-np.array(predicted_class_store)==0)/n_examples_predict*100
            if print_mode:
                df = pd.DataFrame(data=np.concatenate(
                    (np.array(predicted_class_store)[:, None], np.array(actual_class_store)[:, None]), axis=1),
                                  columns=['Predicted', 'Actual'])
                print('SNN accuracy : {}%'.format(accuracy))
                print(df)
                print(cm)
            return predicted_class_store, cm, accuracy
        else:
            return predicted_class_store