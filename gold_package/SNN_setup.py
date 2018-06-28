import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, Flatten, merge, Input, Lambda, Concatenate
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle

# Own Scripts


def create_SNN_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, activation='relu',
                       optimizer='Adam', loss='mse', patience=4, reg_term=0, feature_vector_dim=10, dropout=0):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer', 'loss', 'patience',
             'reg_term', 'dropout'
             'feature_vector_dim']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, loss, patience, reg_term,
              dropout,
              feature_vector_dim]
    hparams = dict(zip(names, values))
    return hparams


class SNN:
    def __init__(self, SNN_hparams, features_c_dim, features_d_dim):
        self.hparams = SNN_hparams
        self.features_c_dim = features_c_dim
        self.features_d_dim = features_d_dim

        left_input_c = Input(shape=(features_c_dim,))
        left_input_d = Input(shape=(features_d_dim,))
        right_input_c = Input(shape=(features_c_dim,))
        right_input_d = Input(shape=(features_d_dim,))

        # Can add some layers before combining for future use.
        # Remember to change singlenet input dim if new layers is added before singlenet.

        left_combined = Concatenate()([left_input_c, left_input_d])
        right_combined = Concatenate()([right_input_c, right_input_d])

        encoded_l = self.singlenet()(left_combined)
        encoded_r = self.singlenet()(right_combined)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(input=[left_input_c, left_input_d, right_input_c, right_input_d], output=prediction)

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
                        input_dim=self.features_c_dim + self.features_d_dim,
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
    def __init__(self, model, x_train):
        """
        :param model: SNN class model
        :param x_train: Features_labels class containing training dataset.
        """
        self.model = model
        self.x_train = x_train
        self.n_examples = x_train.count
        self.features_c_norm = x_train.features_c_norm
        self.features_d_hot = x_train.features_d_hot
        self.labels = x_train.labels
        self.features_c_dim = self.features_c_norm[0][1].shape[1]
        self.features_d_dim = self.features_d_hot[0][1].shape[1]
        self.n_classes = x_train.n_classes

    def get_batch(self, batch_size):
        categories = rng.choice(self.n_classes, size=(batch_size,), replace=True)
        pairs = [[np.zeros((batch_size, self.features_c_dim)), np.zeros((batch_size, self.features_d_dim))] for i in
                 range(2)]
        pairs = [item for sublist in pairs for item in sublist]  # To flatten nested list to list
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]  # Set the current category (aka class) for the loop
            n_examples = self.features_c_norm[category][1].shape[0]  # Get no. of examples for current class in the loop
            idx_1 = rng.randint(0, n_examples)  # Choose one example out of the examples in the class
            # [left_c, left_d, right_c, right_d]
            pairs[0][i, :] = self.features_c_norm[category][1][idx_1, :]  # Class, features, example, all input features
            pairs[1][i, :] = self.features_d_hot[category][1][idx_1, :]
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1, self.n_classes)) % self.n_classes  # Randomly choose diff class
                n_examples = self.features_c_norm[category_2][1].shape[0]
            idx_2 = rng.randint(0, n_examples)
            pairs[2][i, :] = self.features_c_norm[category_2][1][idx_2, :]
            pairs[3][i, :] = self.features_d_hot[category_2][1][idx_2, :]
        return pairs, targets

    def generate(self, batch_size):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size)
            yield (pairs, targets)

    def train(self, steps, batch_size, verbose=1, save_mode=False, save_dir='./save/models/', save_name='SNN.h5'):
        self.model.fit_generator(generator=self.generate(batch_size), steps_per_epoch=batch_size, epochs=steps,
                                 verbose=verbose)
        if save_mode:
            self.model.save(save_dir + save_name)

    def get_oneshot_predict_batch(self, category, x_predict_c_norm, x_predict_d_hot):
        x_category_c_norm = self.features_c_norm[category][1]
        x_category_d_hot = self.features_d_hot[category][1]
        n_examples_category = x_category_c_norm.shape[0]
        pairs = []
        pairs.append(x_category_c_norm)
        pairs.append(x_category_d_hot)
        pairs.append(np.repeat(x_predict_c_norm, n_examples_category, axis=0))
        pairs.append(np.repeat(x_predict_d_hot, n_examples_category, axis=0))
        return pairs

    def oneshot_predict(self, x_predict_c_norm_a, x_predict_d_hot_a, compare_mode=False, print_mode=False):
        """

        :param x_predict: np array of no. examples x features_dim
        :param compare_mode: compare between predicted and actual labels
        :return: Confusion matrix
        """
        n_examples_predict = x_predict_c_norm_a.shape[0]
        model = self.model
        predicted_class_store = []
        for j in range(n_examples_predict):
            predicted_labels_store = []
            for i in range(self.n_classes): # For one example, check through all classes
                pairs = self.get_oneshot_predict_batch(category=i,
                                                       x_predict_c_norm=x_predict_c_norm_a[np.newaxis, j, :],
                                                       x_predict_d_hot=x_predict_d_hot_a[np.newaxis, j, :])
                # Vector of scores for one example against one class
                predicted_labels = model.predict(pairs)
                # Avg score for that one class
                n_class_example_count=self.features_c_norm[i][1].shape[0]
                predicted_labels_store.append(np.sum(predicted_labels) / n_class_example_count)
            # After checking through all classes, select the class with the highest avg score and store it
            predicted_class_store.append(predicted_labels_store.index(max(predicted_labels_store)))
        """
        if compare_mode:
            conc, t, T = feature_splitter(x_predict)
            labels = reaction(Features(conc, t, T, mode='n'))
            actual_class_store = labels.binning(self.bins)
            cm = confusion_matrix(actual_class_store, predicted_class_store)
            accuracy = np.count_nonzero(
                np.array(actual_class_store) - np.array(predicted_class_store) == 0) / n_examples_predict * 100
            if print_mode:
                df = pd.DataFrame(data=np.concatenate(
                    (np.array(predicted_class_store)[:, None], np.array(actual_class_store)[:, None]), axis=1),
                    columns=['Predicted', 'Actual'])
                print('SNN accuracy : {}%'.format(accuracy))
                print(df)
                print(cm)
            return predicted_class_store, cm, accuracy
        else:
        """
        return predicted_class_store

