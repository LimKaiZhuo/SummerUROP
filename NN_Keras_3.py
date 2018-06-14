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
import os, itertools
# Own Scripts
from UROP_Package.features_labels_setup import Features, Labels
from UROP_Package.reactions_list import denbigh_rxn
from UROP_Package.DNN_setup import create_random_features, create_hparams, New_DNN
from UROP_Package.GAN_setup import create_GAN_hparams, create_normalized_feature_label, \
    save_new_training_testing_validation_examples, GAN, generating_new_examples_using_GAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)


def test_DNN():
    test = New_DNN(3, 5, create_hparams(hidden_layers=[30, 10], activation='relu'))
    test.train_model_earlystopping(500, 50, 10, plot_mode=True)


# test_DNN()

def test_GAN():
    GAN_hparams = create_GAN_hparams(epochs=1000, batch_size=64, generator_hidden_layers=[50, 50],
                                     discriminator_hidden_layers=[10, 10], generator_dropout=0.2,
                                     discriminator_dropout=0.2)
    test = GAN(8, 8, GAN_hparams)
    test.train_GAN(create_normalized_feature_label(40), plot_mode=True)


# test_GAN()

def meshgrid_GAN():
    linspace = np.arange(10, 200, 10)
    meshgrid = list(itertools.product(linspace, linspace))

    for i in range(len(meshgrid)):
        g_hl = meshgrid[i][0]
        d_hl = meshgrid[i][1]
        GAN_hparams = create_GAN_hparams(epochs=100, batch_size=64, generator_hidden_layers=[g_hl, g_hl],
                                         discriminator_hidden_layers=[d_hl, d_hl], generator_dropout=0.2,
                                         discriminator_dropout=0.2)
        test = GAN(8, 8, GAN_hparams)
        test.train_GAN(create_normalized_feature_label(1000), plot_mode=True)

        if i % 10 == 0:  # To clear Tensorflow computational graph after every 10 run to ensure Keras does not slow down
            K.backend.clear_session()


# meshgrid_GAN()


def training_DNN_and_GAN(training_size=40, testing_size=15, validation_size=15, features_dim=8,
                         numel_new_examples=50, set_number=3,
                         examples_dir='./save/concat_examples/',
                         training_name='training_examples.npy',
                         testing_name='testing_examples.npy',
                         validation_name='validation_examples.npy',
                         load_existing_data=False):
    store=[]
    for cnt in range(set_number):
        if not load_existing_data:
            # If load_existing_data is false, means need to create the 3 new examples file
            [training_examples, testing_examples, validation_examples] = save_new_training_testing_validation_examples(
                training_size, testing_size, validation_size)
        else:
            training_examples = np.load(examples_dir + training_name)
            testing_examples = np.load(examples_dir + testing_name)
            validation_examples = np.load(examples_dir + validation_name)

        # Running DNN with given dataset only
        DNN_hparams = create_hparams(patience=4, hidden_layers=[30, 10], activation='relu', verbose=0)
        real_DNN_model = New_DNN(3, 5, DNN_hparams)
        real_validation_loss = real_DNN_model.train_model_earlystopping(
            training_size=training_examples,
            testing_size=testing_examples,
            validation_size=validation_examples,
            normalized_input_loading_mode=True, plot_mode=True, save_mode=True, max_epochs=1000)

        # Create GAN Generator Model
        GAN_hparams = create_GAN_hparams(epochs=1000, activation='relu', batch_size=64,
                                         generator_hidden_layers=[50, 50],
                                         discriminator_hidden_layers=[10, 10],
                                         generator_dropout=0.2,
                                         discriminator_dropout=0.2)
        GAN_model = GAN(features_dim=features_dim, labels_dim=features_dim, GAN_hparams=GAN_hparams)

        # Use both training and testing, but not validation data to train GAN
        GAN_training_examples = np.concatenate((training_examples, testing_examples), axis=0)
        GAN_model.train_GAN(GAN_training_examples, save_name='GAN_generator.h5', save_mode=True, plot_mode=True,
                            show_plot=True)

        # Generating augmented examples
        augmented_training_examples = generating_new_examples_using_GAN(numel_new_examples=numel_new_examples,
                                                                        save_mode=True)

        # Training augmented DNN
        augmented_DNN_hparams = create_hparams(patience=4, hidden_layers=[30, 10], activation='relu', verbose=0)
        augmented_DNN_model = New_DNN(3, 5, augmented_DNN_hparams)
        augmented_validation_loss = augmented_DNN_model.train_model_earlystopping(
            training_size=augmented_training_examples,
            testing_size=testing_examples,
            validation_size=validation_examples,
            normalized_input_loading_mode=True, plot_mode=True, save_mode=True, max_epochs=1000)

        print('Real examples validation loss = {}, Augmented examples validation loss = {}' .format(
        real_validation_loss, augmented_validation_loss))
        store.append((real_validation_loss,augmented_validation_loss))

    # Writing stored validation losses to excel temp file.
    temp=pd.DataFrame(data=store,columns=['Real val loss','Augmented val loss'])
    writer = pd.ExcelWriter('temp.xlsx')
    temp.to_excel(writer,'Data')
    pd.DataFrame(DNN_hparams).to_excel(writer,'DNN')
    pd.DataFrame(GAN_hparams).to_excel(writer, 'GAN')
    writer.save()




training_DNN_and_GAN(training_size=10000,testing_size=100,validation_size=100, numel_new_examples=1000,set_number=7, load_existing_data=False)
