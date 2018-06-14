import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from scipy.integrate import odeint
from tensorflow.python.data import Dataset
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format



trans=[[0.5,10],[50,100],[200,400]]

def transform_input_features_range(input_features,trans_info):
    '''''
    All range input as array of [lower,upper]
    Change concentration, time, temp, from range of 0 to 1 (from random generator) to desired range
    Purpose is to normalise all input features to 0 to 1 range for better neural network training
    From normalised to actual concentration
    '''''

    transformed_features=pd.DataFrame()
    for i in range(input_features.shape[1]):
        transformed_features[input_features.columns[i]] = input_features.iloc[:,i].apply(lambda x: x * (trans_info[i][1] - trans_info[i][0]) + trans_info[i][0])

    return transformed_features

def inverse_trans_for_labels(input_labels,trans_info):
    """""
    From actual concentration to normalised
    """""
    inverse_trans_labels=pd.DataFrame()
    for i in range(input_labels.shape[1]):
        inverse_trans_labels[input_labels.columns[i]]=input_labels.iloc[:,i].apply(lambda x: (x-trans_info[0][0])/(trans_info[0][1] - trans_info[0][0]))

    return inverse_trans_labels

def trans_for_labels(input_labels,trans_info):
    """""
    From normalised to actual concentration
    """""
    inverse_trans_labels=pd.DataFrame()
    for i in range(input_labels.shape[1]):
        inverse_trans_labels[input_labels.columns[i]]=input_labels.iloc[:,i].apply(lambda x: x * (trans_info[0][1] - trans_info[0][0]) + trans_info[0][0])

    return inverse_trans_labels

def reaction(c,t,E,T):
    '''''
    c : For input conc. in terms of [Ca,Cb]
    t : time
    E : Input constants for rate constant, [A1, A2, E1, E2]
    T : Isothermal Operating Temperature
    '''''
    ca=c[0]
    cb=c[1]
    A1=E[0]
    A2=E[1]
    E1=E[2]
    E2=E[3]
    k1=A1*np.exp(-E1/(8.314*T))
    k2=A2*np.exp(-E2/(8.314*T))
    dcadt=-k1*np.square(ca)+k2*cb
    dcbdt=k1*np.square(ca)-k2*cb
    return [dcadt,dcbdt]

def conc_out(input_features,trans,normalize=True):
    c_out=[]
    input_features=transform_input_features_range(input_features,trans)
    for x in range(input_features.shape[0]):
        ca0=input_features.iloc[x,0]
        t=input_features.iloc[x,1]
        T=input_features.iloc[x,2]

        # time points
        t = np.linspace(0, t, 1000)

        # solve ODEs
        c0 = [ca0, 0]
        c = odeint(reaction, c0, t, args=([10, 10, 10000, 15000], T))
        c_out.append(c[-1,:])
    c_out_df=pd.DataFrame(data=c_out,columns=['ca','cb'])
    if normalize:
        c_out_df=inverse_trans_for_labels(c_out_df,trans)
    return c_out_df

#Making Data
training_features=pd.DataFrame(data=np.random.random_sample([500,3]),columns=['ca','t','T'])
training_labels=conc_out(training_features,trans)
validation_features=pd.DataFrame(data=np.random.random_sample([30,3]),columns=['ca','t','T'])
validation_labels=conc_out(validation_features,trans)

def nn(input_featurs,hidden_layers=[10]):
    net=tf.layers.dense(input_featurs,hidden_layers[0],activation=tf.nn.sigmoid)
    if len(hidden_layers)>1:
        for i in range(len(hidden_layers)-1):
            net=tf.layers.dense(net,hidden_layers[i+1],activation=tf.nn.sigmoid)
    logits=tf.layers.dense(net,2,activation=None)
    return logits

def train_nn_regression_model(
        learning_rate,
        epochs,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):

    #Dataset API
    training_ds=tf.data.Dataset.from_tensor_slices((training_examples,training_targets)).shuffle(buffer_size=10000).repeat().batch(batch_size=batch_size)
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_examples, validation_targets)).batch(batch_size=validation_examples.shape[0]).repeat()
    iter=tf.data.Iterator.from_structure(training_ds.output_types,training_ds.output_shapes)
    features,labels=iter.get_next()
    training_init=iter.make_initializer(training_ds)
    validation_init=iter.make_initializer(validation_ds)

    # Create Optimisation Model
    predictions=nn(features,hidden_units)
    predictions_activation=predictions
    loss=tf.reduce_mean(tf.losses.mean_squared_error(labels=labels,predictions=predictions))
    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        #Training neural network
        sess.run(tf.global_variables_initializer())
        #Training Mode
        sess.run(training_init)
        for epoch in range(epochs):
            epoch_loss=0
            for _ in range(int(training_examples.shape[0]/batch_size)):
                _,loss_value=sess.run([train_op,loss])
                epoch_loss+=loss_value
            print('Epoch : ',epoch+1, ' out of ', epochs,' . Epoch loss = ',epoch_loss)
        # Validation Mode
        sess.run(validation_init)
        validation_labels=trans_for_labels(pd.DataFrame(data=sess.run(labels), columns=['Ca', 'Cb']),trans)
        validation_predictions=trans_for_labels(pd.DataFrame(data=sess.run(predictions_activation), columns=['VP_Ca', 'VP_Cb']),trans)
        validation_out=pd.concat([validation_labels,validation_predictions],axis=1,sort=False)
        print(validation_out)

train_nn_regression_model(
        learning_rate=0.002,
        epochs=200,
        batch_size=50,
        hidden_units=[10],
        training_examples=training_features,
        training_targets=training_labels,
        validation_examples=validation_features,
        validation_targets=validation_labels)
