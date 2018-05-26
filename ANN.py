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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

def my_input_fn(features, targets, batch_size=1,num_epochs=None,shuffle=True):
    #Creating Dataset importing function, returning get_next from iterator
    #features=features.to_dict('list')

    ds=tf.data.Dataset.from_tensor_slices((features,targets))
    if shuffle:
        ds=ds.shuffle(buffer_size=10000)

    ds=ds.batch(batch_size).repeat(num_epochs)

    features,labels=ds.make_one_shot_iterator().get_next()
    return features,labels

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

    # Create input functions.

    features,labels=my_input_fn(training_examples,training_targets,batch_size=batch_size)
    predictions=nn(features,hidden_units)
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions,labels=labels))
    train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        #Training neural network
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss=0
            for _ in range(int(training_examples.shape[0]/batch_size)):
                _,loss_value=sess.run([train_op,loss])
                epoch_loss+=loss_value
            print('Epoch : ',epoch+1, ' out of ', epochs,' . Epoch loss = ',epoch_loss)

        #Comparing actual output conc vs predicted output conc using training dataset
        training_features,training_labels=my_input_fn(training_examples,training_targets,batch_size=training_examples.shape[0],shuffle=False)
        training_predictions=nn(training_features,hidden_units)
        validation_features, validation_labels = my_input_fn(validation_examples, validation_targets,batch_size=validation_examples.shape[0], shuffle=False)
        validation_predictions = nn(validation_features, hidden_units)
        sess.run(tf.global_variables_initializer())     #To initialise new prediction nodes
        training_predictions=trans_for_labels(pd.DataFrame(data=sess.run(training_predictions),columns=['TP_Ca','TP_Cb']),trans)
        validation_predictions = trans_for_labels(pd.DataFrame(data=sess.run(validation_predictions), columns=['VP_Ca', 'VP_Cb']), trans)
        #Converting Targets from normalized to actual concentrations
        training_targets=trans_for_labels(training_targets,trans)
        validation_targets = trans_for_labels(validation_targets, trans)
        #Printing out first 5 data
        print(training_targets.head())
        print(training_predictions.head())
        print(validation_targets.head())
        print(validation_predictions.head())


train_nn_regression_model(
        learning_rate=0.02,
        epochs=30,
        batch_size=50,
        hidden_units=[100],
        training_examples=training_features,
        training_targets=training_labels,
        validation_examples=validation_features,
        validation_targets=validation_labels)