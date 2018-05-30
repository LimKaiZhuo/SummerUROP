import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def features_n2a(input_features,trans):
    #Transforms features [Ca,t,T] from normalised values of [0,1] to actual values as specified in trans
    transformed_features = input_features*(trans[:,1]-trans[:,0])+trans[:,0]
    return transformed_features

def features_a2n(input_features,trans):
    #Transforms features [Ca,t,T] from actual values as specified in trans to normalised values of [0,1]
    transformed_features = (input_features-trans[:,0])/(trans[:,1]-trans[:,0])
    return transformed_features

def labels_a2n(input_labels,trans):
    """""
    Labels [Ca,Cb]: actual concentration to normalised
    """""
    transformed_labels=(input_labels-trans[0,0])/(trans[0,1]-trans[0,0])
    return transformed_labels

def labels_n2a(input_labels,trans_info):
    """""
    Labels [Ca,Cb]: normalised to actual concentration
    """""
    transformed_labels=input_labels*(trans[0,1]-trans[0,0])+trans[0,0]
    return transformed_labels

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

def conc_out(input_features,trans,normalized_inputs=True,normalize_outputs=True):
    """
    :param input_features: Normalised [Ca,t,T]
    :param trans: Range of values for Ca,t,T, 3x2 matrix
    :param normalize: Whether to normalise output label concentrations
    :return: Label concentrations
    """
    c_out=[]
    if normalized_inputs:
        input_features=features_n2a(input_features,trans)
    if input_features.ndim==2:
        for x in range(input_features.shape[0]):
            ca0=input_features[x,0]
            t=input_features[x,1]
            T=input_features[x,2]

            # time points
            t = np.linspace(0, t, 1000)

            # solve ODEs
            c0 = [ca0, 0]
            c = odeint(reaction, c0, t, args=([10, 10, 10000, 15000], T))
            c_out.append(c[-1,:])
        c_out=np.array(c_out)
        if normalize_outputs:
            c_out=labels_a2n(c_out,trans)
    elif input_features.ndim==1:
        ca0 = input_features[0]
        t = input_features[1]
        T = input_features[2]

        # time points
        t = np.linspace(0, t, 1000)

        # solve ODEs
        c0 = [ca0, 0]
        c = odeint(reaction, c0, t, args=([10, 10, 10000, 15000], T))
        c_out=c[-1, :]
        if normalize_outputs:
            c_out = labels_a2n(c_out, trans)

    return c_out

def train_nn_regression_model(
        learning_rate,
        epochs,
        batch_size,
        hidden_units,
        training_size,
        test_size,
        validation_size,
        trans,
        model_dir):

    #Prepping Data
    training_features = np.random.random_sample([training_size, 3])
    training_labels = conc_out(training_features, trans)
    #test_features = np.random.random_sample([test_size, 3])
    #test_labels = conc_out(test_features, trans)
    validation_features = np.random.random_sample([validation_size, 3])
    validation_labels = conc_out(validation_features, trans)

    #Creating Model
    model=Sequential()
    model.add(Dense(10,input_dim=3,activation='sigmoid'))
    model.add(Dense(2,activation=None))
    model.compile(optimizer='Adam',loss='mse')

    #Training Model
    model.fit(training_features,training_labels,epochs=epochs,batch_size=batch_size)

    #Comparing to Validation
    validation_predictions=model.predict(validation_features)
    validation_out=pd.DataFrame(data=np.concatenate([features_n2a(validation_features,trans),labels_n2a(validation_labels,trans),labels_n2a(validation_predictions,trans)],axis=1),columns=['Ca','t','T','A_Ca','A_Cb','P_Ca','P_Cb'])
    print(validation_out)

    #Saving Model
    model.save(model_dir)
    del model

trans = np.array([[0.5, 10], [50, 100], [200, 400]])
new_trans=np.array([[0.5, 10], [500, 1000], [200, 400]])

def run_train():
    #Function to train Keras Model
    train_nn_regression_model(
        learning_rate=0.002,
        epochs=200,
        batch_size=50,
        hidden_units=[30],
        training_size=2000,
        test_size=20,
        validation_size=30,
        trans=trans,
        model_dir='./save/model.h5')
#run_train()

def run_predictions(size,new_trans):
    #Function to try out prediction using trained Keras Model. New trans can adjust input features to be of a different range
    model=keras.models.load_model('./save/model.h5')
    x=np.random.random_sample([size,3])
    x_features=features_a2n(features_n2a(x,new_trans),trans)
    x_labels=conc_out(x_features,trans)
    x_predictions=model.predict(x_features)
    x_out = pd.DataFrame(data=np.concatenate(
        [features_n2a(x_features, trans), labels_n2a(x_labels, trans),
         labels_n2a(x_predictions, trans)], axis=1), columns=['Ca', 't', 'T', 'A_Ca', 'A_Cb', 'P_Ca', 'P_Cb'])
    print(x_out)
#run_predictions(100,trans)

#Part 2

def ODE_target_loss(features,target_Cb):
    labels=conc_out(features,trans,normalized_inputs=False,normalize_outputs=False)
    loss=abs(labels[0,1]-target_Cb)
    print('Features=',features,'Loss=',loss)
    return loss

def model_target_loss(features,target_Cb,model):
    features=features_a2n(features,trans)
    predictions=labels_n2a(model.predict(features),trans)
    loss=abs(predictions[0,1]-target_Cb)
    print('Features=',features_n2a(features,trans),'Predictions=',predictions[0,1],'Loss=',loss)
    return loss

def optimize_ODE_and_model(features, target_Cb,limits):
    """
    :param features: Incoming Ca together with initial t and T, must be one example at one time only, actual values NOT normalized
    :param target_Cb: Target output Cb NOT normalized
    :param limits: 2x2 matrix, [[lower t, upper t],[lower T,upper T]] NOT normalized
    :return:
    """
    if features.ndim==2:
        Ca0=features[0,0]
        t0=features[0,1]
        T0=features[0,2]
    elif features.ndim==1:
        Ca0=features[0]
        t0=features[1]
        T0=features[2]

    dimensions=[Real(low=limits[0,0],high=limits[1,0],name='t'),Real(low=limits[0,1],high=limits[1,1],name='T')]
    @use_named_args(dimensions=dimensions)
    def ODE_fitness(t,T):
        feature=np.array([[Ca0,t,T]])
        return ODE_target_loss(feature,target_Cb)

    search_result = gp_minimize(func=ODE_fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=40,
                                x0=[t0,T0])

    ODE_results=search_result.x


    @use_named_args(dimensions=dimensions)
    def model_fitness(t,T):
        feature=np.array([[Ca0,t,T]])
        return model_target_loss(feature,target_Cb,model)

    search_result = gp_minimize(func=model_fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=40,
                                x0=[t0,T0])

    model_results=search_result.x

    return [ODE_results,model_results]

def run_optimize_comparison():
    global model
    model=keras.models.load_model('./save/model.h5')
    x=optimize_ODE_and_model(np.array([8,50,144]),7,np.array([[50,100],[200,400]]))
    print('optimal ODE, Model ',x)
    print(model_target_loss([[8,x[0][0],x[0][1]]],7,model))

run_optimize_comparison()