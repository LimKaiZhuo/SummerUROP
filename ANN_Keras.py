import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
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
        model_dir,
        reg_term=0,
        ):
    activation='relu'

    #Prepping Data
    training_features = np.random.random_sample([training_size, 3])
    training_labels = conc_out(training_features, trans)
    test_features = np.random.random_sample([test_size, 3])
    test_labels = conc_out(test_features, trans)
    validation_features = np.random.random_sample([validation_size, 3])
    validation_labels = conc_out(validation_features, trans)

    #Creating Model
    model=Sequential()
    model.add(Dense(hidden_units[0],input_dim=3,activation=activation,kernel_regularizer=regularizers.l2(reg_term)))
    if len(hidden_units)>1:
        for i in range(len(hidden_units)-1):
            model.add(Dense(hidden_units[i],activation=activation,kernel_regularizer=regularizers.l2(reg_term)))
    model.add(Dense(2,activation='linear'))
    model.compile(optimizer='Adam',loss='mse')

    #Setting Up Early Stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=4),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    #Training Model
    history=model.fit(training_features,training_labels,epochs=epochs,callbacks=callbacks,batch_size=batch_size,
              validation_data=(test_features,test_labels))
    best_loss_value=history.history['val_loss'][-5]
    best_loss_epoch=len(history.history['val_loss'])-4
    #Comparing to Validation
    validation_predictions=model.predict(validation_features)
    validation_out=pd.DataFrame(data=np.concatenate([features_n2a(validation_features,trans),labels_n2a(validation_labels,trans),labels_n2a(validation_predictions,trans)],axis=1),columns=['Ca','t','T','A_Ca','A_Cb','P_Ca','P_Cb'])
    print(validation_out)

    #Saving Model
    model.save(model_dir)
    del model
    return [best_loss_epoch,best_loss_value]

trans = np.array([[0.5, 10], [50, 100], [200, 400]])
new_trans=np.array([[0.5, 10], [500, 1000], [200, 400]])

def run_train(number_of_runs):
    #Function to train Keras Model
    run_store=[]
    for i in range(number_of_runs):
        single_run=train_nn_regression_model(
                    learning_rate=0.002,
                    epochs=300,
                    batch_size=50,
                    hidden_units=[30],
                    training_size=100,
                    test_size=100,
                    validation_size=100,
                    trans=trans,
                    reg_term=0.000,
                    model_dir='./save/model.h5')
        run_store.append(single_run)
    run_store=pd.DataFrame(data=run_store,columns=['Epochs','Val_Loss'])
    print(run_store)
    writer = pd.ExcelWriter('temp.xlsx')
    run_store.to_excel(writer)
    writer.save()
run_train(15)

def run_predictions(size,new_trans):
    #Function to try out prediction using trained Keras Model. New trans can adjust input features to be of a different range
    model=keras.models.load_model('./save/History/hp_2.h5')
    x=np.random.random_sample([size,3])
    x_features=features_a2n(features_n2a(x,new_trans),trans)
    x_labels=conc_out(x_features,trans)
    x_predictions=model.predict(x_features)
    x_out = pd.DataFrame(data=np.concatenate(
        [features_n2a(x_features, trans), labels_n2a(x_labels, trans),
         labels_n2a(x_predictions, trans)], axis=1), columns=['Ca', 't', 'T', 'A_Ca', 'A_Cb', 'P_Ca', 'P_Cb'])

    print(x_out)
#run_predictions(100,trans)

# Part 1b: Hyperparameter Tuning
def hyperparameter_tuning(total_run):
    hl_1 = Integer(low=5,high=800,name='hidden_layer_1')
    hl_2 = Integer(low=0,high=800,name='hidden_layer_2')
    hl_3 = Integer(low=0, high=800, name='hidden_layer_3')
    batch_size=Integer(low=60,high=80,name='batch_size')
    reg_term=Real(low=0,high=0.01,name='reg_term')
    activation = Categorical(categories=['relu'], name='activation')
    dimensions =[hl_1,hl_2,hl_3,batch_size,reg_term,activation]
    default_parameters = [30, 10, 0, 70, 0, 'relu']
    best_model_dir = './save/best_model.h5'
    global best_loss
    best_loss = 9999999.9999
    global run_count
    run_count=1

    training_features = np.random.random_sample([500, 3])
    training_labels = conc_out(training_features, trans)
    test_features = np.random.random_sample([100, 3])
    test_labels = conc_out(test_features, trans)

    @use_named_args(dimensions=dimensions)
    def fitness(hidden_layer_1,hidden_layer_2,hidden_layer_3,batch_size,reg_term,activation):
        global run_count
        print('Run Number :',run_count)
        run_count += 1
        print('hidden_layer_1 =', hidden_layer_1)
        print('hidden_layer_2 =', hidden_layer_2)
        print('hidden_layer_3 =', hidden_layer_3)
        print('batch_size =', batch_size)
        print('reg_term =', reg_term)
        print('activation =', activation)
        model = Sequential()
        model.add(
            Dense(hidden_layer_1, input_dim=3, activation=activation, kernel_regularizer=regularizers.l2(reg_term)))
        if hidden_layer_2>0:
            model.add(
                Dense(hidden_layer_2, input_dim=3, activation=activation, kernel_regularizer=regularizers.l2(reg_term)))
            if hidden_layer_3>0:
                model.add(
                    Dense(hidden_layer_3, input_dim=3, activation=activation, kernel_regularizer=regularizers.l2(reg_term)))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='Adam', loss='mse')

        history = model.fit(x=training_features,
                            y=training_labels,
                            epochs=100,
                            batch_size=batch_size,
                            validation_data=[test_features,test_labels],
                            verbose=0)
        loss = history.history['val_loss'][-1]
        print('loss = ',loss)
        global best_loss
        if loss < best_loss:
            model.save(best_model_dir)
            best_loss=loss
        del model
        keras.backend.clear_session()

        return loss

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=total_run,
                                x0=default_parameters)
    plot_convergence(search_result)
    print(pd.DataFrame(data=[search_result.x],columns=['hidden_layer_1','hidden_layer_2','hidden_layer_3','batch_size','reg_term','activation'],index=[0]))
    print('Best Loss =',search_result.fun)
    results=list(zip(search_result.func_vals, search_result.x_iters))
    new_results=[]
    for k in range(len(results)):
        new_results.append([results[k][0]]+results[k][1])
    results=pd.DataFrame.from_records(data=new_results,columns=['Loss','hidden_layer_1','hidden_layer_2','hidden_layer_3','batch_size','reg_term','activation'])
    writer = pd.ExcelWriter('Hyperparameters_Tuning.xlsx')
    results.to_excel(writer)
    writer.save()
    plt.show()

#hyperparameter_tuning(400)




#Part 2

def ODE_target_loss(features,target_Cb):
    labels=conc_out(features,trans,normalized_inputs=False,normalize_outputs=False)
    loss=abs(labels[0,1]-target_Cb)
    #print('Features=',features,'Loss=',loss)
    return loss

def model_target_loss(features,target_Cb,model):
    features=features_a2n(features,trans)
    predictions=labels_n2a(model.predict(features),trans)
    loss=abs(predictions[0,1]-target_Cb)
    #print('Features=',features_n2a(features,trans),'Predictions=',predictions[0,1],'Loss=',loss)
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
                                n_calls=20,
                                x0=[t0,T0])

    ODE_results=search_result.x


    @use_named_args(dimensions=dimensions)
    def model_fitness(t,T):
        feature=np.array([[Ca0,t,T]])
        return model_target_loss(feature,target_Cb,model)

    search_result = gp_minimize(func=model_fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=20,
                                x0=[t0,T0])

    model_results=search_result.x
    loss=np.square(ODE_results[0]-model_results[0])+np.square(ODE_results[1]-model_results[1])
    return features.tolist()+[target_Cb]+ODE_results+model_results+[loss]

def run_optimize_comparison(number_of_test):
    global model
    model=keras.models.load_model('./save/History/hp_5.h5')
    testing_examples=features_n2a(np.random.random_sample((number_of_test,3)),trans)
    target_Cb=np.random.random_sample(number_of_test)
    # Generates random number between 0 to 1, must multiply by Ca0 later on to get a target Cb
    output=[]
    for i in range(testing_examples.shape[0]):
        x=optimize_ODE_and_model(testing_examples[i,:],target_Cb[i]*testing_examples[i,0],np.array([[50,200],[100,400]]))
        print('Test Data:',i+1,'Output:',x)
        output.append(x)
    output=np.array(output)
    total_loss=np.sum(output[:,8])
    output=pd.DataFrame(data=output,columns=['Ca0','t','T','T_Cb','O_t','O_T','M_t','M_T','Loss'])
    writer=pd.ExcelWriter('Optimise.xlsx')
    output.to_excel(writer)
    writer.save()
    print(output)
    print(total_loss)

#run_optimize_comparison(300)
