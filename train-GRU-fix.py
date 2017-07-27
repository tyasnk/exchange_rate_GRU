# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:18:00 2017

@author: tyasnuurk
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
seed = 2017
tf.set_random_seed(seed)
np.random.seed(seed)


dataset = pd.read_csv('data/USDIDR-011014-310317.csv', 
                      #delimiter=',',
                      usecols=[3])

df = dataset.values
df = df.astype('float32')

len_train = int(len(df)*0.70)
len_val = int(len(df)*0.10)
#len_test = int(len(df)*0.20)
train = df[0:len_train]
validation = df[len_train:len_train+len_val]
test = df[len_train+len_val:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1)).fit(train)
train = scaler.transform(train)
validation = scaler.transform(validation)
test = scaler.transform(test)

#%%
#create time series from dataset
def create_sliding_windows(data,lag=1):
    X = [] 
    y = []
    for i in range(len(data)-lag-1):
        cek = data[i:(i+lag),0]
        X.append(cek)
        y.append(data[i+lag,0])
    return np.array(X), np.array(y)

lag = 3
X_train, y_train = create_sliding_windows(train,lag)
y_train = np.reshape(y_train,(len(y_train),1))
X_val, y_val = create_sliding_windows(validation,lag)
X_test, y_test = create_sliding_windows(test, lag)

#%%

num_hidden = 4
lr = 0.001

input_dim = 1
num_steps = lag
num_target = 1

batch_size = 128
training_epochs = 300
L2_coef = 0.0

weights = tf.Variable(tf.truncated_normal([num_hidden,num_target]),name="rnn/weight_out")
biases = tf.Variable(tf.zeros(num_target),name="rnn/bias_out")


def GRU(dataX,weight,bias,timesteps):
    dataX = tf.transpose(dataX,[1,0,2])
    data = tf.reshape(dataX,[-1,input_dim])
    H_split = tf.split(axis = 0, num_or_size_splits= timesteps, value = data)
    gru_cell = tf.contrib.rnn.GRUCell(num_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(gru_cell,H_split,dtype=tf.float32)
    o = tf.matmul(outputs[-1], weight) + bias
    return {'dataX':dataX,'data':data, 
            'H_split' : H_split,
            'outputs' : outputs, 
            'states' : states, 
            'final output' : o
            }


Z = tf.placeholder(tf.float32,[None,num_steps,input_dim])
y = tf.placeholder(tf.float32,[None,1])

model = GRU(Z,weights,biases,num_steps)
pred = model['final output']

cost = tf.losses.mean_squared_error(y,pred)

sum_of_weight = sum(tf.nn.l2_loss(w) for w in tf.trainable_variables() if not
                    ("bias" in w.name))

l2 = L2_coef*sum_of_weight
cost = cost+l2

optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

val_cost = tf.losses.mean_squared_error(y,pred)
#%%
X_valid = np.reshape(X_val,(X_val.shape[0],num_steps,input_dim))
y_valid = np.reshape(y_val,(len(y_val),1))

train_loss = []
saved_val_loss = []
saver = tf.train.Saver()
save_dir = 'checkpoints/'

import os

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'best_validation')
best_validation = 10000000
best_epoch = 0

X_y = np.concatenate((X_train,y_train),axis=1)

if X_y.shape[0] % batch_size > 0:
    minus = batch_size - X_y.shape[0] % batch_size
    added_data = np.random.permutation(X_y)
    extra_data = added_data[:minus]
    new_data = np.append(X_y,extra_data,axis=0)
else:
    new_data = X_y

new_data = np.random.permutation(new_data)

init = tf.global_variables_initializer()

print ("Start Training")
with tf.Session() as sess:
    sess.run(init)
    all_var = [tf_var for tf_var in tf.trainable_variables()]    
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int((len(new_data)/batch_size))
        ptr = 0
        
        for i in range(total_batch):         
            inp = new_data[ptr:ptr+batch_size,np.arange(lag)]
            out = new_data[ptr:ptr+batch_size,-1]
                 
            inp = np.reshape(inp,(batch_size,num_steps,input_dim))
            out = np.reshape(out,(batch_size,1))
                    
            start = int(round(time.time() * 1000))
            feeds = {Z:inp, y: out}
            sess.run(optimizer, feed_dict=feeds)
            ptr += batch_size
            avg_loss += sess.run(cost, feed_dict=feeds)
            end = int(round(time.time() * 1000))
        avg_loss/=total_batch
        print ("Epoch: %03d/%03d, loss: %.9f, time: %.0f second" % (epoch, training_epochs, avg_loss, 
                                                                float(end-start)/1000))
    
        train_loss.append(avg_loss)
        val_loss = sess.run(val_cost,feed_dict={Z:X_valid,y:y_valid
                                            })
        if val_loss < best_validation:
            best_validation = val_loss
            saver.save(sess=sess,save_path=save_path)
            best_epoch = epoch
        
        saved_val_loss.append(val_loss)
        print("val loss :%.10f"%(val_loss))
    variables_names = [v.name for v in tf.trainable_variables()]
    trained_var = sess.run(variables_names)
    
    rnndataX = sess.run(model['dataX'],feed_dict=feeds)
    rnndata = sess.run(model['data'],feed_dict=feeds)
    rnnHsplit = sess.run(model['H_split'],feed_dict=feeds)
    rnnStates = sess.run(model['states'],feed_dict=feeds)
    rnnOutputs = sess.run(model['outputs'],feed_dict=feeds)
    rnnFinalOutput = sess.run(model['final output'],feed_dict=feeds)
#%%
with tf.Session() as sess:
    
    saver.restore(sess=sess,save_path=save_path)
    y_pred_val = sess.run(pred,feed_dict={Z:X_valid})   
    y_pred_val = np.array(y_pred_val)
    y_pred_val = np.reshape(y_pred_val,(len(X_valid),1))

y_val = np.reshape(y_val,(-1,1))
y_pred_val = np.reshape(y_pred_val,(-1,1))

y_val_transform = scaler.inverse_transform(y_val)
y_pred_val_transform = scaler.inverse_transform(y_pred_val)
#%%
#predict
with tf.Session() as sess:
    saver.restore(sess=sess,save_path=save_path)
    X_test_1 = np.reshape(X_test,(X_test.shape[0],num_steps,input_dim))
    y_pred = sess.run(pred,feed_dict={Z:X_test_1})   
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred,(len(X_test),1))

y_test = np.reshape(y_test,(-1,1))
y_pred = np.reshape(y_pred,(-1,1))

y_test_transform = scaler.inverse_transform(y_test)
y_pred_transform = scaler.inverse_transform(y_pred)
simpan = pd.DataFrame(y_pred_transform)
simpan.to_csv('dinggoplot.csv',header=False)
#%%
plt.plot(train_loss)
plt.plot(saved_val_loss)
plt.title('model loss lr=%.3f, L2_coef = %.f'%(lr,L2_coef))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%%
plt.plot(y_pred_val_transform)
plt.plot(y_val_transform)
plt.title('validation')
plt.ylabel('value')
plt.xlabel('time')
plt.legend(['Prediksi Data Validasi', 'Data Validasi'], loc='upper left')
plt.show()
#%%
plt.plot(y_pred_transform)
plt.plot(y_test_transform)
plt.title('testing')
plt.ylabel('value')
plt.xlabel('time')
plt.legend(['Prediksi Data Uji', 'Data Uji'], loc='lower right')
plt.show()
        
#%%
def mean_absolute_error(targets,predictions):
    return np.mean(np.abs(targets-predictions))

def mean_absolute_percentage_error(targets,predictions): 
    return np.mean(np.abs((targets - predictions) / targets)) * 100

def rmse(targets,predictions):
    return np.sqrt(np.mean((targets-predictions) ** 2))

def mse(targets,predictions):
    return np.mean(((targets-predictions) ** 2))

def nmse(targets,predictions):
    target_mean = np.mean(targets)
    return np.mean(((targets-predictions) ** 2))/np.mean(((targets-target_mean) ** 2))

def dstat_measure(targets,predictions):    
    n = len(targets)
    alpha = 0
    for i in range(n-1):
        if(((predictions[i+1]-targets[i])*(targets[i+1]-targets[i]))>0):
            alpha += 1
    dstat = (1/n)*(alpha)*100
    return dstat

RMSE_Score_on_val = rmse(y_val_transform,y_pred_val_transform)
#NMSE_Score_on_val = nmse(y_val_transform,y_pred_val_transform)
MAE_Score_on_val = mean_absolute_error(y_val_transform,y_pred_val_transform)
MAPE_Score_on_val = mean_absolute_percentage_error(y_val_transform,y_pred_val_transform)
#Dstat_Score_on_val = dstat_measure(y_val_transform,y_pred_val_transform)

print("Validation   RMSE :  %.10f"%(RMSE_Score_on_val))
#print("Validation   NMSE :  %.10f"%(NMSE_Score_on_val))
print("Validation   MAE :   %.10f"%(MAE_Score_on_val))
print("Validation   MAPE :  %.10f"%(MAPE_Score_on_val))
#print("Validation Dstat: %.10f%%"%(Dstat_Score_on_val))

#%%
RMSE_onTest = rmse(y_test_transform,y_pred_transform)
MAE_onTest = mean_absolute_error(y_test_transform,y_pred_transform)
#NMSE_onTest = nmse(y_test_transform,y_pred_transform)
MAPE_onTest = mean_absolute_percentage_error(y_test_transform,y_pred_transform)
Dstat_onTest = dstat_measure(y_test_transform,y_pred_transform)

print("Test RMSE    : %.10f"%(RMSE_onTest))
#print("Test NMSE : %.10f"%(NMSE_onTest))
print("Test MAE : %.10f"%(MAE_onTest))
print("Test MAPE : %.10f"%(MAPE_onTest))
print("Test Dstat : %.10f%%"%(Dstat_onTest))
#%%
#def predict(lag=lag):
#    input_pred = np.arange(lag)
#    for i in input_pred:
#        if (i+1) - lag != 0:
#            input_pred[i] = input("nilai tukar %.0f hari sebelumnya : "%(lag-(i+1)))
#        else:
#            input_pred[i] = input("nilai tukar hari ini : ")
#    input_pred = scaler.transform(input_pred)
#    input_pred = np.reshape(input_pred,(1,lag,1))
#    with tf.Session() as sess:
#        saver.restore(sess=sess,save_path=save_path)
#        out_pred = sess.run(pred,feed_dict={X:input_pred})   
#        out_pred = np.array(out_pred)
#        #out_pred = np.reshape(out_pred,(len(input_pred),1))
#    out_pred = np.reshape(out_pred,(1,-1))
#    out_pred_transform = int(scaler.inverse_transform(out_pred))
#    return print("prediksi nilai tukar esok hari : %.0f"%(out_pred_transform))

#%%
from tkinter import *

fields = []
for i in range(lag):
    if (i+1) - lag != 0:
        fields.append("Masukkan nilai tukar %.0f hari sebelumnya : "%(lag-(i+1)))
    else:
        fields.append("Masukkan nilai tukar hari ini : ")
def fetch(entries):
    input_form = []
    for entry in entries:
        text  = entry[1].get()
        input_form.append(text)
    input_pred = np.array(input_form)
    input_pred = scaler.transform(input_pred)
    input_pred = np.reshape(input_pred,(1,lag,1))
    with tf.Session() as sess:
        saver.restore(sess=sess,save_path=save_path)
        out_pred = sess.run(pred,feed_dict={Z:input_pred})   
        out_pred = np.array(out_pred)
    out_pred = np.reshape(out_pred,(1,-1))
    out_pred_transform = int(scaler.inverse_transform(out_pred))
    textbox.insert(END,'Prediksi nilai tukar mata uang esok hari : %i \nTingkat Keyakinan Akurasi : %.4f%%' % 
                   (out_pred_transform,Dstat_onTest))
    return print(input_pred)
def makeform(root, fields):
    
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=32, text=field, anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__':
    root = Tk()
    root.wm_title('model GRU-RNN untuk peramalan nilai tukar mata uang Rupiah terhadap Dolar Amerika')
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
    b1 = Button(root, text='Prediksi',
                command=(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    textbox = Text(root,height=3)
    textbox.pack(side=RIGHT, padx=5, pady=5)   
    root.mainloop()