
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:


#define constants
#unrolled through 20 time steps
time_steps=200
#hidden LSTM units
num_units=128
#rows of 28 pixels
num_input=14
#learning rate for adam
learning_rate=0.002
#mnist is meant to be classified in 10 classes(0-9).
n_classes=1
#size of batch
batch_size=128


# In[ ]:


data = pd.read_csv('/Users/ajay/Downloads/LTSM/jena_climate_2009_2016.csv')
data.head()
Y = data['T (degC)']
X = data.drop('T (degC)',axis=1)
X = data.drop('Date Time',axis=1)
X.head()


# In[ ]:



a = 420551-120
X_train = X.head(a).values
Y_train = Y.head(a).values 
X_test = X.tail(120).values
Y_test = Y.tail(120).values


# In[ ]:


rand_start = np.random.randint(0,len(X_train)-time_steps)
Y1_data = Y_train[rand_start:rand_start+time_steps].reshape(-1,n_classes)


# In[ ]:


def next_batch(X_data,Y_data,time_steps):
    rand_start = np.random.randint(0,len(X_train)-time_steps)
    X1_data = X_data[rand_start:rand_start+time_steps].reshape(-1,time_steps,num_input) 
    Y1_data = Y_data[rand_start:rand_start+time_steps].reshape(-1,n_classes)
    return X1_data,Y1_data


# In[ ]:


#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,num_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])


# In[ ]:


input=tf.unstack(x ,time_steps,1)


# In[ ]:


#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")


# In[ ]:


#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias


# In[ ]:


#loss_function
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
loss = tf.reduce_sum(tf.square(prediction-y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



# In[ ]:


saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


# In[ ]:


#initialize variables
error = [] 
num_train_iterations = 5000
#
init=tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, Y_batch = next_batch(X_train,Y_train,time_steps)
        sess.run(opt, feed_dict={x: X_batch, y: Y_batch})
        
        if iteration % 100 == 0:
                
            mse = loss.eval(feed_dict={x: X_batch, y: Y_batch})
            print(iteration, "\tMSE:", mse)
            error.append(mse)
    # Save Model for Later
    saver.save(sess, "./ex_time_series_model")


# In[ ]:


plt.plot(error)

