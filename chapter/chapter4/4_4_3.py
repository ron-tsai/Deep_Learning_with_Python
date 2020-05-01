#引入数据；处理数据
from keras.datasets import imdb
import numpy as np
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train=vectorize_sequences(train_data)

x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32')

y_test=np.asarray(test_labels).astype('float32')

#搭建神经网络

from keras import models
from keras import layers


#原始模型
original_model=models.Sequential()
original_model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
original_model.add(layers.Dense(16,activation='relu'))
original_model.add(layers.Dense(1,activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])

original_hist=original_model.fit(x_train,y_train,
                                 epochs=20,
                                 batch_size=512,
                                 validation_data=(x_test,y_test))

#正则化模型
from keras import regularizers

l2_model=models.Sequential()
l2_model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
                          activation='relu',input_shape=(10000,)))
l2_model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1,activation='sigmoid'))

l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])
l2_model_hist=l2_model.fit(x_train,y_train,
                           epochs=20,
                           batch_size=512,
                           validation_data=(x_test,y_test))

#画图
epochs=range(1,21)

original_val_loss=original_hist.history['val_loss']
l2_model_val_loss=l2_model_hist.history['val_loss']

import matplotlib.pyplot as plt

plt.plot(epochs,original_val_loss,'b+',label='Original model')
plt.plot(epochs,l2_model_val_loss,'bo',label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
