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

#搭建神经网络

from keras import models
from keras import layers

#创建dropout神经网络
dpt_model=models.Sequential()
dpt_model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16,activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1,activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

dpt_model_hist=dpt_model.fit(x_train,y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test,y_test))

#画图
epochs=range(1,21)

original_val_loss=original_hist.history['val_loss']
dpt_model_val_loss=dpt_model_hist.history['val_loss']

import matplotlib.pyplot as plt

plt.plot(epochs,original_val_loss,'b+',label='Original model')
plt.plot(epochs,dpt_model_val_loss,'bo',label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
