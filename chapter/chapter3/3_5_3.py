from keras.datasets import reuters
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

#反向字典
word_index=reuters.get_word_index()
revers_word_index=dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire=''.join([revers_word_index.get(i-3,'?') for i in train_data[0]])

import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train=vectorize_sequences(train_data)
x_text=vectorize_sequences(test_data)

# def to_one_hot(labels,dimension=46):
#     results=np.zeros((len(labels),dimension))
#     for i,label in enumerate(labels):
#         results[i,label]=1.
#         print((i,label))
#         print(results)
#     return results
#
#
# one_hot_train_labels=to_one_hot(train_labels)
# one_hot_test_labels=to_one_hot(test_labels)
# print(one_hot_train_labels)
# print(one_hot_train_labels.shape)


#方法2

one_hot_train_labels=np.array(train_labels)
one_hot_test_labels=np.array(test_labels)

print(one_hot_train_labels)

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',
                       input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))#softmax事64个输出加总为1


model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#切数据

x_val=x_train[:1000]    #验证资料
partial_x_train=x_train[1000:]  #训练资料

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

#估计参数（每次丢512个）
history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=10,
                  batch_size=512,
                  validation_data=(x_val,y_val))

results=model.evaluate(x_text,one_hot_test_labels)

import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.show()


plt.clf()

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

prediction=model.predict_classes(x_text)

import pandas as pd
pd.crosstab(test_labels,prediction,
            rownames=['label'],colnames=['predict'])


from sklearn.metrics import classification_report
print(classification_report(test_labels,prediction))