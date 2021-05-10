import tensorflow as tf

#keras is the higherlevel API when building neural networks
#using different functional API's of keras for layrs/model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test, y_test)= mnist.load_data()
print(x_train.shape)
print(y_train.shape)

#flattening as a single row
x_train= x_train.reshape(-1,28*28).astype("float32")/255.0
x_test= x_test.reshape(-1,28*28).astype("float32")/255.0

#sequantial API(one input, one output)
#sending a list of layers

model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512,activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

model=keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10))

#extracting specific layer features
model=keras.Model(inputs=model.inputs,
                  outputs=[model.get_layer('my_layer').output])
feature = model.predict(x_train)
print(feature.shape)

import sys
sys.exit()

#more information about the model
#rint(model.summary())

#Functional API(more flexible than sequential API)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model= keras.Model(inputs=inputs,outputs=outputs)


#configuring the training part of our network
#loss functions and optimizers
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),  #learningrate
    metrics=["accuracy"], #running accurracy so far
)


#specifies the concrete training of the network
model.fit( x_train, y_train, batch_size=32, epochs=5, verbose=2)  #batchsize o train on,verbose=2 for printing after each epoch
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

