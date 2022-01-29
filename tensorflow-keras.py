import tensorflow as tf

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Unpacks dataset

# Normalize the dataset so the network can learn more easily
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # Sequential is most common model, there is also functional api and model subclassing https://keras.io/api/models/
model.add(tf.keras.layers.Flatten())
# Hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 units in the layer (128 neurons), second parameter is the activation function which is the rectify linear function (default basically)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output layer, will have less neurons than the hidden layers (probability distribution)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10) # Train model

# Validate that model learnt patterns
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save and load models
model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict([x_test])

import numpy as np

print(np.argmax(predictions[0])) # Prints the prediction for the first sample

import matplotlib.pyplot as plt

plt.imshow(x_test[0])
plt.show()

# Following code shows data that is passed into the neural network
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# print([0])