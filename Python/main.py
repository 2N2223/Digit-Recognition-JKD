import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import datasets
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Softmax
from keras.models import Sequential
# import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.version)
a = datasets.mnist.load_data()
print(a)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

print(train_images.shape)
print(train_labels)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = Sequential()
model.add(Conv2D(6, kernel_size=3, strides=1, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=5, strides=1, padding='valid'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, Softmax()])

tfjs.converters.save_keras_model(probability_model, 'JS_model')

print("Save JS Model Success!")
