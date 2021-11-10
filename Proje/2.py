import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
import numpy
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import os


cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)= cifar10.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


def create_model():
  model = models.Sequential()
  model.add(layers.Conv2D(40, (5, 5), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(60, (3, 3), activation='tanh'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(80, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((3, 3)))


  model.add(layers.Flatten())
  model.add(layers.Dense(80, activation='relu'))
  model.add(layers.Dense(20))


  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  return model

model = create_model()

checkpoint_path = "training_1/cp.ckpt"

model.load_weights(checkpoint_path)

# Re-evaluate the model

loss, acc = model.evaluate(x_test, y_test, verbose=2)
#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)
model.save('digits.model')
for x in range(1,7):
    img=cv.imread(f'{x}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
