pip install tensorflow
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Load VGG16 without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
model.save("cifar10_vgg16_model.h5")
import cv2
img = cv2.imread("/content/testimg.png.jpeg")
img = cv2.resize(img, (32, 32))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Predicted class:", np.argmax(prediction))
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          epochs=1)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import the ImageDataGenerator class

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          epochs=1)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import the ImageDataGenerator class

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          epochs=1)