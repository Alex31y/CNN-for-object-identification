import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam

vgg16_model = tf.keras.applications.vgg16.VGG16()
# vgg16_model.summary()
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))

# banana training
train_path = "data/frutta/Banana/train"
preprocessing_function = tf.keras.applications.vgg16.preprocess_input

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224, 224), classes=['banana', 'notabanana'], batch_size=20)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 20, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img.astype(np.uint8))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


imgs, labels = next(train_batches)
plotImages(imgs)
print(labels)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, epochs=100, verbose=2)

model.save('models/VGG16banana.h5')
