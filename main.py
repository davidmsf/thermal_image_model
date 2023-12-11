import tensorflow as tf
from PIL import Image
import scipy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def process_images():
    # Set up data generators for training and validation sets
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,  # normalize pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # for validation split
    )

    train_generator = train_datagen.flow_from_directory(
        '../leak_image_dataset/',  # replace with your dataset path
        target_size=(224, 224),  # or any size that you want
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
                '../leak_image_dataset/',  # same path as training
                target_size=(224, 224),
                batch_size=32,
                save_to_dir='aug',
                class_mode='binary',
                subset='validation'
    )

    validation_filenames = validation_generator.filenames
    validation_labels = validation_generator.classes
    for filename, label in zip(validation_filenames, validation_labels):
        print(filename, label)

    return train_generator, validation_generator


def train_model(train_generator, validation_generator):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    save_model(model)


def save_model(model):
    model.save('leak_img_1.model')


def load_model():
    return tf.keras.models.load_model('leak_img_1.model')


def predict_from_validation_generator(model, data):
    data, labels = data.next()  # Get the next batch
    predictions = model.predict(data)
    predicted_classes = (predictions > 0.5).astype("int32")
    print(predicted_classes, labels)

    # Plot the image and the prediction
    plt.imshow(data[21])  # Assuming the data generator yields batches with data in the correct image format
    plt.title(
        f"Predicted: {'No Leak' if predicted_classes[21][0] == 1 else 'Leak'}, Actual: {'No Leak' if labels[21] == 1 else 'Leak'}")
    plt.show()


def predict_from_img(model_predict, path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)/255.0
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model_predict.predict(img_batch)
    print(prediction[0][0])
    predicted_class = (prediction > 0.5).astype("int32")
    print(f"Predicted class: {'No Leak' if predicted_class == 1 else 'Leak'}")


    plt.imshow(img_array)
    plt.title(f"Prediction: {'No Leak' if predicted_class == 1 else 'Leak'}")
    plt.show()


if __name__ == '__main__':
    train_generator, validation_generator =
    process_images()
    train_model(train_generator, validation_generator)
    model = load_model()
    predict_from_validation_generator(model, validation_generator)
    predict_from_img(model, '../leak_image_dataset/5_Perfume.png')

