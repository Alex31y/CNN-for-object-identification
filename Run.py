import glob
import PySimpleGUI as sg
from PIL import Image, ImageTk
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('C:/Users/39389/PycharmProjects/ML1/models/VGG16banana.h5')

def parse_folder(path):
    images = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png')
    return images


def prepare_image(path):
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

def load_image(path, window):
    try:
        image = Image.open(path)
        image = image.resize((400, 400))
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")


def main():
    elements = [
        [
            sg.Text("Punteggio: "),
            sg.Text(key='value')
        ],
        [sg.Image(key="image")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), enable_events=True, key="file"),
            sg.FolderBrowse(),
        ],
        [
            sg.Button("Prev"),
            sg.Button("Next"),
        ],
        [
            sg.Button("Banana"),
            sg.Button("NoBanana"),
            sg.Text(key='result'),
        ]
    ]
    window = sg.Window("Image Viewer", elements, size=(450, 600))
    images = []
    location, score = 0, 0
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "file":
            images = parse_folder(values["file"])
            if images:
                load_image(images[0], window)
        if event == "Next" and images:
            if location == len(images) - 1:
                location = 0
            else:
                location += 1
            load_image(images[location], window)
        if event == "Prev" and images:
            if location == 0:
                location = len(images) - 1
            else:
                location -= 1
            load_image(images[location], window)
        if event == "Banana" and images:
            prediction = model.predict(prepare_image(images[location]))
            prediction = np.argmax(prediction, axis=-1)
            if prediction == 0:
                score += 1
                window['result'].update("Hai indovinato! Era una banana")
            else:
                score -= 1
                window['result'].update("Sbagliato! Non era una banana")
            window['value'].update(score)
        if event == "NoBanana" and images:
            prediction = model.predict(prepare_image(images[location]))
            prediction = np.argmax(prediction, axis=-1)
            if prediction == 1:
                score += 1
                window['result'].update("Hai indovinato! Non era una banana")
            else:
                score -= 1
                window['result'].update("Sbagliato! Era una banana")
            window['value'].update(score)

    window.close()


if __name__ == "__main__":
    main()

