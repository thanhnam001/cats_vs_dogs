import streamlit as st

st.title('Image classification')
st.header('Cats vs Dogs')
st.text('Upload images')

import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file, threshold):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = image_array#normalized_image_array

    # run the inference
    prediction = model.predict(data)
    prediction = 1 if prediction > threshold else 0
    return prediction #np.argmax(prediction) # return position of the highest probability

uploaded_file = st.file_uploader("Choose a cat or dog image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded imgage.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model.h5', 0.5)
    if label == 0:
        st.write("Cat")
    else:
        st.write("Dog")