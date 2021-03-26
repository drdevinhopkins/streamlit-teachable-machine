import streamlit as st
from img_classification import teachable_machine_classification
import keras
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

import zipfile
import tempfile
import os
import tensorflow as tf


from PIL import Image

st.title('Teachable Machine to Streamlit')

stream = st.file_uploader('Teachable Machine model zip file', type='zip')
if stream is not None:
    myzipfile = zipfile.ZipFile(stream)
    with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()[0]  # e.g. "model.h5py"
        model_dir = os.path.join(tmp_dir, root_folder)
        # st.info(f'trying to load model from tmp dir {model_dir}...')
        model = tf.keras.models.load_model(model_dir)
        labels_df = pd.read_csv(
            tmp_dir+"/labels.txt", sep=" ", header=None, names=["index", "label"], index_col=0)
        st.write(labels_df.label.tolist())


uploaded_file = st.file_uploader(
    "Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, model)
    # if label == 0:
    #     st.write("CAT")
    # else:
    #     st.write("DOG")
    st.write(labels_df.loc[label]['label'])
