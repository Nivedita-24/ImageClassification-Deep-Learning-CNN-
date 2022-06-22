
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('cnn_casava_leaves.hdf5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Cassava Leaf Classification
         """
         )

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (133,100)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_resize = (cv2.resize(img, dsize=(100, 133),    interpolation=cv2.INTER_CUBIC))/255.       
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Cassava Bacterial Blight (CBB)")
    elif np.argmax(prediction) == 1:
        st.write("Cassava Brown Streak Disease (CBSD)")
    elif np.argmax(prediction) == 2:
        st.write("Cassava Green Mottle (CGM)")
    elif np.argmax(prediction) == 3:
        st.write("Cassava Mosaic Disease (CMD)")      
    else:
        st.write("Healthy")
    
    st.text("Probability (0: CBB, 1: CBSD, 2: CGM, 3: CMD, 4: Healthy")
    st.write(prediction)
    
