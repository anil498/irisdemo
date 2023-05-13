# from flask import Flask
# app=Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, Flask!"

#//-----------for streamlit
# import streamlit as st
# st.write("Hello from Streamlit")


#//------------------ my code 
import streamlit as st
import pandas as pd
import numpy as np

from predicction import predict


st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into \
setosa, versicolor, virginica")

st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
 st.text('Sepal characteristics')
 sepal_l = st.slider("Sepal lenght (cm)", 1.0, 8.0, 0.5)
 sepal_w = st.slider("Sepal width (cm)", 2.0, 4.4, 0.5)
with col2:
 st.text('Pepal characteristics')
 petal_l = st.slider("Petal lenght (cm)", 1.0, 7.0, 0.5)
 petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)

 #st.button('Predict type of Iris')
 st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown('by anil ')
    
#///-------------nirbhay code
# import streamlit as st
# file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
# import cv2
# from PIL import Image, ImageOps
# import numpy as np
# st.set_option('deprecation.showfileUploaderEncoding', False)
# def import_and_predict(image_data, model):
    
#         size = (180,180)    
#         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#         image = np.asarray(image)
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
#         img_reshape = img[np.newaxis,...]
    
#         prediction = model.predict(img_reshape)
        
#         return prediction
# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions = import_and_predict(image, model)
#     score = tf.nn.softmax(predictions[0])
#     st.write(prediction)
#     st.write(score)
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )