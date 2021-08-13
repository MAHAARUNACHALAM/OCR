#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import streamlit as st


st.title('Meter Reading')

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
uploader=st.file_uploader('Please upload an Image')
# st.button('read')
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
img =cv2.imread(uploader)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 0, 2000, apertureSize=5)
# cv2.imshow('test', edges)

Oimg=[]

#image_ = Image.fromarray(np.uint8(img))
vertical_crop = img[0:img.shape[0],0:84]
Oimg.append(vertical_crop)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop)))

vertical_crop1 = img[0:img.shape[0], 80:160]
Oimg.append(vertical_crop1)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop1)))

vertical_crop3 = img[0:img.shape[0], 160:220]
Oimg.append(vertical_crop3)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop3)))

vertical_crop4 = img[0:img.shape[0], 220:280]
Oimg.append(vertical_crop4)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop4)))

vertical_crop5 = img[0:img.shape[0], 280:340]
Oimg.append(vertical_crop5)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop5)))

vertical_crop6 = img[0:img.shape[0], 360:420]
Oimg.append(vertical_crop6)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop6)))

vertical_crop7 = img[0:img.shape[0], 420:490]
Oimg.append(vertical_crop7)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop7)))

vertical_crop8 = img[0:img.shape[0], 490:550]
Oimg.append(vertical_crop8)
# Oimg.append(Image.fromarray(np.uint8(vertical_crop8)))

import datetime
import time
output=[]
# image = Image.open('5.jpeg')
for i in Oimg:
    #magee= Image.fromarray(np.uint8(i))
    st=str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    edges = cv2.Canny(i, 0, 2000, apertureSize=5)
        # since findContours affects the original image, we make a copy
    imagee= edges.copy()
    imagee= Image.fromarray(np.uint8(imagee))
    imagee.save(str(st)+".png")
    with open(str(st)+".png","rb") as img1:
        image = Image.open(img1).convert('RGB')
        
        # Replace this with the path to your image
#image2 = cv2.imread("WhatsApp Image 2021-08-09 at 3.00.36 PM.jpeg")

        
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
#         image.show()

        # Normalize the imageoooooo
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        print(prediction)
        
        j=0
        for i in prediction[0]:
            if(i>0.5):
                output.append(j)
            j=j+1
st.write(output)


# In[ ]:




