import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from PIL import Image

#Modelo preentrenado
model = InceptionV3(weights='imagenet')

# Cargar y preprocesar la imagen de entrada
img_path = 'carros.jpg' 
img = Image.open(img_path)  
img = img.resize((299, 299))  
x = np.array(img)  
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]


for label, description, probability in decoded_preds:
    print("Clase: {}, Descripción: {}, Probabilidad: {:.2f}%".format(label, description, probability*100))
