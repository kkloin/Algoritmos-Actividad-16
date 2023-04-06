# Librerias
import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo de TensorFlow
model = load_model('MobileNetSSD_deploy.prototxt.txt')

# Capturar el video
cap = cv2.VideoCapture(0)

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Expandir las dimensiones de la imagen para que coincidan con la entrada del modelo
    gray = cv2.resize(gray, (3, 90))
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    
    # Detectar objetos utilizando el modelo cargado
    objects = model.predict(gray)
    
    # Reducir las dimensiones de la salida del modelo para obtener las coordenadas de los objetos detectados
    objects = objects.reshape(-1, objects.shape[-1])

    # Dibujar una caja delimitadora alrededor de cada objeto detectado
    for obj in objects: 
        x, y, w, h = obj[:4]
        x = int(x * frame.shape[1] / 3)
        y = int(y * frame.shape[0] / 90)
        w = int(w * frame.shape[1] / 3)
        h = int(h * frame.shape[0] / 90)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 90, 255), 2)
    
    # Mostrar el video procesado en una ventana
    cv2.imshow('video', frame)
    
    # Esperar a que se presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la señal de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
