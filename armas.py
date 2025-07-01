
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Se usa un modelo preentrenado de YOLOv8 con nombre yolov8n.pt. 
# Si no esta descargado, se descargará automáticamente.
modelo = YOLO("yolov8n.pt")

#Funcion para procesar una imagen con el modelo anteriormente cargado
def procesar_imagen(image_path):
    img = cv2.imread(image_path)
    # Se configura un umbral de confianza de 70% (conf = 0.7)
    results = modelo.predict(img, conf=0.7)
    
    # Por cada elemento detectado, se dibuja un rectángulo alrededor del objeto y se muestra la etiqueta con la confianza.
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{modelo.names[int(box.cls)]}: {box.conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Se muestra la imagen con los objetos detectados.
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

