
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Funcion para generar espectrogramas de audio y guardarlos como imágenes
def creear_espectograma(audio_path, save_path=None):
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    if save_path:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(save_path)
        plt.close()
    
    return S_db

# Modelado de red neuronal convolucional para clasificar sonidos peligrosos
def construir_modelo(input_shape=(128, 128, 3)):  # Espectrogramas como imágenes RGB
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Binario: peligroso/no peligroso
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Función para entrenar el modelo con los espectrogramas generados
def train_model(audio_files, labels):
    spectrograms = [creear_espectograma(f) for f in audio_files]
    X = np.array(spectrograms).reshape(-1, 128, 128, 1)
    X = np.repeat(X, 3, axis=-1)  # Convertir a "RGB" para compatibilidad
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = construir_modelo()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    return model

# Ejemplo de uso 
# audios = ["grito.wav", "disparo.wav", ...] # Lista de nombres de archivos de audio
# labels = [1, 1, 0, ...]  # 1 para sonido peligroso, 0 para sonido no peligroso
# model = train_model(audios, labels) 