# Aplicación de sistemas inteligentes conectados por redes para la asistencia de seguridad publica en Bogotá

## Descripción
Este codigo fuente es un prototipo de aplicacion para el analisis de recursos de audio e imagenes
Se pretende con esto una opcion tecnológica que combina visión artificial y análisis de audio para detectar:
- Objetos peligrosos (armas blancas/de fuego)
- Comportamientos sospechosos
- Sonidos de riesgo (disparos, gritos)

## Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, Linux 64-bit o macOS
- **Hardware**:
  - CPU: 4 núcleos (Intel i5/Ryzen 5 o superior)
  - RAM: 4GB mínimo (8GB recomendado)
- **Python**: 3.8, 3.9 o 3.10
- **Dependencias**: ultralytics, matplotlib, cv2, librosa, numpy y tensorflow

## Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/CamiloRodriguezG/ProyectoSeguridad.git
cd ProyectoSeguridad
```
### 2. Instalacion de dependencias
```python
pip install opencv-python ultralytics matplotlib tensorflow
```
### 3. Configurar
Puede dirigirse directamente al archivo que desee probar y utilizar las funciones
Hay archivos de prueba en las carpetas /imgs y /audios
Luego podra ejecutar individualmente el archivo modificado.
