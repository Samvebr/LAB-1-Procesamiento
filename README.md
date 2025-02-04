# LAB-1-Procesamiento
El análisis de señales biomédicas permite extraer información relevante mediante herramientas estadísticas. Estas señales contienen datos útiles, como amplitud y frecuencia, pero también ruido. En este laboratorio, se emplearán señales fisiológicas extraídas de la base de datos de Physionet, y empleando el compilador Python se calcularán sus estadísticos descriptivos (media, desviación estándar, coeficiente de variación, histograma y función de probabilidad) utilizando funciones predefinidas y programando desde cero. Además, se analizará la relación señal-ruido (SNR) al contaminarlas con distintos tipos de ruido (gaussiano, ruido de impulso y ruido de artefacto).

## Grafica de la señal y datos estadisticos.

- En la base de datos de Physionet se escogió la señal “a04.dat” y “a04.hea” del estudio Apnea-ECG Database, para que el código pueda leer correctamente los archivos es necesario que se encuentren dentro de la misma carpeta del proyecto.

- Posteriormente se agregaron las bibliotecas “wfdb”; lee los registros de las señales fisiológicas de formatos .dat y .hea, y extrae la frecuencia de muestreo y los nombres de los canales, “pandas”; se emplea para organizar datos el DataFrame, “matplotlib.pyplot”; graficar señales e histogramas., “numpy”; cálculos matemáticos y generación de ruido y “scipy.stats”; modelo estadístico y distribución normal.

```bash
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
```
- Para graficar la señal se empieza colocando el nombre del archivo en la variable nombre_registro y el tiempo que se desea a analizar y graficar, para nuestro caso 10s.

```bash
nombre_registo ='a04'
tiempo_max = 10 #Analiza los primeros 10 segundos
```
- Se lee la señal original mediante la función leer_senal(nombre_registro) y se estable el tiempo máximo de muestreo y se almacena en la variable muestras_max.

```bash
senal, fs, canales, tiempo, df= leer_senal(nombre_registro)
muestras_max = int(fs *tiempo_max)
```

- La función leer_senal(nombre_registro) permite leer los archivos .dat y. hea, obtiene la señal de la matriz, la frecuencia de muestreo, establecer los nombres de los canales, la duración total en segundos y el DataFrame que organiza los datos de la señal y los asigna a un dato de tiempo, además muestra los datos en pantalla de la frecuencia de muestro, la duración de la señal y los canales donde se encuentra la matriz de datos.

```bash
record = wfdb.rdrecord(record_name)  # Lee los archivos .dat y .hea
signal = record.p_signal  # Obtiene la señal en formato de matriz
fs = record.fs  # Frecuencia de muestreo
canales = record.sig_name  # Nombres de los canales (derivaciones)
duracion = len(signal) / fs  # Duración de la señal en segundos

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración de la señal: {duracion:.2f} segundos")
print(f"Canales disponibles: {canales}")

tiempo = [i / fs for i in range(len(signal))]  
df = pd.DataFrame(signal, columns=canales)  
df.insert(0, "Time (s)", tiempo)
return senal, fs, canales, tiempo, df

```
- Después se grafico los primeros 10 segundos de la señal empleando la función graficar señal, además se muestra en la gráfica los datos estadísticos de la señal, los cuales se obtienen a partir de la función calcular_estadisticas_texto que tiene como parámetros la señal y los primero 10 segundos y muestra en consola los datos estadísticos de la señal (media, desviación estándar, coeficiente de variación, histograma y función de probabilidad).

```bash
texto_estadisticas = calcular_estadisticas_texto(senal, muestras_max)
graficar_senal (tiempo, senal, cananles, tiempo_max, "Senal-ECG")
print( "Estadísticas de la señal original:")
print (texto_estadisticas)
```

- La función “graficar_senal” tiene como parámetros tiempo, senal, canales, tiempo_max, titulo, “texto_anotacion=None”, se establecen los parámetros y el tamaño de la muestra de la señal que se quiere graficar, luego se escoge el tamaño de la gráfica, el nombre de los ejes, el titulo del gráfico, la cuadrilla y se escribe la instrucción plt.show() para que se muestre el grafico.

```bash
def graficar_senal(tiempo,senal,canales,tiempo_max,titulo, texto_anotacion=None):

muestra_max = int(fs *tiempo_max)
tiempo_limitado = tiempo[:muestras_max]
senal_limitada = senal[:muestras_max, :]
fig, ax= plt.subplots(figsize=(12, 6))
for i, canal in enumerate(canales):
 ax.plot(tiempo_limitado, senal_limitada[:, i], label=canal)
ax.set_title(titulo)
ax.set_xlabel("tiempo (s)")
ax.set_ylabel("amplitud (mV)")
ax.legend()
ax.grid()
if texto_anotacion is not None:

  ax.text(0.02, 0.98, texto_anotacion, trasform=ax.transAxes, fontsize=10,
          verticalaligment=?'top', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()
```

## Calculos estadisticos.

- Media: se utiliza el comando mean y los datos del vector precedido por la librería numpy.
- Desviación estándar: se utiliza el comando std y los datos del vector precedido por la librería numpy.
- El coeficiente de variación: se calcula con las variables de desviación entre la media multiplicado por 100.
  
 Posteriormente se muestran los datos calculados en pantalla y se retorna la variable que los contiene, como se muestra a continuación:

 ```bash
def calcular_estadisticas_texto(senal, muestra_max):

texto=""
num_canales = senal.shape[1]
for canal in range(num_canales):
datos = senal[:muestras_max, canal]
media = np.mean(datos)
desviacion = np.std(datos)
coeff_var= (desviacion/media) * 100
texto += f"Canal {canal+1}:\nMedia = {media::2f} mV\nDesv = {desviacion:.2f} mV
return texto
```
  
