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
##Grafica.

  ![Image](https://github.com/user-attachments/assets/709e3189-dc1e-44c0-af7f-42c05e17f283)

Estadisticas del ECG:
- Media: 0.0061 mV
- Desviacion estandar: 0.3507 mV
- coeficiente de variación: 5753.52 %

##Histograma.
 
Para la grafica del histograma y la densidad de probabilidad de la señal, se definió la función “calcular_estadisticas_y_histograma” que contiene los parámetros de senal,     canales y muestras_max, los otros comandos utilizados son:

- Plt.hist: genera un histograma de la amplitud de la señal. 
- Bins: divide el rango en los valores deseados.
- Density: muestra el conteo absoluto de muestras en cada intervalo. 
- Edgecolor: para especificar el contorno de las barras.
- Conteo: numero de datos en cada bin del histograma.
- Bordes: lista con los bordes de cada bin del histograma. 
- norm.pdf(bordes, media, desv_std): calcula la función de densidad de probabilidad normal (Pdf) con la media de deviación estándar de la señal.
- Plt.plot: Grafica la curva en rojo.
  
   ![Image](https://github.com/user-attachments/assets/e25fe8a4-9470-459b-aeba-8d4a6164577b)

## Ruidos y calculo SNR.

- La señal de ruido Se refiere a la proporción entre la potencia de una señal (información relevante) y la potencia del ruido de fondo (información irrelevante). Para la practica por cada se señal se calcularon dos amplitudes distintas.
  
- Para todos se calcula el SNR mediante la función “calcular_SNR” el primer parámetro defino en la función es la matriz con la señal original sin ruido, el segundo parámetro es la matriz con la señal con ruido agregado y el tercer parámetro es el número máximo de muestras a analizar.

```bash
def calcular_SNR(senal_original, senal_ruidosa, muestra_max):
lista_snr= []
for canal in range(senal_original.shape[1]):
orig = senal_original [:muestras_max, canal]
ruidosa = senal_ruidosa [:muestras_max, canal]
componente_ruido= ruidosa - orig
rms_senal = np.sqrt(np.mean(orig**2))
rms_ruido = np.sqrt(np.mean(componente_ruido**2))
snr:db = 20 * np.log10(rms_senal / rms_ruido)
lista_snr.append(snr_db)
 return lista_snr
```

- Se crea una lista una lista para almacenar el SNR calculado para cada canal, luego “senal_original.shape[1]” obtiene el número de canales y analiza cada canal por separado, “orig” se extraen los primeros valores de la señal original y “ruidosa” se extraen los valores de la señal con ruido. Se calcula el ruido tomando la señal ruidosa menos la original, y posteriormente el RMS que es un valor necesario para calcular el SNR. El SNR, calcula la relación señal ruido en decibeles usando la formula:


- Se almacena el dato obtenido y retorna el valor almacenado para posteriormente mostrarlo en pantalla con la función “generar_texto_SNR(lista_snr, canales)”

```bash
def generar_texto_SNR(lista_snr, canales):

texto= ""
for idx, canal in enumerate(canales):
    texto += f"{canal}:\nSNR = {lista_snr[idx]:.2f} dB\n\n"
return texto
``` 
## Gaussiano:

Se define la función "agregar_ruido_gaussiano", posteriormente se generan los siguientes pasos:

1.	Se recorre cada canal de la señal utilizando un ciclo for, accediendo a cada columna de la matriz senal.
2.	Se genera un vector de ruido ruido1 con distribución gaussiana de media 0 y desviación estándar amplitud_ruidoo, con un tamaño de muestras_max.
3.	Se agrega el ruido gaussiano a las primeras muestras_max posiciones de la señal en el canal correspondiente.
4.	La función devuelve la señal modificada con el ruido agregado.


```bash

def agregar_ruido_gaussiano(senal,muestras_max,amplitud_ruidoo=0.1):
    for canal in range(senal.shape[1]):
        ruido1=np.random.normal(0,amplitud_ruidoo, muestras_max)#media, desviacion estandar, el cero indica el canal de
        senal[:muestras_max, canal] += ruido1
        return senal
```
 
- El código agrega ruido gaussiano a una señal ECG con diferentes amplitudes (0.5 y 2.0), calcula la relación señal-ruido (SNR) y la muestra en pantalla. Finalmente, grafica la señal resultante con el ruido añadido para evaluar su impacto visualmente.

 ```bash
print("\n--- Ruido Gaussiano ---")
for amplitud in [1.0, 3.0]:  # 1.0: amplitud pequeña, 3.0: amplitud grande
    senal_gaussiano = senal.copy()  # Copia de la señal original
    senal_gaussiano = agregar_ruido_gaussiano(senal_gaussiano, muestras_max, amplitud_ruidoo=amplitud)
    snr_valores = calcular_SNR(senal, senal_gaussiano, muestras_max)
    texto_SNR = generar_texto_SNR(snr_valores, canales)
    print(f"SNR para Ruido Gaussiano (amplitud={amplitud}):")
    print(texto_SNR)
    graficar_senal(tiempo, senal_gaussiano, canales, tiempo_max,
    f"Señal ECG con Ruido Gaussiano (Amplitud={amplitud}) - Primeros 10 Segundos", texto_SNR)
```
 

 
 # Grafica con amplitud 0.5
![image](https://github.com/user-attachments/assets/e49eb09d-5c77-4795-873b-04f4b1644f86)

- El valor del SNR para la señal ECG con ruido Gausiano de amplitud 1.0 es de -8.93 dB, al ser un valor negativo el SNR significa que la señal de ruido es mayor 8,93 db a  la señal original.


# Grafica con amplitud 3.0
![image](https://github.com/user-attachments/assets/b8906e02-f51f-402c-b252-1afe77891363)

## Artefacto:
Se define la función “agregar_ruido_artefacto”, posteriormente se generan los siguientes pasos:

1.Se genera un vector de tiempo “t” con “muestras_max” puntos, desde 0 hasta 10 segundos y se divide en un intervalo de 0 a 10, tomando valores equidistantes.
2. Se define la frecuencia del ruido y se genera la variable “ruido”, generando un coseno desplazado
3. Se agrega el ruido a las señal en el canal correspondiente 

```bash
def agregar_ruido_artefacto(senal, muestra_max, amplitud= 1.0):
   t = np.linspace (0,10, muestra_max)
   f0= 5
   ruido=amplitud * np.cos(2*np.pi*f0*t - np.pi/2)
   for canal in range(senal.shape[1]):
       senal[:muestra_max, canal] += ruido
   return senal
```
- Posteriormente se llama la función anteriormente definida para ser graficada, tanto para una amplitud de 0,5 como para una amplitud de 2.0, modificando el parámetro de amplitud de la función, con la instrucción print se muestra en pantalla el valor del cálculo del SNR.

  ```bash
