# Procesamiento de Imágenes
Luciano Masuelli  
Facundo Gaviola  
## Trabajo Práctico 4
### Ejercicio 11

Para este practico se implementaron las siguientes funciones que seran utilizadas a lo largo de todo el practico:
```py
def show_image(image, title=None):
    """Display an image using matplotlib."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    
def calcular_dft_centrada(imagen_grayscale):
    dft = cv2.dft(np.float32(imagen_grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)
    return np.fft.fftshift(dft)
    
def mostrar_frecuencias(imagen_grayscale, dft_shift, mostrar_fase=False, titulo='', titulo_magnitud='Espectro de magnitud'):
    real = dft_shift[:, :, 0]
    imag = dft_shift[:, :, 1]
    magnitud = 20 * np.log1p(np.sqrt(real**2 + imag**2))

    if mostrar_fase:
        fase = np.angle(real + 1j * imag)

    n_subplots = 3 if mostrar_fase else 2
    plt.figure(figsize=(12, 4 if not mostrar_fase else 6))

    plt.subplot(1, n_subplots, 1)
    plt.imshow(imagen_grayscale, cmap='gray')
    plt.title('Imagen original' + (f' - {titulo}' if titulo else ''))
    plt.axis('off')

    plt.subplot(1, n_subplots, 2)
    plt.imshow(magnitud, cmap='gray')
    plt.title(titulo_magnitud)
    plt.axis('off')

    if mostrar_fase:
        plt.subplot(1, n_subplots, 3)
        plt.imshow(fase, cmap='gray')
        plt.title('Fase')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def reconstruir_imagen_idft(dft_shift, mostrar=False, titulo="Imagen Reconstruida"):
    # Paso 1: Desplazar frecuencias nuevamente
    dft_ishift = np.fft.ifftshift(dft_shift)

    # Paso 2: Aplicar IDFT
    img_back = cv2.idft(dft_ishift)

    # Paso 3: Obtener la magnitud (real e imaginario)
    img_recuperada = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Paso 4: Normalizar al rango [0, 255]
    img_recuperada = cv2.normalize(img_recuperada, None, 0, 255, cv2.NORM_MINMAX)
    img_recuperada = np.uint8(img_recuperada)

    # Mostrar si se solicita
    if mostrar:
        plt.figure(figsize=(5, 5))
        plt.imshow(img_recuperada, cmap='gray')
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return img_recuperada

def crear_mascara_frecuencia(shape, radio, tipo='pasa_bajas', forma='circular'):
    filas, columnas = shape
    centro_fila, centro_columna = filas // 2, columnas // 2

    Y, X = np.ogrid[:filas, :columnas]

    if forma == 'circular':
        distancia = np.sqrt((X - centro_columna)**2 + (Y - centro_fila)**2)
        if tipo == 'pasa_bajas':
            mascara = distancia <= radio
        elif tipo == 'pasa_altas':
            mascara = distancia > radio
        else:
            raise ValueError("Tipo de filtro debe ser 'pasa_bajas' o 'pasa_altas'")
    elif forma == 'cuadrada':
        mitad_lado = radio
        mascara = np.zeros((filas, columnas), dtype=bool)
        if tipo == 'pasa_bajas':
            mascara[
                centro_fila - mitad_lado:centro_fila + mitad_lado,
                centro_columna - mitad_lado:centro_columna + mitad_lado
            ] = True
        elif tipo == 'pasa_altas':
            mascara[:, :] = True
            mascara[
                centro_fila - mitad_lado:centro_fila + mitad_lado,
                centro_columna - mitad_lado:centro_columna + mitad_lado
            ] = False
        else:
            raise ValueError("Tipo de filtro debe ser 'pasa_bajas' o 'pasa_altas'")
    else:
        raise ValueError("Forma debe ser 'circular' o 'cuadrada'")

    # Repetimos para los dos canales (real, imaginario)
    return np.repeat(mascara[:, :, np.newaxis], 2, axis=2).astype(np.float32) 
```
--- 
**11. Transformada de Fourier inversa: Realiza la transformada de Fourier inversa para recuperar la imagen
original a partir de su version filtrada en el dominio frecuencial. Compara la imagen original con la
imagen recuperada.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#11).

Al realizar la transformada inversa se obtiene la imagen original con la cual se generaron las magnitudes de las 
frecuencias. Esto deberia ser siempre asi siempre y cuando no se hayan realizado alteraciones a las frecuencias.
Las imagenes onvolucradas son las siguientes:

![Imagen](/Entrega%202/Images/img1.png)
![Imagen](/Entrega%202/Images/img2.png)
![Imagen](/Entrega%202/Images/img3.png)
![Imagen](/Entrega%202/Images/img4.png)
---
**(a) ¿Como se visualiza la diferencia entre las frecuencias altas y bajas en una imagen? Ejercicio sugerido: Aplicar 
la Transformada de Fourier (DFT) y mostrar la magnitud del espectro centrado con fftshift.**

Se puede obervar una clara diferencia entre las imagenes posteriormente usadas en el dominio frecuencial. Por un lado, 
la imagen del bosque posee transiciones de colores mas suaves y no hay cambios o lineas tan marcas dentro de la imagen 
(las lineas que mas destacan son los troncos de los arboles). 

Esto se ve representado en el dominio frecuencial medinte una acumulacion de las frecuencias bajas en el centro de la 
imagen. Por el otro lado, en la otra imagen se puede apreciar que tiene muchas frecuencias altas ya que la imagen del 
paisaje tiene grandes lineas y bordes que marcan un cambio abrupto en el color de la imagen. Esto se ve representado en 
el espectro frecuencial como unos rayos o lineas que no se encuentran tan centrados en la region central de la 
representacion de las magnitudes frecuencias.  
---
**(b) ¿Que ocurre si eliminamos las componentes de alta frecuencia de una imagen? ¿Y si eliminamos
las de baja frecuencia? Ejercicio sugerido: Aplicar filtros pasa bajos y pasa altos en el dominio
de la frecuencia y reconstruir la imagen con la transformada inversa.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#b).

Al eliminar las componentes de alta frecuencia provocamos que la imagen se vuelva mas "suave" en el sentido de que se 
empizan a distorsionar algunos bordes o detalles mas finos de la imagen ya que estos representan los cambios mas bruscos 
dentro de la imagen. Generalmente esto se hace cuando se quiere crear un efecto de desenfoque, se quiere reducir el 
ruido o se quiere realizar una compresion de la imagen.

Por el otro lado, si eliminamos las frecuencias bajas de la imagen se van a mostrar principalmente los bordes y 
contornos de la misma con y sacando los colores dentro de los objetos. Se usa principalmente para marcar bordes o 
detalles dentro de la imagen.

Los resultados se pueden ver en las siguientes imagenes (Los resultados estan en escala de grises):

![Imagen](/Entrega%202/Images/baja_frec.webp)
![Imagen](/Entrega%202/Images/img5.png)
![Imagen](/Entrega%202/Images/img6.png)
---
**(c) ¿Que representa la fase de la transformada de Fourier de una imagen? ¿Que ocurre si se conserva
solo la fase o solo la magnitud? Ejercicio sugerido: Reconstruir una imagen usando solo la
magnitud y fase de otra imagen, intercambiar fase y magnitud entre dos imagenes distintas.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#c).

La fase es la encargada de codificar la localizacion espacial de los distintos objetos dentro de la imagen, que con 
ayuda de la magnitud, forman a la imagen final. A la hora de reconstruir una imagen solo con la fase se puede observar 
que la mayoria de las formas, contornos y bordes se mantienen y se puede observar cierta semenjanza con la imagen 
original pero con los colores y tonalidades totalmente alterados. Cuando reconstruimos utilizando solo la magnitud la 
reconstruccion es una imagen totalmente lisa y sin color.

Cuando realizamos el experimento de intercambiar la fase y magnitud de 2 imagenes obtenemos que cuando mantenemos la 
magnitud pero cambiamos la fase, la imagen resultante tendra los colores, intensidades y texturas de originales pero 
tomando las formas y contornos de la segunda imagen. Cuando hacemos el caso contrario y mantenemos la fase e 
intercambiamos la magnitud se mantienen a grandes rasgos los contornos y detalles pero los colores y tonalidades se ven 
alterados. 

En las siguientes imagenes se pueden ver la reconstruccion de la imagen original utilizando solo la fase y la magnitud:

![Imagen](/Entrega%202/Images/img7.png)

En las siguientes imagenes se pueden ver el resultado de intercambiar la fase y magnitud entre dos imagenes:

![Imagen](/Entrega%202/Images/img8.png)
![Imagen](/Entrega%202/Images/img9.png)
![Imagen](/Entrega%202/Images/img10.png)
![Imagen](/Entrega%202/Images/img11.png)
---
**(d) ¿Por que se centra la transformada de Fourier para su visualizacion? ¿Que efecto tiene? Ejercicio
sugerido: Mostrar el espectro de magnitud con y sin aplicar fftshift.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#d).

La transformada de fourier se centra para su visualizacion ya que por defecto la implementacion de la DFT coloca las 
frecuencias bajas en las esquinas y las frecuencias altas en el centro. Esto no es muy intuitivo a la hora de realizar 
el analisis de la imagen ya que al centrar las frecuencias bajas en el centro podemos ver de forma mas rapida e 
intuitiva si la imagen tiene predominio de frecuencias bajas o altas. Ademas, al centrar las frecuencias bajas en el 
centro, se facilita la aplicacion de mascaras o efectos visuales a las imagenes.

En las siguientes imagenes se puede ver el espectro de magnitud con y sin aplicar fftshift (Se utilizaron las imagenes
de la pregunta anterior para los espectros):

![Imagen](/Entrega%202/Images/img12.png)
![Imagen](/Entrega%202/Images/img13.png)
---
**(e) ¿Como se comporta la transformada de Fourier ante la traslacion o rotacion de una imagen?
Ejercicio sugerido: Aplicar una traslacion o rotacion y comparar los espectros de magnitud y
fase antes y despues.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#e).

A la hora de aplicar una traslacion se puede apreciar un cambio en la fase de la imagen en los cuales se pueden apreciar
ciertos patrones de ondas en las direccion en la cual se realizo la traslacion. La magnitud en teoria deberia mantenerse
de la misma manera que antes de la traslacion ya que no se ve afectado el contenido de la imagen. Sin embargo, al 
realizar la traslacion parte de la imagen sale del recuadro de la misma generando un espacio en negro y, por lo tanto, 
creando bordes y alteraciones a los colores.

Por el otro lado, cuando realizamos una rotacion se ven alteradas tanto la magnitud como la fase. La fase muestra de 
forma espacial hacia donde se realizo la rotacion dejando un patron mas marcado e intenso. Por su parte, la magnitud 
tambien muestra la rotacion aplicada a la imagen y en este caso, tambien incrementan las frecuencias altas debido a la 
aparicion de un fondo negro (y por lo tanto nuevos bordes) en aquellos lugares donde la imagen no esta presente debido 
a la rotacion.

Las siguientes imagenes muestran las imagenes con traslaciones y rotaciones y su respectivo espectro:

![Imagen](/Entrega%202/Images/img14.png)
![Imagen](/Entrega%202/Images/img15.png)
![Imagen](/Entrega%202/Images/img16.png)
---
**(f) ¿Como se refleja una estructura periodica en el dominio frecuencial? Ejercicio sugerido: Usar
imagenes sinteticas (rejillas, lıneas) y observar como se representan sus frecuencias dominantes.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#f).

A la hora de analizar el espacio frecuencial con imagenes que contengan patrones periodicos podemos decir que se pueden 
observar que tanto en la magnitud como en la fase vamos a encontrar patrones periodicos bien definidos. Dentro de estos 
patrones se pueden observar caracteristicas importantes como que tanto la fase como la magnitud poseen patrones bastante 
similares o en algunos casos practicamente identicos. Tambien es importante remarcar que se mantiene cierta 
proporcionalidad espacial entre la imagen original y el espacio frecuencial. Identificar estos patrones puede ser de 
gran ayuda a la hora de aplicar filtros o poder distinguir si una imagen ha tenido algun procesamiento previo.

Las siguientes imagenes muestran el espectro de magnitud y fase de unas imagenes sinteticas con un patron periodico:

![Imagen](/Entrega%202/Images/img17.png)
![Imagen](/Entrega%202/Images/img18.png)
![Imagen](/Entrega%202/Images/img19.png)
![Imagen](/Entrega%202/Images/img20.png)
---
**(g) ¿Que diferencias se observan en el espectro de imagenes suaves vs. imagenes con bordes pronunciados? Ejercicio 
sugerido: Comparar el espectro de una imagen desenfocada vs. la original con bordes definidos**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#g).

Se puede apreciar que cuando aplicamos un suavizado a una imagen con bordes marcadas, su espectro de magnitud pierde en 
gran medida las componentes altas y concentran mas las componenetes bajas. Esto se debe a que al suavizar toda la imagen 
los bordes de la misma ya no son tan abruptos y por lo tanto se representan con frecuencias mas bajas.

En las siguientes imagenes se puede ver el espectro de magnitud de una imagen con bordes marcados y su version suavizada:

![Imagen](/Entrega%202/Images/img21.png)
![Imagen](/Entrega%202/Images/img22.png)
---
**(h) ¿Que ocurre si aplicamos un filtro de forma circular o rectangular en el espectro? ¿Como cambia
la imagen? Ejercicio sugerido: Implementar mascaras ideales de paso bajo y paso alto circulares
y cuadradas y observar sus efectos.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#h).

Cuando aplicamos un filtro de pasa baja permitimos que pasen solo aquellas frecuencias ubicadas en el centro (bajas). 
Esto provoca que en la reconstruccion la imagen se vea borrosa o desenfocada pero mantiene a grandes rasgos la 
intensidad de colores.

Cuando aplicamos un filtro de pasa baja permitimos que pasen solo aquellas funciones de frecuencias ubicadas en los 
bordes (altas). Esto provoca que cunado reconstruyamos la imagen se resalten los bordes o cambios mas bruscos dentro de 
la imagen.

Es importante destacar que al aplicar mascaras que corten abruptamente las magnitudes pueden provocar artefactos en la 
imagen reconstruida. Es recomendable utilizar mascaras mas suaves.

En las siguientes imagenes podemos ver los resultados de aplicar filtros de pasa baja y pasa alta circulares:

![Imagen](/Entrega%202/Images/img23.png)
![Imagen](/Entrega%202/Images/img24.png)
![Imagen](/Entrega%202/Images/img25.png)
![Imagen](/Entrega%202/Images/img26.png)
---
**(i) ¿Cual es la relacion entre el patron de una imagen (orientacion, repeticion) y la simetrıa del
espectro? Ejercicio sugerido: Usar imagenes diagonales o repetitivas y analizar la simetrıa del
espectro.**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#i).

Se puede apreciar que en base a la orientacion de los patrones se puede determinar la orientacion de los picos en el 
dominio de frecuencia. Tambien es importante recalcar como se aprecia que en muchos casos la visualizacion de la 
magnitud tiene una orientacion contraria a la mostrada en la imagen. Por ejemplo si la imagen tiene un patron de lines 
horizontales, la magnitud va a tener un linea vertical. Tambien es importante remarcar que los patrones mantienen la 
simetria tanto en la imagen original como en la de magnitud.

Las siguientes imagenes muestran el espectro de magnitud y fase de unas imagenes sinteticas con un patron periodico:

![Imagen](/Entrega%202/Images/img27.png)
![Imagen](/Entrega%202/Images/img28.png)
![Imagen](/Entrega%202/Images/img29.png)
![Imagen](/Entrega%202/Images/img30.png)
---
**(j) ¿Como puede usarse el dominio frecuencial para eliminar ruido periodico en una imagen? Ejercicio sugerido: 
Introducir ruido periodico artificialmente y diseñar un filtro para suprimirlo en el dominio de la frecuencia**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#j).

Es posible eliminar el ruido repetitivo o periodico dentro de una imagen ya que este se puede ver reflejado en la 
magnitud en pequeños puntos brillantes repetitivos. Para esto es necesario crear una mascara o filtro capaz de 
identificar y posteriormente eliminar (caso ideal) o atenuar aquellas frecuencias asociadas al ruido. En este caso 
concreto utilizamos un filtro el cual lleva a 0 aquellos valores en donde encontramos ruido (ya que nosotros aplicamos 
el ruido, tambien sabemos donde se encuentra y que patron sigue).

Esto es muy util para poder recuperar la calidad de aquellas imagenes que poseean mucho ruido o simplemente se requiera 
de una gran precision y calidad de imagen. 

Las siguientes imageness muestran el proceso de introducir ruido periodico y posteriormente eliminarlo:

![Imagen](/Entrega%202/Images/img31.png)
![Imagen](/Entrega%202/Images/img32.png)
![Imagen](/Entrega%202/Images/img33.png)
![Imagen](/Entrega%202/Images/img34.png)
![Imagen](/Entrega%202/Images/img35.png)
---

