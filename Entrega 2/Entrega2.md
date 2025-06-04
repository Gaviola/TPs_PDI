# Procesamiento de Imágenes Entrega 2
### **Luciano Masuelli**  
### **Facundo Gaviola**  
## Trabajo Práctico 4

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
>**11. Transformada de Fourier inversa: Realiza la transformada de Fourier inversa para recuperar la imagen
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
>**(a) ¿Como se visualiza la diferencia entre las frecuencias altas y bajas en una imagen? Ejercicio sugerido: Aplicar 
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
>**(b) ¿Que ocurre si eliminamos las componentes de alta frecuencia de una imagen? ¿Y si eliminamos
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
>**(c) ¿Que representa la fase de la transformada de Fourier de una imagen? ¿Que ocurre si se conserva
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
>**(d) ¿Por que se centra la transformada de Fourier para su visualizacion? ¿Que efecto tiene? Ejercicio
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
>**(e) ¿Como se comporta la transformada de Fourier ante la traslacion o rotacion de una imagen?
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
>**(f) ¿Como se refleja una estructura periodica en el dominio frecuencial? Ejercicio sugerido: Usar
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
>**(g) ¿Que diferencias se observan en el espectro de imagenes suaves vs. imagenes con bordes pronunciados? Ejercicio 
sugerido: Comparar el espectro de una imagen desenfocada vs. la original con bordes definidos**

Se puede ejecutar en el siguiente [Notebook](./TP4/TP4.ipynb#g).

Se puede apreciar que cuando aplicamos un suavizado a una imagen con bordes marcadas, su espectro de magnitud pierde en 
gran medida las componentes altas y concentran mas las componenetes bajas. Esto se debe a que al suavizar toda la imagen 
los bordes de la misma ya no son tan abruptos y por lo tanto se representan con frecuencias mas bajas.

En las siguientes imagenes se puede ver el espectro de magnitud de una imagen con bordes marcados y su version suavizada:

![Imagen](/Entrega%202/Images/img21.png)
![Imagen](/Entrega%202/Images/img22.png)
---
>**(h) ¿Que ocurre si aplicamos un filtro de forma circular o rectangular en el espectro? ¿Como cambia
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
>**(i) ¿Cual es la relacion entre el patron de una imagen (orientacion, repeticion) y la simetrıa del
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
>**(j) ¿Como puede usarse el dominio frecuencial para eliminar ruido periodico en una imagen? Ejercicio sugerido: 
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
## Trabaja Práctico 5
### Ejercicio 3
>**Umbralización híbrida (combinación de Otsu + morfología). ¿Cómo mejorar la segmentación de objetos con ruido o regiones conectadas? Práctica sugerida: Aplicar Otsu, luego refinar con cv2.morphologyEx() (apertura o cierre).**

Se puede ejecutar en el siguiente [Notebook](./TP5/tp5.ipynb#3).

La siguiente función es la encargada de binarizar la imagen con Otsu y luego aplicar apertura y cierre para realizar una comparación de ambas técnicas.
```py
def ej_3(img,kernel_size=3):
    _, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    cierre = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
    apertura = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
    return img_otsu, cierre, apertura
```
A continucación se muestran los resultados en dos ejemplos diferentes 
![alt text](/Entrega%202/Images/img36.png)
![alt text](/Entrega%202/Images/img37.png)

Se observa cómo luego de aplicar Otsu para la binarización de la imagen del paisaje, si se quisiera segmentar para diferenciar el lago, aplicando apertura se unen los huecos permitiendo distinguir el agua de las demás.  
Para el ejemplo del círculo, si consideramos los circulos interiores como ruido, observamos que con una clausura se une casi toda la región.  

---

### Ejercicio 4
>**Segmentación por detección de bordes. ¿Cómo se puede usar la información de bordes para segmentar una imagen? Práctica sugerida: Detectar bordes con cv2.Canny() o skimage.filters.sobel, luego aplicar umbral y cerrar regiones con morfología.**

Se puede ejecutar en el siguiente [Notebook](./TP5/tp5.ipynb#4).

La información de bordes es fundamental para la segmentación de imágenes, ya que los bordes suelen marcar las fronteras naturales entre objetos o regiones con diferentes propiedades visuales, como intensidad, color o textura.

Cuando se detectan los bordes en una imagen, por ejemplo usando el algoritmo de Canny, se obtiene una representación binaria donde los píxeles que forman parte de un borde tienen un valor alto (generalmente 255) y el resto son cero. Esta imagen de bordes no segmenta por sí sola, pero puede guiar o asistir a los algoritmos de segmentación, aportando información estructural clave.

Una forma de utilizar esta información es como entrada para métodos como Watershed, donde los bordes pueden actuar como barreras que impiden la fusión de regiones durante la expansión del algoritmo. También pueden servir como base para generar máscaras o semillas que indiquen con más precisión dónde se encuentran los objetos de interés y sus límites.  


En el ejercicio se aplicó Canny para la obtención de bordes para luego aplicar clausura, uniendo las regiones y consiguiendo aproximar una máscara para segmentación.  

```py
def ej_4(img,kernel_size=5):
    bordes = cv2.Canny(img, 100, 200)
    _, img_otsu = cv2.threshold(bordes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size,kernel_size), np.uint8)

    cierre = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
    
    return bordes,img_otsu, cierre
```

![alt text](/Entrega%202/Images/img38.png)

---

### Ejercicio 8
>**Segmentación basada en regiones (crecimiento o split-merge). ¿Cómo se puede segmentar una
imagen expandiendo regiones homogéneas? Práctica sugerida: Usar skimage.segmentation.flood() o
flood fill() para realizar crecimiento de regiones desde semillas.**

Se puede ejecutar en el siguiente [Notebook](./TP5/tp5.ipynb#8).

Una imagen se puede segmentar de esta manera partiendo de un punto o semilla en la imágen dentro del objeto a segmentar, el algoritmo se encarga de ir recorriendo los pixeles adyacentes y clasificarlos como parte del objeto siempre que sean similares a los anteriores y hasta llegar a algún pixel que se encuentre fuera del objeto, en este momento seguirá un camino distinto hasta haber encontrado todos los pixeles pertenecientes a este.  
Para este ejericico se utilizó `skimage.segmentation.flood()` que retorna una matriz booleana del tamaño de la imagen original con 0 (ceros) representando el fondo y 1  (unos) los objetos iguales o similares.  

```py
def ej8(img):
    # Image size
    h, w = img.shape[:2]
    return flood(img, (round(50),round(50)), tolerance=0.05)
```
![alt text](/Entrega%202/Images/img39.png)
![alt text](/Entrega%202/Images/img40.png)

---

### Ejercicio 10
>**Segmentación por combinación de técnicas (pipeline) Pregunta: ¿Qué beneficios tiene combinar varias técnicas de segmentación en un mismo flujo de procesamiento? Práctica sugerida: Aplicar primero Canny + morfología para generar una máscara, luego segmentar con Watershed o K-means sobre la región recortada.**

El beneficio se encuentra en que cada técnica aporta distintas ventajas para poder realizar la segmentación. En el ejemplo de este ejercicio, por un lado Canny detecta contornos de forma nítida y precisa, luego las técnicas de morfología permiten suavizar la imagen (si fuera necesario) y rellenar las regiones para finalmente poder realizar la segmentación aplicando el algoritmo de Watershed.  
Para el algoritmo de Watershed se hicieron algunos procesamientos previos para obtener la información necesaria:  
- Transformada de distancia: Sirve para detectar regiones "seguras" del primer plano: los puntos más internos de los objetos, que seguro pertenecen a un solo objeto. Esto a grandes razgos crea un "mapa de calor" indicando en cada píxel cuan lejos está del fondo o el exterior del objeto a segmentar.  
- Región segura del fondo: Aumenta las áreas negras de la imagen con `cv2.dilate` (que corresponden al fondo). Así se obtiene una región del fondo que sabemos con certeza que es fondo.  
- Región desconocida: Resta el primer plano del fondo para obtener una zona de incertidumbre donde no se está seguro si pertenece al fondo o a algún objeto.
- Etiquetado de marcadores: `connectedComponents` da un número distinto a cada región de primer plano segura. Se le suma 1 para que el fondo tenga el valor 1 (y no 0).Se marca la zona desconocida con 0, que es lo que necesita cv2.watershed() para saber qué zonas tiene que decidir a cuál marcador pertenecen.


```py
def ej10(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny edge detection
    edges = cv2.Canny(gray, 150, 500)

    # Rellenar objetos: morfología de cierre + dilatación
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    kernel = np.ones((5, 5), np.uint8)


    # Invertir para que los objetos sean blancos
    mask = cv2.bitwise_not(dilated)
    #mask = dilated

    # Transformada de distancia para encontrar objetos (región segura)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Región segura del fondo
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Región desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Etiquetado de marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    print("Markers unique values:", np.unique(markers))

    # Watershed
    markers = cv2.watershed(img, markers)

    
    # Crear una imagen de salida coloreada
    segmented = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255), (255,0,255), (128,128,0), (128,0,128), (0,128,128)]

    for label in np.unique(markers):
        if label == -1:
            segmented[markers == label] = [0, 0, 0]  # borde
        elif label == 1:
            segmented[markers == label] = [128, 128, 128]  # fondo (opcionalmente distinto)
        else:
            segmented[markers == label] = colors[(label - 2) % len(colors)]

    # Mostrar resultados
    show_images([img, edges, mask, segmented], 
                titles=['Original Image', 'Canny Edges', 'Mask', 'Segmented Image'])
```

![alt text](/Entrega%202/Images/img41.png)

---

### Ejercicio 11
>**Elegir y describir alguna de las siguientes tecnicas de segmentación:  
(a) Basada en Clustering  
(b) Basada en Grafos  
(c) Basadas en Modelos Probabilísticos y Estadísticas**

**Segmentación basada en Grafos**  
La segmentación basada en grafos es un enfoque que modela una imagen como un grafo, donde cada píxel (o superpíxel) representa un nodo y las conexiones entre estos nodos, llamadas aristas, reflejan relaciones de similitud entre los píxeles, como la diferencia de color, textura o intensidad. La idea central es dividir la imagen en regiones que sean internamente coherentes y externamente distintas, utilizando propiedades del grafo.

En este contexto, el algoritmo más conocido es el de Felzenszwalb y Huttenlocher, que construye una jerarquía de regiones conectadas a partir de un análisis de las diferencias entre los nodos conectados. Se define una medida de disimilitud entre regiones, y se van agrupando aquellas cuya diferencia interna es menor que un umbral dinámico. Esto permite una segmentación eficiente y rápida que se adapta automáticamente a la estructura de la imagen, sin necesidad de parámetros globales como el número de regiones a extraer.

Este enfoque tiene ventajas como su velocidad y su capacidad de manejar bien imágenes con ruido o estructuras complejas. Además, al trabajar con superpíxeles o regiones en lugar de píxeles individuales, puede mantener bordes nítidos y producir segmentaciones con significado semántico más alto.

![image.png](/Entrega%202/Images/img42.png)

---
## Trabajo Práctico 6
## Representacion y Descripcion de Caracterısticas
### Ejercicio 2
>**Representacion por relleno de regiones. Identificar los objetos en una imagen binaria y colorear
cada region detectada. Sugerencia: scikit-image: measure.label, regionprops, label2rgb.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ1.ipynb#2).

Los distintos algoritmos involucrados cumplen las siguientes funciones:
1) `label()` recorre la imagen y asigna etiquetas diferentes a cada conjunto de píxeles conectados (usando un algoritmo como Union-Find o Flood-Fill).
2) `regionprops()` analiza esos conjuntos etiquetados para extraer características geométricas y estadísticas.
3) `label2rgb()` toma la matriz de etiquetas y le asigna un color diferente a cada etiqueta, lo que permite visualizar los objetos.

El resultado final es una imagen donde cada objeto detectado se muestra con un unico color (segmentacion). Es una 
tecnica bastante simple de utilizar pero puede tener problemas en imagenes complejas

Ejemplo de imagen original y su resultado:

![Imagen](/Entrega%202/Images/img43.png)

### Ejercicio 4
>**Calculo de propiedades geometricas. Extraer area, perımetro, excentricidad y compacidad de
cada region segmentada. Sugerencia: regionprops de skimage.measure.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ1.ipynb#4).

Descricion de las propiedades geométricas:

 **Área**: Cantidad de píxeles que forman la región. 
 
 **Perímetro**: Longitud de la frontera del objeto (estimado). 
 
 **Excentricidad**: Relación entre los ejes de la elipse que mejor ajusta la región (0: círculo, 1: línea).
 
 **Compacidad**: Medida de forma: $C = \frac{P^2}{4\pi A}$, idealmente ≈1 para círculos.
 
Basicamente todas estas propiedades nos puede ubicar en datos como el tamaño de los objetos, su forma y como se distribuyen en el espacio.

Excentricidad → Cerca de 0: círculo | Cerca de 1: elipse larga / línea.

Compacidad → Cerca de 1: figura compacta (círculo). Alta compacidad → figura irregular o alargada. 

La siguiente imagen muestra como se identifican las regiones dentro de una imagen:

![Imagen](/Entrega%202/Images/img44.png)

Y los resultados de las propiedades geometricas:

Región 1:
  Área: 5183.0
  Perímetro: 283.41
  Excentricidad: 0.03
  Compacidad: 1.23
------------------------------
Región 2:
  Área: 4069.0
  Perímetro: 234.89
  Excentricidad: 0.03
  Compacidad: 1.08
------------------------------
Región 3:
  Área: 4081.0
  Perímetro: 235.14
  Excentricidad: 0.09
  Compacidad: 1.08

......

---
### Ejercicio 6
>**Descriptores de textura con GLCM. Calcular contraste, correlacion y homogeneidad de regiones
usando matrices de co-ocurrencia. skimage.feature.greycomatrix, greycoprops.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ1.ipynb#6).

Una GLCM (Matriz de Co-ocurrencia de Niveles de Gris) es una herramienta para analizar la textura de una imagen, 
cuantificando cómo se relacionan los niveles de gris entre píxeles vecinos. Cada celda (i,j) de la GLCM indica cuántas 
veces ocurre una pareja de píxeles donde un píxel tiene valor i y su vecino tiene valor j, bajo cierta dirección y 
distancia. Gracias a esto podemos captar patrones de textura como rugosidad, homogeneidad, y dirección, al observar cómo 
se distribuyen las intensidades vecinas. Para esto tenemos 3 descriptores principales:

**Contraste**: Mide la diferencia local de los niveles de gris. Un valor alto indica una textura 'rugosa'.

**Correlación**: Mide la linealidad de la relación entre los niveles de gris de los píxeles vecinos. Un valor alto 
indica una textura homogénea y repetitiva.

**Homogeneidad**: Mide la cercanía de la distribución de los elementos a la diagonal de la GLCM. Un valor alto indica 
una textura muy uniforme.

Por ejemplo utilizaremos la siguiente imagen para analizar:

![Imagen](/Entrega%202/Images/img45.png)

de los cuales obtuvimos los siguientes resultados:

Contraste:[[156.73700224 229.77816315 153.39982103 287.75068691]]

Correlación:[[0.96655226 0.95100431 0.96748594 0.93864191]]

Homogeneidad:[[0.21780395 0.19163367 0.26018306 0.18837595]]

---

### Ejercicio 8
>**Relacion espacial entre regiones. Determinar si las regiones estan adyacentes o si una esta contenida en otra. 
skimage.measure.regionprops + analisis de coordenadas / bounding boxes.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ1.ipynb#8).

Para este ejercicio se creo una imagen de ejemplo con 4 regiones, de las cuales, la region 1 esta contenida dentro de 
la 2, la 2 es adyacente a la 3 y la 4 no tiene relacion con niguna otra.

Una vez aplicados los algoritmos los resultados obtenidos son los siguientes:

![Imagen](/Entrega%202/Images/img46.png)

y los resultados obtenidos son los siguientes:

--- Relaciones de Contención ---

Región 1 está contenida en Región 2.

Región 3 NO está contenida en Región 2.

--- Relaciones de Adyacencia ---

Región 2 es adyacente a Región 3.

Región 3 NO es adyacente a Región 4.

---
## Reconocimiento de Patrones
### 1. Template Matching
>**Buscar una figura conocida dentro de una imagen mediante una plantilla.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ2.ipynb#1).

Se utilizó la función `cv2.matchTemplate` para localizar una mariposa dentro de una imagen más grande. El método compara la plantilla con todas las posiciones posibles en la imagen y devuelve la ubicación con mayor similitud. Se dibuja un rectángulo sobre la coincidencia encontrada.

**Código principal:**
```py
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image_color, top_left, bottom_right, (0, 0, 255), 2)
```
Se probó con dos plantillas distintas sobre la misma imagen, mostrando los resultados visualmente.
![alt text](/Entrega%202/Images/img47.png)
![alt text](/Entrega%202/Images/img48.png)

---

### 2. Clasificación basada en características
>**Extraer características simples de regiones segmentadas y clasificarlas usando KNN.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ2.ipynb#2).

Para este ejercicio se segmentaron figuras geométricas (círculos, triángulos, cuadrados) en una imagen binaria.  
Primero se lee la imagen en escala de grises y se invierten los colores para que las figuras sean blancas sobre fondo negro. Después se aplica un umbral automático usando el método de Otsu para separar las figuras del fondo. Luego, etiqueta cada una de las regiones detectadas y extrae características geométricas como área, excentricidad y perímetro. A esas regiones se les asignan etiquetas manuales según su forma (círculo, triángulo o cuadrado), siguiendo el orden visual en que aparecen. Con esas características y etiquetas, se entrena un clasificador K-Nearest Neighbors (KNN).  
Los datos se escalan, se dividen en conjunto de entrenamiento y prueba, y finalmente se entrena el modelo y se evalúa su rendimiento imprimiendo un reporte de clasificación.

**Código principal:**
```py
# 1. Leer imagen en escala de grises
image = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

# 2. Invertimos colores (blanco fondo -> negro fondo)
image_inv = cv2.bitwise_not(image)

# 3. Umbral binario + Otsu
_, thresh = cv2.threshold(image_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Etiquetar regiones
labeled_image = label(thresh)

# 5. Extraer características
features = ['area']
props = regionprops_table(labeled_image, intensity_image=image_inv, properties=features)
df = pd.DataFrame(props)

# 6. Asignar etiquetas manuales
df['label'] = [
    1, 0, 0, 0, 0, 1,
    0, 1, 1, 0, 1, 1,
    0, 0, 1, 0, 0, 1,
    0, 0, 0, 1, 0, 0,
    0, 1, 0, 1, 1, 0,
    1, 1, 1, 0, 0, 1
] # 0: círculo, 1: cuadrado

# 7. Clasificación con KNN
X = df[features]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```
El clasificador logró distinguir correctamente las figuras según sus características.

              precision    recall  f1-score   support

           0       1.00      1.00      1.00         5
           1       1.00      1.00      1.00         6

    accuracy                           1.00        11
   macro avg       1.00      1.00      1.00        11
weighted avg       1.00      1.00      1.00        11

![alt text](/Entrega%202/Images/img49.png)


---

### 5. Reconocimiento estructural con grafos
>**Representar caracteres como grafos de líneas y nodos. Clasificarlos según su estructura.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ2.ipynb#5).

Se generaron imágenes de letras y se extrajo su esqueleto. Cada píxel del esqueleto se consideró un nodo y se conectó con sus vecinos, formando un grafo. Se extrajeron descriptores como cantidad de nodos, aristas, extremos, uniones y ciclos. Se entrenó un KNN para clasificar letras (A, B, C)  a partir de estos descriptores con un dataset que cuenta con 3 imágenes de cada letra.

**Código principal:**
```py
def extract_graph_features(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary // 255)
    coords = np.argwhere(skeleton)
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G.nodes:
                    G.add_edge((y, x), (ny, nx_))
    
    # Extraer descriptores simples del grafo
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    endpoints = len([n for n in G.nodes if G.degree[n] == 1])
    junctions = len([n for n in G.nodes if G.degree[n] > 2])
    cycles = len(nx.cycle_basis(G))

    return [num_nodes, num_edges, endpoints, junctions, cycles]
```
El método permite distinguir letras según su estructura topológica.  
Los resultados devuelven un bajo accuracy, probablemente debido a la limitada cantidad de datos para entrenamiento.

              precision    recall  f1-score   support

           0       0.00      0.00      0.00         2
           1       0.33      0.50      0.40         2
           2       0.33      0.50      0.40         2

    accuracy                           0.33         6
   macro avg       0.22      0.33      0.27         6
weighted avg       0.22      0.33      0.27         6

**Imágenes del dataset**  
![alt text](/Entrega%202/Images/img50.jpeg)  
![alt text](/Entrega%202/Images/img51.jpeg)  
![alt text](/Entrega%202/Images/img52.png)  
---

### 6. Clasificación de dígitos con CNN (MNIST)
>**Entrenamiento de una red neuronal convolucional para reconocimiento de dígitos.**

Se puede ejecutar en el siguiente [Notebook](./TP6/EJ2.ipynb#6).

Se utilizó el dataset MNIST. Las imágenes se normalizaron y se definió una arquitectura CNN con dos capas convolucionales y una densa. Se entrenó el modelo y se evaluó la precisión sobre el conjunto de test.

**Código principal:**
```py
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)
```
Se obtuvo una precisión superior al 98% en el conjunto de test, mostrando la efectividad de las CNN para reconocimiento de patrones complejos.

![alt text](/Entrega%202/Images/img53.png)

---

## Conclusiones

Se exploraron distintos enfoques para el reconocimiento de patrones en imágenes: desde métodos clásicos como template matching y clasificación basada en descriptores geométricos, hasta técnicas estructurales con grafos y modelos avanzados de deep learning. Cada método tiene ventajas según el tipo de problema y la complejidad de los patrones a reconocer.


