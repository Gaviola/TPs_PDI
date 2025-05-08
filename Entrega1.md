# Procesamiento de Imágenes
Luciano Masuelli  
Facundo Gaviola  
## Trabajo Práctico 1
### 1. Modos de color en imágenes
### Ejercicio 6  
**La conversion de una imagen de color a escala de grises se puede hacer de varias formas. El ejercicio consiste en convertir la imagen de Lenna color a escala de grises utilizando diferentes metodos.** 

**(a) Usando la libreria cv2 y el metodo `cvtColor()`**  

```py
lenna_gris = cv2.cvtColor(lenna_color, cv2.COLOR_BGR2GRAY) # Solo queda una tupla con las dimensiones de la imagen. No hay canales
show_image(lenna_gris, 'Imagen Lenna Gris')
```

![alt text](/Images/image.png)

**(b) Usando la formula de luminancia**  

```py
#Formula de luminancia: Y = 0.299*R + 0.587*G + 0.114*B. Esta formula se desprende de como el ojo humano percibe los colores. 
b, g, r = cv2.split(lenna_color)
lenna_gris = 0.299 * r + 0.587 * g + 0.114 * b
lenna_gris = lenna_gris.astype(np.uint8) # el resultado de la operacion anterior deevuelve float, openCV no puede
# trabajar con float sino valores en el rango 0-255.
show_image( lenna_gris, 'Imagen Lenna Gris (Luminancia)')
```
![alt text](/Images/image-1.png)

**(c) Usando scickit-image y el metodo `rgb2gray()`**  

```py
lenna_color_rgb = cv2.cvtColor(lenna_color, cv2.COLOR_BGR2RGB)  # Convertimos a RGB
lenna_gris = ski.color.rgb2gray(lenna_color_rgb)  # Convertimos a escala de grises
lenna_gris = (lenna_gris * 255).astype(np.uint8)
show_image(lenna_gris, 'Imagen Lenna Gris (skimage)')
```
![alt text](/Images/image-2.png)

Se pueden ejecutar los ejercicios en el siguiente [Notebook](./TP1/ej1.ipynb#6). En la sección que corresponde al ejercicio 6 se encuentra la función `ej6()` que contiene el código anterior.  

**(d) ¿Que pasa con los canales?**  
Al transformar la imagen a una escala de grises, perdemos los distintos canales de colores de la imagen quedando un solo canal con los valores de [0-255] donde, 0 = negro, 255 = blanco [1-254] = grises. Esto es basicamente lo que hace la formula de luminancia.  

**(e) ¿Que profundidad de bits tiene la imagen?**  
La profundidad de bits se ve alterada ya que cuando teniamos la imagen a color poseiamos 3 canales con 256 valores para cada uno, dando como resultado una profundidad de 24 bits (256 x 256 x 256 = 2^8 x 2^8 x 2^8). Al reducir la imagen a un solo canal al transformarla a una escala de grises la imagen va a poseer 8 bits de profundiad (256 = 2^8).

**(f) Evaluar con otra imagen de mayor profundidad** 

**(g) ¿Que sucede con la imagen? ¿Ha cambiado algo?**  
De manera similar a como ocurre con una imagen de profundidad de 24 bits a color, si transformamos una imagen cuya profundidad es mayor, como por ejemplo 36 (4096 x 4096 x 4096 = 2^12 x 2^12 x 2^12), a una escala de grises quedaria una imagen de una profundidad de 12 (4096 = 2^12).  

### Ejercicio 7
**Convertir la imagen de Lenna a otros modos de color, como CMYK, HSV, HSL. Mostrar el resultado.**  

Al realizar la conversión a CMYK se utiliza el método `split()` de la librería cv2 para obtener los valores R,G y B. Luego se normalizan y se calcula el valor de K como la diferencia entre 1 y el máximo valor de R, G o B. Se define al denominador como (1-K) + 1e-8, evitando la división por cero, y se calculan los valores C,M,Y de la siguiente manera:  
C = (1 - R - K) / denom  
M = (1 - G - K) / denom  
Y = (1 - B - K) / denom  
Finalmente se convierten los valores al rango [0, 255] y se unen los canales utilizando `cv2.merge()`.  

Para los modos de color HSV y HSL se utilizó `cv2.cvtColor()` para realizar la conversión.  

Los resultados fueron los siguientes:  

![alt text](/Images/image-3.png)
![alt text](/Images/image-4.png)
![alt text](/Images/image-5.png)  



El código se muestra a continuación:
```py
def ej7():
    lenna_BGR = cv2.imread('Lenna.png')
    lenna_RGB = cv2.cvtColor(lenna_BGR, cv2.COLOR_BGR2RGB)

    # Convertir a CMYK
    R, G, B = cv2.split(lenna_RGB)
    # Normalizar los valores de R, G, B a [0, 1]
    R = R.astype(np.float32) / 255.0
    G = G.astype(np.float32) / 255.0
    B = B.astype(np.float32) / 255.0

    K = 1 - np.maximum(R, np.maximum(G, B))
    # Evitar división por cero cuando K = 1
    denom = (1 - K) + 1e-8

    # Calcular C, M, Y
    C = (1 - R - K) / denom
    M = (1 - G - K) / denom
    Y = (1 - B - K) / denom

    # Convertir a rango [0, 255] para mostrar o guardar
    C = (C * 255).astype(np.uint8)
    M = (M * 255).astype(np.uint8)
    Y = (Y * 255).astype(np.uint8)
    K = (K * 255).astype(np.uint8)

    # Unir los canales
    lenna_CMYK = cv2.merge((C, M, Y, K))

    lenna_HSV = cv2.cvtColor(lenna_BGR, cv2.COLOR_BGR2HSV)
    lenna_HSL = cv2.cvtColor(lenna_BGR, cv2.COLOR_BGR2HLS)
```
Se puede ejecutar en el siguiente [Notebook](./TP1/ej1.ipynb#6)

### Ejercicio 8
**Tomar la imagen convertida en escala de grises y volver a convertir al en modo RGB. ¿Que ha sucedido?**  

La imagen se sigue viendo gris al transformarse a RGB. Lo que si cambia es el hecho de que en este caso volvemos a tener 3 canales, pero dentro de cada pixel tenemos el mismo valor en cada canal. Esto se debe a que al transformar la imagen de RGB a gris utilizamos la formula de luminancia, el cual nos da 1 solo valor por pixel. Esto provoca una perdida de informacion ya que no se puede recuperar la informacion original de los colores partiendo unicamente de la imagen en escala de grises.  

![alt text](/Images/image-6.png)
![alt text](/Images/image-7.png)

El código se muestra a continuación:
```py
def ej8():
    lenna_gris = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)  # Cargar la imagen en escala de grises
    lenna_rgb = cv2.cvtColor(lenna_gris, cv2.COLOR_GRAY2RGB)  # Convertir a RGB

    # Mostrar la imagen original y la convertida
    show_image(lenna_gris, 'Imagen Lenna Gris (Original)')
    show_image(lenna_rgb, 'Imagen Lenna Gris (Convertida a RGB)')
```
Se puede ejecutar en el siguiente [Notebook](./TP1/ej1.ipynb#6)


### 2. Compresión de Imágenes
### Ejercicio 2
**Dar detalles de las siguientes métricas de calidad de compresión (PSNR, SSIM)**   

**PSNR**: La relación señal-ruido máxima (PSNR) es una métrica de referencia completa no lineal que compara los valores de los píxeles de la imagen de referencia original con los de la imagen degradada. Para calcular la PSNR, primero se debe calcular el error cuadrático medio (MSE). Cuanto menor sea el MSE, menor será el error y mayores serán los resultados de la PSNR. La idea principal es que cuanto mayor sea la puntuación de la PSNR, mejor se habrá reconstruido la imagen degradada en comparación con la imagen de referencia, lo que a su vez significa que el algoritmo utilizado para la reconstrucción también es mejor. 
Sin embargo, la mayor desventaja que presenta PSNR es que sus puntuaciones no siempre se correlacionan con la calidad percibida. Un área común donde se aprecia esto es en la borrosidad, donde por ejemplo si se tienen dos imágenes identicas salvando que una es mas borrosa generalmente presentan un puntaje similar.  

**SSIM**: El índice de similitud estructural (SSIM) es una métrica de referencia completa no lineal que compara la luminancia, el contraste y la estructura de la imagen original y la degradada. En otras palabras, SSIM mide las diferencias entre las propiedades (luminancia, contraste y estructura) de los píxeles  
El SSIM se mide en una escala de 0 a 1, donde cuanto más cercana sea la puntuación a 1, más similar será la imagen degradada a la imagen de referencia. Como se mencionó anteriormente, el SSIM es una métrica no lineal: los resultados de 0,97 a 1 indican una degradación mínima, los de 0,95 a 0,97 representan una degradación baja y los resultados por debajo de estos rangos indican una degradación media o alta.  
SSIM es muy sensible a cualquier tipo de cambio estructural, como el estiramiento de una imagen, rotaciones o distorsiones similares. También se ve muy afectado por los bloques y el desenfoque.  
Como desventaja, SSIM no es bueno para evaluar cambios en el tono de la imagen y factores similares.

### Ejercicio 6
**Implementar un modelo de compresion basado en codificacion Run-Length Encoding (RLE). El
algoritmo Run-Length Encoding (RLE) reduce el tamaño de una imagen representando secuencias consecutivas de pıxeles identicos como una sola entrada. Para ello convertir una imagen en escala de grises. luego, implementar el algoritmo RLE para comprimir la imagen. Posteriormente implementar una funcion para descomprimir la imagen. Al finalizar, mostrar la imagen original y la imagen reconstruida. Probar con dos o tres imagenes que tengan diferentes caracterısticas, modos de color. utilizar alguna de las metricas nombradas anteriormente e evaluar el resultado de la misma.**  

Para la implementación de este ejericicio se desarrolló la función `RLE()` que se encarga de generar la codificación RLE, mediante un bucle que va almacenando en una lista la codificación, una tupla de la forma (pixel, cantidad), de pixeles contínuos iguales.  
La función `RLE_decoded()` es la encargada de decodificar la imagen antes codificada con RLE. Para esto, recorre la lista y mediante la función `extend()` se extiende la cantidad de pixeles repetidos. Finalmente retorna un array con la forma original de la imagen.  

El siguiente es el código desarrollado:  

```py
def RLE(img):
    img_flat = img.flatten()
    img_rle = [] # Imagen codificada con RLE
    
    # Guardamos el primer pixel encontrado
    last_pxl = img_flat[0]
    count = 1
    
    for pxl in img_flat[1:]:  
        if np.array_equal(pxl, last_pxl): # Si es igual al anterior sumamos 1 a la longitud
            count += 1
        else:
            img_rle.append((last_pxl, count)) # Si no es igual guardamos la cadena contigua
            last_pxl = pxl
            count = 1
    img_rle.append((last_pxl, count)) # Guardamos la ultima cadena
    return img_rle

def RLE_decoded(img_rle, shape):
    decoded_pxl = []
    for pxl, count in img_rle:
        decoded_pxl.extend([pxl] * count) # Repetimos el pixel la cantidad de veces que indica la cadena
    
    return np.array(decoded_pxl, dtype=np.uint8).reshape(shape) # Convertimos a array y le damos la forma original
```
Se puede ejecutar en el siguiente [Notebook](./TP1/ej2.ipynb)

A continucaicón se muestran los resultados obtenidos en dos imágenes distintas:  
![alt text](/Images/image-8.png)
![alt text](/Images/image-9.png)
![alt text](/Images/image-10.png)
![alt text](/Images/image-11.png)

| Imagen | PSNR (dB) | SSIM   |
|--------|-----------|--------|
| 1      | inf       | 1.0000 |
| 2      | inf       | 1.0000 |


Como se puede ver en los resultados, dado que el algoritmo RLE es un algoritmo de compresion sin perdida, las metricas PSNR y SSIM nos dicen que ambas imagenes son iguales.  


## Trabajo Práctico 2
### 1 Histogramas 
### Ejercicio 7
**Transformar la distribucion de intensidades de una imagen para que se parezca a la de otra. Implementar el ajuste de histograma usando OpenCV o skimage.exposure.match histograms(). Comparar los histogramas antes y despues del ajuste.**  

Para llevar a cabo este ejercicio se utilizó `skimage.exposure.match_histograms()` para cada canal (B,G,R).  
A continuación se muestra el código:
```py
def match_hist_color(source, reference):
    matched = source.copy()
    for i in range(3):  # B, G, R
        matched[:,:,i] = match_histograms(source[:,:,i], reference[:,:,i])
    return matched
```
Se puede ejecutar en el siguiente [Notebook](./TP2/EJ1.ipynb)

Las imágenes de entrada y la resultante son las siguientes:  
![alt text](/Images/image-12.png)
![alt text](/Images/image-13.png)
![alt text](/Images/image-14.png)

Para realizar la comparación, se graficaron los histogramas de cada imagen, donde se puede apreciar cómo el histograma de la imagen resultante se asemeja a el de la referencia, el canal azúl tiene una menor intensidad y frecuencia, mientras que el verde se amplía a frecuencias más altas. Se nota tambíen que al hacer esto hay una pérdida de información en algunas intensidades de color.
![alt text](/Images/image-15.png) 
![alt text](/Images/image-16.png)
![alt text](/Images/image-17.png)


### Ejercicio 8
**Aplicar ecualizacion de histograma a una imagen en escala de grises. Comparar la imagen original
con la ecualizada.**  

Para este ejercicio se utilizó la función `cv2.equalizeHist()`.  
El código se encuentra en el siguiente [Notebook](./TP2/EJ1.ipynb).  

Los resultados fueron los siguientes:  

En este caso la mariposa pierde calidad de imagen al aplicar la ecualizacion. Esto puede deberse a que la imagen de por si posee un buen balance en el histograma, sumado a que la el valor blanco en la imagen tiene una mayor representacion en el histograma, lo que tiende a dividir en 2 partes el histograma.
![alt text](/Images/image-18.png)  
![alt text](/Images/image-19.png)  
![alt text](/Images/image-20.png)  
![alt text](/Images/image-21.png)  



Al utilizar una imagen mas simple se puede observar como la ecualizacion mejora el contraste y balancea el histograma de la imagen.  
![alt text](/Images/image-22.png)
![alt text](/Images/image-23.png)  
![alt text](/Images/image-24.png)
![alt text](/Images/image-25.png)

### Ejercicio 9
**Implementar una umbralizacion manual eligiendo un valor de umbral. Usar el metodo de Otsu
para calcular un umbral optimo automaticamente.**

El código se encuentra en el siguiente [Notebook](./TP2/EJ1.ipynb). 
```py
# Implementacion con umbralizacion manual

img = cv2.imread('bosque.jpg', cv2.IMREAD_GRAYSCALE)
umbral = 180
_, img_umbral = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY) # Devuelve el umbral y la imagen umbralizada. Devuelve -1 si no encuentra un umbral

# Implementacion con umbralizacion de Otsu

umra_otsu, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

show_image(img, 'Imagen Original')
show_image(img_umbral, 'Imagen Umbralizada Manual')
show_image(img_otsu, 'Imagen Umbralizada Otsu')

```
Para ambas umbralizaciones se utiliza la funcion `cv2.threshold()`pero para la umbralizacion con el metodo de otsu se 
utilizo la funcion  con el flag `cv2.THRESH_OTSU`. El flag `cv2.THRESH_BINARY` indica que se va a realizar una 
umbralizacion binaria.

El metodo de Otsu busca encontrar un umbral en el que dado un fondo y un objeto en el frente estos esten lo mas separados 
posibles lo que implica que tengan una varianza minima entre ambos y sus medias estan lo mas alejadas posibles (medido 
en el valor de intensidad)

En las siguientes imagenes se puede apreciar la diferencia entre aplicar un umbral manualmente (180) y el obtenido por 
el metodo de Otsu (127).

![alt text](/Images/image-26.png)
![alt text](/Images/image-27.png)
![alt text](/Images/image-28.png)

### Ejercicio 11
**Implementar la transformacion gamma I’=I
y, permitiendo ajustar el valor de y dinamicamente.
Aplicar diferentes valores de y en distintas regiones de la imagen (por ejemplo, usando una mascara 
o adaptando y en funcion del brillo local). Visualizar el efecto de la correccion gamma en la imagen
y en su histograma.**

El codigo se puede ejecutar en el siguiente [Notebook](./TP2/EJ1.ipynb).

La funcion transformacion gamma se implemento de la siguiente manera:
```py
def gamma_correction(img, gamma=2.8):
    val_max = 256
    intensidad_max = val_max - 1
    tabla_trans = np.zeros((val_max,), dtype=np.uint8) 
    
    for intensidad in range(val_max):
        val_escala = (intensidad / intensidad_max) # Escalamos al intervalo [0,1]
        val_trans = np.power(val_escala, gamma) # Aplicamos la transformacion gamma
        intensidad_salida = int(np.round(val_trans * intensidad_max)) # Escalamos al intervalo [0,255]
        tabla_trans[intensidad] = np.clip(intensidad_salida, 0, 255) # Limitamos el valor a [0,255]
    
    img_trans = cv2.LUT(img,tabla_trans) # Aplicamos la tabla de transformacion a la imagen
    return img_trans
```

En base a esta funcion de la de la transformacion gamma se procedieron a crear distintas regiones en la imagen en 
funcion de mascaras (La imagen se dividio en 4 regiones) y se aplico un valor de gamma distinto para cada una de ellas.
La funcion cv2.LUT() es una lookup table que mapea cada pixel de la imagen original a su transformacion coorespondiente.

Los resultados obteinidos fueron los siguientes:

![alt text](/Images/image-29.png)
![alt text](/Images/image-30.png)
![alt text](/Images/image-31.png)
![alt text](/Images/image-32.png)
![alt text](/Images/image-33.png)
![alt text](/Images/image-34.png)
![alt text](/Images/image-35.png)
![alt text](/Images/image-36.png)

En dichas imagenes podemos ver como dependiendo del valor de gamma se puede apreciar un aumento o disminucion en el 
brillo (para gamma menor a 1 se oscurece mientra que con gamma mayor a 1 se aclara). Por el otro lado, a la hora de
analizar el histograma, este sufre una traslacion de los valores hacia la izquierda o derecha dependiendo del valor de 
gamma (para gamma menor a 1 se corre hacia la derecha y para gamma mayor a 1 a la izquierda).

### 2 Combinacion de imagenes

### Ejercicio 3
**Multiplicacion y division de imagenes: Multiplicar y divide dos imagenes pıxel a pıxel utilizando
cv2.multiply() y cv2.divide(), observando como afecta el brillo y contraste.**

El código se encuentra en el siguiente [Notebook](./TP2/EJ2.ipynb).

Se utilizaron las siguietes imagenes para realizar las operaciones:

![alt text](/Images/image-37.png)
![alt text](/Images/image-38.png)

se realizaron img1 op img2 y img2 op img1. Los resultados obtenidos de las operaciones fueron los siguientes:

![alt text](/Images/image-39.png)
![alt text](/Images/image-40.png)
![alt text](/Images/image-41.png)
![alt text](/Images/image-42.png)

Al realizar la multiplicacion se tiende a aumentar el brillo de los pixeles, mientras que al aplicar la division esta va
a depender del valor de la imagen que se utilice como divisor. Si el divisor es mayor a el dividendo el resultado tiende
osurecer el pixel, por el lado contrario, si el divisor es menor a el dividendo el resultado tiende a aclarar el pixel.

### Ejercicio 5
**Combinacion con operadores logicos: Usa operadores booleanos (cv2.bitwise and, cv2.bitwise or,
cv2.bitwise xor) para fusionar imagenes basandose en una mascara binaria. Describir que sucede en
cada caso**

El código se encuentra en el siguiente [Notebook](./TP2/EJ2.ipynb).

Para este ejercicio se utilizaron las siguientes imagenes:

![alt text](/Images/image-43.png)
![alt text](/Images/image-44.png)

Y posteriormente se aplico una umbralizacion a la primera imagen com un valor de 60 resultando en lo siguiente:
![alt text](/Images/image-45.png)

Despues se procedio a realizar las operaciones logicas obteniendo las siguientes imagenes:

![alt text](/Images/image-46.png)
![alt text](/Images/image-47.png)
![alt text](/Images/image-48.png)

La mascara en cada una de las 3 operaciones delimita la zona en la cual se realizaran las comparaciones de bit a bit. Una vez delimitada la zona aplica la opracion correspondiente (and, or o xor) entre los bits que representan cada pixel.
 
Ej: pixel1 = 150, pixel2 = 200 y su respectivo valor binario es 10010110 y 11001000. Si aplicamos las operaciones bit a bit el resultado seria el siguiente:
##### and: 10010110 and 11001000 = 10000000 = 128 
##### or: 10010110 or 11001000 = 11011110 = 222
##### xor: 10010110 xor 11001000 = 01011110 = 94
Desde el punto de vista de la imagen resultante, el resultado de cada operacion es el siguiente:
##### and: Es una especie de fusion de ambas imagenes en donde se oscurece la imagen y utiliza de fondo la segunda imagen 
##### or: Fusiona ambas imagenes pero en este caso se un incremento del color blanco en el resultado. En este caso la imagen 2 utilizada de fondo no es tan clara
##### xor: En este caso se remarca mas el contraste entre ambas imagenes

### Ejercicio 8
**Uso de operadores logicos para reemplazar partes de una imagen: Reemplazar un area especıfica
de una imagen con otra utilizando operadores logicos y relacionales para definir la region de interes
(ROI).**

El código se encuentra en el siguiente [Notebook](./TP2/EJ2.ipynb).

Para este ejercicio se utilizaron las siguientes imagenes como fondo y objeto:

![alt text](/Images/image-49.png)
![alt text](/Images/image-50.png)

Para realizar esta tarea se utilizara la formula: 
##### R:= (B AND NOT C) OR (A AND C)
Donde A es el fondo, B es el objeto, C es la mascara del objeto y NOT C su mascara invertida.

De esta manera vamos a obtener las siguientes imagenes parciales durante el proceso:
## C
![alt text](/Images/image-51.png)
## NOT C
![alt text](/Images/image-52.png)
## A AND C
![alt text](/Images/image-53.png)
## B AND NOT C
![alt text](/Images/image-54.png)

Para finalmente aplicar la formula y obtener la imagen final:
## R:= (B AND NOT C) OR (A AND C)
![alt text](/Images/image-55.png)

### 3 Dominio Espacial
### Ejercicio 8
**Suavizado y Sobel: Aplicar un filtro gaussiano antes del operador de Sobel y analizar las diferencias
en la deteccion de bordes.** 

El codigo se encuentra en el siguiente [Notebook](./TP2/EJ3.ipynb).

Para este ejercicio se implementaron las funciones de filtro gaussiano y de sobel. 

```py
def kernel_gaussiano(size, sigma):
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalizamos para que sume 1
    return kernel

def filtro_gaussiano(img, size=3, sigma=1):
    h1, w1 = img.shape
    h2 = size
    w2 = size
    mask = kernel_gaussiano(size, sigma)
    if h2 % 2 == 0 or w2 % 2 == 0:
        raise ValueError("El tamaño del filtro debe ser impar.")
    
    if h2 > h1 or w2 > w1:
        raise ValueError("El tamaño del filtro no puede ser mayor que la imagen.")
    
    g = np.zeros_like(img, dtype=np.float32)
    for y in range(w1):
        for x in range(h1):
            valor = 0
            s = -h2 // 2
            for j in range(w2):
                t = -w2 // 2
                for i in range(h2):
                    if 0 <= y+t < w1 and 0 <= x+s < h1:
                        valor += mask[i,j] * img[x+s, y+t]
                    t += 1
                s += 1
            g[x, y] = valor
    return np.clip(g, 0, 255).astype(np.uint8)

def sobel_filter(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    filtered_x = filtro_gris(img, sobel_x, uint8=False)
    filtered_y = filtro_gris(img, sobel_y, uint8=False)
    filtered = np.sqrt(filtered_x**2 + filtered_y**2)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

```

Al aplicar estos filtros obtenemos los siguientes resultados:
![alt text](/Images/image-56.png)
![alt text](/Images/image-57.png)

### Ejercicio 12
**Comparación de Métodos de Detección de Bordes : Comparar Sobel, Prewitt, Laplace y Canny trabajando diversas 
imágenes con características diferentes.**  

El código se encuentra en el siguiente [Notebook](./TP2/EJ3.ipynb).

Para el ejercicio se implemento el filtro de prewitt, laplace, se utilizo el filtro de sobel previamente creado y se 
utilizo la funcion `cv2.Canny()` para aplicar el filtro de canny con distintos valores de threshold.

La implementacion de los filtros de prewitt y laplace son las siguiente:
```py
def prewitt_filter(img):
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)
    

    kernel_y = np.array([[ 1,  1,  1],
                        [ 0,  0,  0],
                        [-1, -1, -1]], dtype=np.float32)

    filtered_x = filtro_gris(img, kernel_x, uint8=False)
    filtered_y = filtro_gris(img, kernel_y, uint8=False)
    filtered = np.sqrt(filtered_x**2 + filtered_y**2)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def laplace_filter(img):
    mask = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=np.float32)
    return filtro_gris(img, mask)
```

A la hora de aplicar los distintos filtros obtenemos algunos de estos resultados (En el notebook hay mas ejemplos):

![alt text](/Images/image-58.png)
![alt text](/Images/image-59.png)
![alt text](/Images/image-60.png)
![alt text](/Images/image-61.png)
![alt text](/Images/image-62.png)

En base a estas imagenes Se observa que los filtros Sobel y Prewitt son bastante similares, aunque se nota cómo Sobel 
enfatiza los bordes izquierdos de las figuras en las imágenes y Prewitt hace algo similar con los bordes derechos. En 
imágenes con muchos detalles como la del bosque, Sobel y Prewitt devuelven un resultado difícil de visualizar, al igual 
que el resto de los filtros. Laplace y Canny en imágenes más simples detallan con claridad los bordes, distinguiendose 
en que Laplace muestra detalles mas finos o delicados de la imagen y Canny resalta con un trazo grueso los detalles. 
Para la imagen del elefante Canny muestra satisfactoriamente el detalle de la figura, mientras que Laplace devuelve una 
imagen oscura con pocos detalles, suponemos que puede tener que ver con el poco contraste entre colores de la imagen; en
la imagen del bosque Laplace resulta devolver la imagen más distinguible, mientras que Canny pierde detalles más finos y
no permite comprenderla.

### Ejercicio 13
**Realce de Detalles: Aplicar un filtro de paso alto y sumarlo a la imagen original para mejorar los
detalles.**

El código se encuentra en el siguiente [Notebook](./TP2/EJ3.ipynb).

En este ejercicio se implemento la funcion de filtro gris la cual se le pasa como parametro un kernel de paso alto.
El codigo es el sguiente:
```py
kernel_paso_alto = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

def filtro_gris(img, mask, uint8=True):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h1, w1 = img.shape
    h2, w2 = mask.shape
    
    g = np.zeros_like(img, dtype=np.float32)
    for y in range(w1):
        for x in range(h1):
            valor = 0
            s = -h2 // 2
            for j in range(w2):
                t = -w2 // 2
                for i in range(h2):
                    if 0 <= y+t < w1 and 0 <= x+s < h1:
                        valor += mask[i, j] * img[x+s, y+t]
                    t += 1
                s += 1
            g[x, y] = valor
    if uint8:
        return np.clip(g, 0, 255).astype(np.uint8)
    else:
        return np.clip(g, 0, 255)
```

Una vez se filtra la imagen y se le suma a la imagen original obteniendo las siguientes imagenes:

![alt text](/Images/image-63.png)
![alt text](/Images/image-64.png)
![alt text](/Images/image-65.png)

### Ejercicio 15
**Filtro de Diferencia Gaussiana (DoG): Aplicar la t´ecnica de Diferencia de Gaussiana para resaltar
bordes.**

El código se encuentra en el siguiente [Notebook](./TP2/EJ3.ipynb).

Para este ejercicio se hizo uso de la funcion `cv2.GaussianBlur()` para aplicar el filtro gaussiano con 2 sigma distintos
y posteriormente aplicarle la diferentecia entre ellos.

La imagen resultante es la siguiente:

![alt text](/Images/image-66.png)
![alt text](/Images/image-67.png)

## Trabajo Práctico 3
### 1 Operadores morfologicos
### Ejercicio 4
**Apertura y clausura morfologica: Aplicar apertura y clausura para eliminar ruido o cerrar huecos.
Comparar la imagen original y la resultante de aplicar el operador. Comentar los efectos visuales.
Comparar con los resultados anteriores. Mostrar 4 subplots: original, apertura, cierre, diferencia
entre ambos.**

El código se encuentra en el siguiente [Notebook](./TP3/EJ1.ipynb).

Para el ejercicio se utilizo la funcion `cv2.morphologyEx()` para aplicar la apertura y clausura morfologica y 
`cv2.absdiff()` para calcular la diferencia entre las imagenes.

La imagen original fue umbralizada con un valor de 127 y posteriormente dicha imagen umbralizada se utilizaria para
aplicar las distintas operaciones:

![alt text](/Images/image-68.png)
![alt text](/Images/image-69.png)

De esta manera podemos concluir lo siguiente:
##### Apertura: Elimina el ruido y suviza los bordes
##### Clausura: Cierra los huecos y une los objetos separados

La diferencia entre aplicar operaciones como dilatacion o erosion es que la apertura y clausura permite tener 
terminaciones mas finas. Dicho de otra manera, si solo aplicaramos una erosion o dilatacion, ademas de eliminar cosas 
como el ruido o agrandar detalles, tambien se crea una pequeña deformacion de los objetos mas grandes. Si aplicamos una 
apertura o clausura podemos conseguir eliminar ruido, suavizar bordes o cerrar huecos (dependiendo que es lo que 
necesitemos hacer) sin perder tanto la forma de los objetos grandes.

### Ejercicio 5
**Operacion de gradiente morfologico: Aplicar el gradiente morfologico (dilatacion - erosion). Visualizar los bordes 
obtenidos mediante esta operacion.**

El código se encuentra en el siguiente [Notebook](./TP3/EJ1.ipynb).

En este ejercicio se utilizo la funcion `cv2.dilate()` y `cv2.erode()` para aplicar la dilatacion y erosion 
respectivamente, tambien se utilizo la funcion `cv2.getStructuringElement()` para crear un kernel cuadrado de 
3x3 y fianlmente se utilizo la funcion `cv2.subtract()` para calcular la diferencia entre las imagenes.

La imagen utilizada primero fue umbralizada con un valor de 127 y posteriormente se aplicaron las operaciones
de dilatacion y erosion.

![alt text](/Images/image-70.png)
![alt text](/Images/image-71.png)
![alt text](/Images/image-72.png)

### Ejercicio 7
**Segmentacion basica con umbral + morfologıa: Aplicar umbral, luego apertura y cierre para
mejorar el resultado. Ideal como paso previo a una segmentacion mas elaborada.**

El código se encuentra en el siguiente [Notebook](./TP3/EJ1.ipynb).

En este ejercicio se volvio a utilizar la funcion `cv2.morphologyEx()` para aplicar la apertura y clausura. El resultado
obtenido es el siguiente:

![alt text](/Images/image-73.png)
![alt text](/Images/image-74.png)




