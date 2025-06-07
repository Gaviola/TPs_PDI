# Restauración Digital de Murales de Dunhuang mediante Detección de Daños

### Facundo Gaviola
### Luciano Masuelli

## 1. Introduccion
En este informe se presenta una metodología de procesamiento de imágenes para la detección automatizada de daños en murales antiguos, como parte de un proyecto de restauración digital. El objetivo principal es desarrollar un modelo capaz de identificar las áreas deterioradas en fotografías de los murales de Dunhuang, obtenidas del dataset público de Dryad (https://datadryad.org/landing/show?id=doi:10.5061/dryad.bnzs7h4jd#readme). A lo largo del tiempo estos murales han sufrido un gran deterioro resultando en fisuras, huecos o sectores despintados o con perdida de color. Es por esto que se propone un enfoque de dos pipelines para generar una máscara de daño precisa, separando la detección de **huecos y zonas despintadas** de la detección de **fisuras finas y rayones**. La combinación de los resultados de ambos procesos genera una máscara de segmentación unificada que servirá como base para utilizar técnicas de reconstrucción de imágenes como lo puede ser tecnicas de inpainting.

## 2. Metodologia

La estrategia principal para la restauracion consiste en encontrar una mascara binaria en donde los pixeles blancos van a representar las zonas dañadas dentro de la imagen. Para la obtencion de estas mascaras se combinaran las mascaras resultantes obtenidas de 2 pipelines de procesamiento independientes, 1 para **fisuras finas y rayones** y otra para **huecos y zonas despintadas**. En general, se decidio ser lo mas conservador posible con la implementacion de las tecnicas ya que si bien se quiere restaurar aquellas partes dañadas, tampoco se quiere afectar en gran medida aquellas partes en buenas condiciones.

### 2.1 Pipeline de fisuras finas y rayones

(Hacer referencia a la funcion en en notebook)

El pipeline tendra como objetivo la identificacion de estructuras u objetos delgados y alargados, similares a fisuras y rayones. Para esto se define el siguiente proceso:

1) Se utiliza **CLAHE** para realzar el contraste en la version de escala de grises de la imagen recibida.
2) Se hace uso de el **operador de Sobel** en las direcciones de X e Y para calcular gradientes. Gracias a esto se pueden detectar cambios bruscos en la intensidad, similar a algo que podria ocurrir en una grieta.
3) Se realiza una **umbralizacion binaria** para conertir los bordes encontrados a una mascara
4) Se Procede a realizar la tecnica **open/close** con un radio pequeño para intentar eliminar algo de ruido sin afectar las lineas de las fisuras.
5) Finalmente, se procede a **analizar y filtrar las regiones** resultantes mediante un **area minima** y su **relacion de aspecto**, se busca que solo persistan aquellas figuras donde el eje mayor es significativamente mas largo que el eje menor.

(Poner imagenes de ejemplo de con las macaras obtenidas por este pipeline)



### 2.2 Pipeline de huecos y zonas despintadas

(Hacer referencia a la funcion en en notebook)

El pipeline tendra como objetivo la identificacion de huecos y zonas despintadas. Tambien en este caso concreto, hacemos uso del color de dichos huecos ya que generalmente estos poseen un color terroso o arcilloso. Para esto se define el siguiente proceso:


(Poner imagenes de ejemplo de con las macaras obtenidas por este pipeline)

### 2.3 Unificacion de mascaras

La máscara final de daño se obtiene combinando las salidas de ambos pipelines mediante una operación lógica OR a nivel de píxel. De esta manera, se integra la detección de daños extensos con la de daños lineales, generando un mapa completo y detallado de todas las áreas que requieren restauración.


## 3. Resultados

(Poner imagenes del muro, su mascara de daños y su version restaurada con inpainting)

(Se puede poner el algunas comparaciones con las mascaras manuales)

(Mencionar que el proceso no es igual para todas las imagenes y pueden encontrarse casos donde no funcionen tan bien)

## 4. Conclusiones

Podemos concluir que la metodología propuesta, basada en un enfoque de doble pipeline, ha demostrado ser eficaz para segmentar diferentes tipos de deterioro presentes en los murales de Dunhuang. Al especializar cada pipeline, se logra un grado considerablemente bueno de precisión y se reduce el riesgo de falsos positivos. La máscara de daño generada es un producto esencial para la reconstruccion y que puede ser de gran ayuda para aplicar tecnicas como las siguientes:

 - Aplicacion de Algoritmos de Inpainting: Utilizar las mascaras generadas para rellenar aquellas zonas dañadas en base a su contenido circundante.
 - Aplicaciones de modelos de Deep Learning: Se pueden hacer uso de los resultados obtenidos para utilizarse como datos de entrenamiento para una red neuronal convolucional (CNN) con el objetivo de detectar mas facilmente el daño y reconstruirlo.