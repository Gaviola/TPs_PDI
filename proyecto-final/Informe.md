# Restauración Digital de Murales de Dunhuang mediante Detección de Daños

**Facundo Gaviola**  
**Luciano Masuelli**

## 1. Introduccion
En este informe se presenta una metodología de procesamiento de imágenes para la detección automatizada de daños en murales antiguos, como parte de un proyecto de restauración digital. El objetivo principal es desarrollar un modelo capaz de identificar las áreas deterioradas en fotografías de los murales de Dunhuang, obtenidas del dataset público de Dryad (https://datadryad.org/landing/show?id=doi:10.5061/dryad.bnzs7h4jd#readme). A lo largo del tiempo estos murales han sufrido un gran deterioro resultando en fisuras, huecos o sectores despintados o con perdida de color. Es por esto que se propone un enfoque de dos pipelines para generar una máscara de daño precisa, separando la detección de **huecos y zonas despintadas** de la detección de **fisuras finas y rayones**. La combinación de los resultados de ambos procesos genera una máscara de segmentación unificada que servirá como base para utilizar técnicas de reconstrucción de imágenes como lo puede ser tecnicas de inpainting.

## 2. Metodologia

### 2.1 Exploracion del dataset

Como primer paso, se realizo una inspeccion de las distintas imagenes contenidas dentro del dataset para poder determinar que estrategias podrian ser efectivas para poder detectar aquellas zonas deterioradas o dañadas. Dentro del mismo nos encontramos con un total de **1000** imagenes de distintas partes del mural las cuales tenian una amplia variedad de formas, colores, figuras y deterioros.

***Imagenes de Ejemplo***

Imagen con fisuras:

![Imagen con fisuras](dataset_con_filtro\img\000000.png)

Imagen con huecos despintados:

![Imagen con huecos](dataset_con_filtro\img\000048.png)

Imagenes con huecos y fisuras

![Imagen con huecos](dataset_con_filtro\img\000056.png)

Toda esta amplia variedad dentro de las imagenes dificulta la correcta deteccion de aquellas zonas afectadas por algun tipo de daño ya que dificulta la obtencion de un patron consistente con el que indentificarlo.

### 2.2 Pipelines

Para este trabajo de identificacion de la zonas dañadas se implementaron 2 pipelines de procesamiento de imagenes con el objetivo de detectar daños superficiales, fisuras, y áreas deterioradas. Se hace uso 2 pipelines debido a la complejidad de los patrones dentro de las imagenes, imposibilitando una correcta deteccion utilizando un unico pipeline.

#### 2.2.1 Pipeline General
El primer pipeline propuesto es mas eficaz para localizar aquellas regiones despintadas o con huecos aunque tiene la capacidad de detectar algunas grietas. Esta se basa en la intensidad del color (se busca un color terroso o arcilloso para los huecos y gruetas), técnicas de mejora de contraste y operaciones morfológicas. A continuaciónse detallan las etapas del procedimiento.

### 1. Preprocesamiento y Mejora de Contraste

Inicialmente, la imagen es convertida del espacio de color BGR al espacio CIELAB, con el fin de trabajar sobre el canal de luminancia (L). A este canal se le aplica una ecualización adaptativa del histograma (CLAHE) con el objetivo de mejorar el contraste local, especialmente en regiones de baja variación tonal. Posteriormente, se reconstruye la imagen LAB y se transforma a escala de grises para facilitar las siguientes etapas de segmentación.

### 2. Filtrado por Color (Gama de Marrones)

En esta etapa, la imagen original es transformada al espacio HSV para facilitar la detección de colores característicos del material dañado, particularmente tonalidades marrones. Se aplica un filtrado por rango de color, con límites inferiores y superiores definidos empíricamente para incluir una amplia gama de variantes del marrón. Esta máscara cromática se emplea como base para la identificación de fisuras.

### 3. Procesamiento Morfológico y Limpieza de Ruido

La máscara resultante es sometida a operaciones morfológicas de apertura y cierre, utilizando discos estructurantes de radio ajustable. Estas operaciones permiten eliminar pequeños artefactos y unificar regiones contiguas. Finalmente, se eliminan los objetos cuya área esté por debajo de un umbral mínimo, lo que permite descartar ruido residual y conservar únicamente las regiones significativas.

### 4. Salida del Proceso

El pipeline devuelve como salida principal una máscara binaria que representa las áreas que cumplen con los criterios de detección definidos.

Este enfoque integral permite una detección robusta de patrones de daño mediante el uso combinado de información tonal, morfológica y cromática, y puede ser ajustado según las condiciones particulares de las imágenes o del material analizado.




## 3. Resultados

(Poner imagenes del muro, su mascara de daños y su version restaurada con inpainting)

(Se puede poner el algunas comparaciones con las mascaras manuales)

(Mencionar que el proceso no es igual para todas las imagenes y pueden encontrarse casos donde no funcionen tan bien)

## 4. Conclusiones

Podemos concluir que la metodología propuesta, basada en un enfoque de doble pipeline, ha demostrado ser eficaz para segmentar diferentes tipos de deterioro presentes en los murales de Dunhuang. Al especializar cada pipeline, se logra un grado considerablemente bueno de precisión y se reduce el riesgo de falsos positivos. La máscara de daño generada es un producto esencial para la reconstruccion y que puede ser de gran ayuda para aplicar tecnicas como las siguientes:

 - Aplicacion de Algoritmos de Inpainting: Utilizar las mascaras generadas para rellenar aquellas zonas dañadas en base a su contenido circundante.
 - Aplicaciones de modelos de Deep Learning: Se pueden hacer uso de los resultados obtenidos para utilizarse como datos de entrenamiento para una red neuronal convolucional (CNN) con el objetivo de detectar mas facilmente el daño y reconstruirlo.