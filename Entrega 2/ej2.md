# Procesamiento de Imágenes - Reconocimiento de Patrones  
Luciano Masuelli  
Facundo Gaviola  

## Trabajo Práctico 6

### 1. Template Matching

**Buscar una figura conocida dentro de una imagen mediante una plantilla.**

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
![alt text](/Entrega%202/Images/imgsEj2/image.png)
![alt text](/Entrega%202/Images/imgsEj2/image-1.png)

---

### 2. Clasificación basada en características

**Extraer características simples de regiones segmentadas y clasificarlas usando KNN.**

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

![alt text](/Entrega%202/Images/imgsEj2/image-2.png)


---

### 5. Reconocimiento estructural con grafos

**Representar caracteres como grafos de líneas y nodos. Clasificarlos según su estructura.**

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
![alt text](/Entrega%202/Images/imgsEj2/a1.jpeg)  
![alt text](/Entrega%202/Images/imgsEj2/b3.jpeg)  
![alt text](/Entrega%202/Images/imgsEj2/c1.png)  
---

### 4. Clasificación de dígitos con CNN (MNIST)

**Entrenamiento de una red neuronal convolucional para reconocimiento de dígitos.**

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

![alt text](/Entrega%202/Images/imgsEj2/model.png)

---

## Conclusiones

Se exploraron distintos enfoques para el reconocimiento de patrones en imágenes: desde métodos clásicos como template matching y clasificación basada en descriptores geométricos, hasta técnicas estructurales con grafos y modelos avanzados de deep learning. Cada método tiene ventajas según el tipo de problema y la complejidad de los patrones a reconocer.
