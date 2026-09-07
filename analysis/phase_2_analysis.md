# Análisis de Detección de Fraude con Tarjetas de Crédito

## 1. Introducción
Este informe presenta el análisis técnico de la fase 2 de un proyecto de detección de fraude en transacciones de tarjetas de crédito. El objetivo principal es evaluar la eficacia de modelos de aprendizaje supervisado (SVM y MLP) para identificar transacciones fraudulentas en un entorno altamente desbalanceado.

## 2. Optimización de Hiperparámetros y Análisis de Modelos

### 2.1 Support Vector Machine (SVM)
Se utilizó una búsqueda de cuadrícula (*GridSearchCV*) para optimizar el modelo, evaluando el parámetro de regularización $C$ y el tipo de kernel.
- **Parámetros óptimos:** $C = 10$ y Kernel RBF.
- **Análisis técnico de parámetros:**
    - **$C$ (Regularización):** El valor de $C=10$ sugiere que el modelo favorece una clasificación precisa de los puntos de entrenamiento sobre una frontera de decisión más suave. Al ser un valor relativamente alto, el modelo busca minimizar el error de clasificación en los datos de entrenamiento, lo cual es crucial en tareas de detección de anomalías donde el margen de error debe ser mínimo.
    - **Kernel RBF (Radial Basis Function):** La elección del kernel RBF indica que la frontera de decisión de los datos de fraude no es lineal. El kernel RBF permite proyectar los datos a una dimensión superior, facilitando la separación de clases complejas mediante funciones de base radial.
    - **Efecto de Gamma (implícito):** Aunque no se realizó un grid search exhaustivo sobre gamma, el uso de RBF implica la presencia de este parámetro, el cual controla el radio de influencia de un solo ejemplo de entrenamiento. Un gamma alto resultaría en fronteras más ajustadas a los datos, mientras que un gamma bajo produciría una frontera más suave.

### 2.2 Multi-Layer Perceptron (MLP)
Se evaluó la arquitectura de la red neuronal mediante la profundidad de las capas ocultas y la función de activación.
- **Parámetros óptimos:** `hidden_layer_sizes = (32,)` y `activation = 'relu'`.
- **Análisis técnico de parámetros:**
    * **`hidden_layer_sizes`:** La elección de una única capa de 32 neuronas sugiere que el problema puede ser resuelto con una complejidad moderada. Una arquitectura más profunda (ej. 64, 32) podría haber capturado interacciones más complejas, pero en este caso, la simplicidad de la capa única permitió un mejor ajuste general o una mayor eficiencia.
    * **`activation` (ReLU):** La función de activación Rectified Linear Unit (ReLU) es estándar para evitar el desvanecimiento del gradiente (*vanishing gradient*) en comparación con funciones sigmoide o tanh, permitiendo un entrenamiento más rápido y efectivo de la red.

## 3. Comparación Técnica de Modelos

A continuación se presenta la comparación de desempeño obtenida tras la validación:

| Modelo | F1 (Validación CV) | F1 (Test) | Gap de Generalización | Tiempo de Entrenamiento (s) |
| :--- | :---: | :---: | :---: | :---: |
| **SVM** | 0.937 | 0.973 | 0.036 | 0.06 |
| **MLP** | 0.937 | 0.919 | 0.018 | 0.46 |

### 3.1 Análisis de Resultados
- **Desempeño Predictivo:** El modelo SVM superó al MLP en la métrica F1 sobre el conjunto de prueba (0.973 vs 0.919). Esto indica que el SVM es más capaz de balancear la precisión y la sensibilidad (*recall*) en datos no vistos.
- **Generalización:** El MLP mostró un "gap" de generalización ligeramente menor (0.018) comparado con el SVM (0.036), lo que sugiere que el MLP fue más consistente entre la validación y la prueba. Sin embargo, la diferencia de desempeño absoluto hace que el SVM sea la opción superior.
- **Eficiencia Computacional:** Existe una diferencia significativa en el tiempo de entrenamiento. El SVM es aproximadamente **8 veces más rápido** que el MLP (0.06s vs 0.46s).

## 4. Análisis Crítico

El experimento presenta puntos clave que deben considerarse para su implementación real:
1. **Estrategia de Muestreo:** Se utilizó un submuestreo (*undersampling*) agresivo para equilibrar las clases en el entrenamiento (solo 300 muestras de la clase "no fraude"). Si bien esto facilita el entrenamiento de modelos en datasets desbalanceados, existe el riesgo de que el modelo no capture la verdadera distribución de la clase mayoritaria, lo que podría elevar la tasa de falsos positivos en producción.
2. **Características (PCA):** El uso de componentes principales (PCA) es efectivo para la reducción de dimensionalidad, pero la pérdida de interpretabilidad de las variables originales dificulta la explicación directa de por qué una transacción fue marcada como fraude (un factor crítico en regulaciones financieras).
3. **Métricas de Evaluación:** El modelo SVM logró un AUC-ROC de 0.992, lo cual es excepcionalmente alto, confirmando la capacidad de separación de las clases en el espacio de características transformado.

## 5. Conclusiones y Recomendaciones Estratégicas

### 5.1 Recomendación de Modelo
Se recomienda la implementación del modelo **SVM (RBF kernel)** para la detección de fraude. Su superioridad en la métrica F1 y su bajísima latencia de entrenamiento lo hacen ideal para entornos de alta transaccionalidad.

### 5.2 Implementación y Escalabilidad
- **Arquitectura de Producción:** Debido a la naturaleza de la detección de fraude (que requiere respuesta inmediata), el modelo SVM debe integrarse en un pipeline de inferencia de baja latencia.
- **Escalabilidad Distribuida:** Para manejar millones de transacciones, se recomienda desplegar el modelo en arquitecturas de analítica distribuida (como Apache Spark con librerías de ML escalables o servicios de inferencia en la nube). Sin embargo, dado que SVM tiene una complejidad cuadrática respecto al número de muestras, para datasets masivos se debería considerar el uso de aproximaciones de kernels o modelos lineales si la escala crece exponencialmente.
- **Mejoras Futuras:**
    * Implementar técnicas de sobremuestreo (ej. **SMOTE**) para tratar el desbalanceo sin perder información de la clase mayoritaria.
    * Integrar variables de contexto (ubicación, tipo de comercio) si la privacidad lo permite, para mejorar la interpretabilidad.
    * Evaluar el modelo bajo un esquema de **costo-sensible**, donde el costo de un falso negativo (fraude no detectado) es significativamente mayor que el de un falso positivo.

