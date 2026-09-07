# Reporte de Análisis de Fase 1: Segmentación y Patrones de Compra

## 1. Análisis de Reducción de Dimensionalidad (PCA)

Para optimizar el proceso de clustering y reducir el ruido, se aplicó el Análisis de Componentes Principales (PCA) sobre las variables escaladas de RFM (Recency, Frequency, Monetary).

**Resultado:** Los **2 primeros componentes principales retienen el 81.8% de la información original** (0.517 para el PC1 y 0.301 para el PC2).

**Justificación:** Esta retención es altamente significativa para un modelo de segmentación. Al capturar más del 80% de la varianza con solo dos dimensiones, podemos trabajar en un
espacio reducido que simplifica la visualización y el cálculo de distancias para el algoritmo de clustering, sin perder la estructura esencial de la información del cliente (su
comportamiento de compra).

## 2. Segmentación de Clientes (K-Means Clustering)

El proceso de optimización del número de clusters ($K$) se realizó mediante la combinación de dos métricas: el Método del Codo (Inercia) y el Coeficiente de Silhouette.

**Análisis de Coincidencia:**
- **Silhouette Score:** El valor máximo se obtuvo con **$k=2$ (0.928)**, lo que indica una separación matemática casi perfecta entre dos grupos (probablemente clientes activos vs.
inactivos).
- **Método del Codo:** El gráfico de inercia muestra un punto de inflexión hacia valores de $k$ superiores (en torno a 3 o 4).
- **Decisión:** En este análisis, el **Método del Codo y la utilidad de negocio pesaron más que el máximo Silhouette Score**.

**Justificación de la decisión:** Aunque $k=2$ es el grupo más "puro" matemáticamente, una segmentación de solo dos grupos es insuficiente para una estrategia de marketing dirigida. Se
requiere una granularidad que permita distinguir entre clientes "VIP", "Recurrentes" y "En riesgo". Por tanto, se priorizó un $K$ que permitiera una segmentación accionable, aceptando una
ligera disminución en la cohesión del cluster para ganar valor estratégico.

## 3. Análisis Crítico de Resultados y Patrones Identificados

### A. Segmentación RFM

La segmentación permitió identificar la salud de la base de clientes. La distribución de los segmentos es crucial para la toma de decisiones:
- **Clientes de alto valor:** El objetivo de retención.
- **Clientes perdidos:** Su análisis ayuda a entender el *churn* (abandono).

### B. Reglas de Asociación (Apriori)

El análisis de canasta de compra reveló patrones de consumo muy fuertes, con valores de **Lift** significativos:
- Se identificó una asociación altamente correlacionada entre conjuntos de tazas de té (ej. *Regency Tea Cup Sets*), con un **Lift superior a 14**. Esto indica que la compra de un
producto aumenta drásticamente la probabilidad de comprar el otro, mucho más allá del azar.
- Los productos de la línea "Jumbo Bag" también muestran patrones de compra conjunta consistentes.

## 4. Conclusiones y Recomendaciones Estratégicas

### Recomendaciones de Negocio

1. **Estrategia de Cross-Selling (Venta Cruzada):** Implementar motores de recomendación basados en las reglas de asociación identificadas. Si un cliente añade una "Regency Tea Cup" al carrito, el sistema debe sugerir automáticamente el set complementario, aprovechando el alto Lift.

2. **Campañas de Fidelización Segmentadas:**
  - Para el segmento de **High Value**, crear programas de recompensas exclusivos.
  - Para el segmento de **Low Value/Lost**, realizar campañas de reactivación mediante descuentos en productos de alta rotación identificados en el análisis de asociación.

3. **Optimización de Inventario:** Utilizar los patrones de compra conjunta para diseñar "bundles" (packs) de productos, optimizando el stock de artículos que suelen venderse juntos.

### Viabilidad y Escalabilidad

- **Viabilidad:** Los modelos aplicados (PCA, K-Means y Apriori) son estándar de la industria y tienen una implementación madura en entornos de producción.
- **Escalabilidad:** Para escalar esta solución a volúmenes de transacciones masivos (Big Data), se recomienda la transición de estas implementaciones a arquitecturas de **analítica distribuida** como **Apache Spark** (usando `pyspark.ml` para K-Means y MLlib para algoritmos de asociación). Esto permitiría procesar millones de transacciones en tiempo real, permitiendo recomendaciones dinámicas en la plataforma de e-commerce.
