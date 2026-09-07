#import "@preview/versatile-apa:7.2.0": (
  abstract-page, appendix, appendix-outline, title-page,
  versatile-apa as apa-style,
)
#import "@preview/orchid:0.1.0": (
  generate-link as orcid-link, logo-icon as orcid-logo,
)

#set text(lang: "es")

#set document(
  title: [Segmentación estratégica de mercado y análisis de cestas],
)

#show: apa-style.with(
  font-size: 12pt,
  running-head: "",
)

#title-page(
  authors: (
    (
      name: [Harrizon Soler],
      affiliations: (
        "ID-1",
      ),
    ),
  ),
  affiliations: (
    "ID-1": [Universidad Santo Tomás],
  ),

  course: [Data Mining],
  instructor: [Hugo Emmanuel Hernandez Ramirez],
  due-date: datetime(day: 6, month: 9, year: 2026).display(),
)

= Introducción

La implementación de las fases se encuentra en el repositorio: #link("https://github.com/harrysoler/data_mining_activity_2")

El presente informe expone los resultados de un proyecto de minería de datos estructurado en dos fases críticas: la segmentación de clientes mediante análisis RFM y la detección de transacciones fraudulentas mediante aprendizaje supervisado. El estudio integra técnicas de reducción de dimensionalidad, clustering, reglas de asociación y modelado predictivo para proporcionar una base sólida para la toma de decisiones estratégicas en marketing y gestión de riesgos.

En el contexto de la economía digital, la capacidad de extraer valor de grandes volúmenes de datos es fundamental para la competitividad. Este proyecto aborda dos pilares estratégicos: la optimización de la experiencia del cliente a través de la personalización de ofertas y la protección de activos mediante la identificación de patrones de fraude. A través de un enfoque metodológico riguroso, se busca transformar datos transaccionales en inteligencia de negocio accionable.

= Metodología

La investigación se dividió en dos fases técnicas. La Fase I empleó un enfoque de aprendizaje no supervisado, utilizando Análisis de Componentes Principales (PCA) para la reducción de dimensionalidad, _K-Means_ para la segmentación y el algoritmo Apriori para el descubrimiento de patrones de compra. La Fase II se centró en el aprendizaje supervisado, comparando el desempeño de _Support Vector Machines_ (SVM) y redes neuronales (_Multi-Layer Perceptron_) mediante la optimización de hiperparámetros con _GridSearchCV_ y el manejo de clases desbalanceadas.

= Desarrollo de la fase I

== Análisis de Reducción de Dimensionalidad (PCA)

Para optimizar el proceso de clustering y mitigar el ruido en las variables RFM, se aplicó el Análisis de Componentes Principales. Los resultados de la reducción de dimensionalidad se detallan en la Tabla 1, donde se observa una alta eficiencia en la captura de la varianza mediante una arquitectura de solo dos componentes.

#figure(
  table(
    columns: (auto, auto, auto),
    [*Componente principal*], [*Varianza explicada*], [*Varianza acumulada*],
    [_PC1_], [$0.517$], [51.7%],
    [_PC2_], [$0.301$], [*81.8%*],
  ),
  caption: "Varianza explicada por los Componentes Principales.",
)

#figure(
  image("images/phase-1-cell-16-0.png", width: 70%),
  caption: "Análisis de componentes principales (PCA) para reducción de
 dimensionalidad.",
)

Esta capacidad de compresión permite proyectar los datos en un espacio bidimensional que preserva la estructura esencial del comportamiento de compra, facilitando un agrupamiento más eficiente.

== Segmentación de Clientes (Clustering)

La determinación del número óptimo de grupos se realizó mediante un análisis comparativo entre el Método del Codo (Inercia) y el Coeficiente de Silhouette. Aunque el valor máximo de Silhouette se obtuvo con $k=2$ (0.928), lo que indicaba una separación matemática casi perfecta, el Método del Codo y los objetivos de negocio sugirieron una mayor granularida.

#figure(
  image("images/phase-1-cell-24-0.png", width: 60%),
  caption: "Optimización del número de clusters mediante Método del Codo y Silhouette
 Score.",
)

Por tanto, se seleccionó un $K=5$ para permitir una segmentación estratégica que distinga perfiles de alto valor, clientes recurrentes y clientes en riesgo de abandono, priorizando la utilidad operativa sobre la cohesión matemática absoluta.

== Visualizaciones de los Clusters Obtenidos

Las visualizaciones realizadas tras la proyección mediante PCA muestran una separación clara de los segmentos en el espacio de características. Los grupos se distribuyen de manera que los clientes con alta frecuencia y alto gasto se sitúan en un extremo del plano, mientras que los clientes inactivos se agrupan en el extremo opuesto.

#figure(
  image("images/phase-1-cell-28-0.png", width: 60%),
  caption: "Análisis de componentes principales (PCA) para reducción de
 dimensionalidad.",
)

Esta dispersión visual confirma que las dimensiones de recencia, frecuencia y monetario son los factores determinantes que separan los distintos comportamientos de compra en la base de datos.

== Resultados de las Reglas de Asociación

Mediante la aplicación del algoritmo Apriori, se identificaron patrones de compra conjunta con una fuerza estadística notable. Un hallazgo crítico fue la asociación entre diversos sets de tazas de té (ej. _Regency Tea Cup Sets_), los cuales presentaron un _Lift_ superior a 14.

Este valor indica que la probabilidad de adquirir estos productos de forma conjunta es catorce veces mayor que la probabilidad de comprarlos de forma independiente, validando la existencia de canastas de compra predecibles.

= Desarrollo de la fase II

== Resultados de la Optimización de Hiperparámetros

El proceso de optimización mediante *GridSearchCV* fue fundamental para ajustar la capacidad de generalización de los modelos. En el modelo SVM, se determinó que un kernel RBF con un parámetro de regularización $C=10$ optimiza la frontera de decisión, permitiendo un ajuste preciso de los puntos de entrenamiento para minimizar errores en la clasificación de fraude. Por su parte, el modelo MLP se optimizó con una arquitectura de una única capa oculta de 32 neuronas y una función de activación ReLU, permitiendo capturar relaciones no lineales de forma eficiente.

#figure(
  image("images/phase-2-cell-5-0.png", width: 60%),
  caption: "Visualización de la optimización de hiperparámetros mediante búsqueda en
 cuadrícula.",
)

== Comparación entre los modelos SVM y Red Neuronal

Al contrastar ambos modelos, el SVM demostró una superioridad técnica en términos de desempeño predictivo y eficiencia operativa. Los resultados comparativos se presentan en la Tabla 2.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [*Modelo*],
    [*F1 (Validación CV)*],
    [*F1 (Test)*],
    [*Brecha de generalización*],
    [*Tiempo de entrenamiento (s)*],

    [_SVM_], [$0.937$], [*$0.973$*], [$0.036$], [*$0.06$*],
    [_MLP_], [$0.937$], [$0.919$], [*$0.018$*], [$0.46$],
  ),
  caption: "Comparativa de desempeño entre modelos SVM y MLP.",
)

#figure(
  image("images/phase-2-cell-11-1.png", width: 60%),
  caption: "Comparativa de matrices de confusión y rendimiento para modelos SVM y
 MLP.",
)

Aunque el MLP demostró una brecha de generalización ligeramente menor (0.018 frente a
0.036), lo que sugiere una mayor consistencia, el _SVM_ es la opción preferible. El SVM no solo ofrece un _F1-Score_ significativamente más alto en el conjunto de prueba, sino que es aproximadamente ocho veces más rápido en su fase de entrenamiento, factor determinante para sistemas de detección en tiempo real que requieren baja latencia.

#figure(
  image("images/phase-2-cell-12-0.png", width: 60%),
  caption: "Visualización comparativa de los resultados de los modelos de detección de
 fraude.",
)

= Discusión de los resultados

La integración de los resultados muestra que la inteligencia de negocio debe equilibrar la precisión técnica con la aplicabilidad práctica. La alta capacidad de los modelos para identificar patrones (ya sea en el consumo de productos o en la detección de fraude) confirma la validez de la metodología. Sin embargo, es imperativo considerar que el uso de _PCA_ y submuestreo de clases puede limitar la interpretabilidad de las variables, un factor crítico para el cumplimiento regulatorio en sectores donde es necesario explicar la razón de una clasificación negativa (ej. rechazo de una transacción).

= Recomendaciones estratégicas derivadas del análisis

En el ámbito comercial, se recomienda implementar motores de recomendación basados en las reglas de asociación identificadas para maximizar el _cross-selling_. Asimismo, la segmentación _RFM_ debe integrarse en las campañas de marketing para dirigir incentivos de retención a los segmentos de alto valor y tácticas de reactivación a los clientes en riesgo.

En cuanto a la gestión de riesgos, se recomienda el despliegue del modelo SVM en la infraestructura de transacciones para su ejecución en tiempo real. Para futuras iteraciones, se sugiere la implementación de técnicas como _SMOTE_ para mejorar el manejo del desbalanceo de clases y la integración de variables contextuales para incrementar la interpretabilidad del modelo.

= Conclusiones

Este análisis integral confirma la viabilidad de utilizar modelos avanzados de minería de datos para la optimización de la estrategia de negocio. La robustez de la segmentación de clientes y la eficiencia del modelo _SVM_ para la detección de fraude ofrecen herramientas estratégicas para la toma de decisiones. Para garantizar la escalabilidad de estas soluciones ante un crecimiento exponencial del volumen de datos, se recomienda la migración hacia arquitecturas de analítica distribuida (como Apache Spark), asegurando la eficiencia y la precisión en entornos de _Big Data_.

#pagebreak()
#bibliography(
  "bibliography/ref.yml",
  full: true,
  title: [References],
)
