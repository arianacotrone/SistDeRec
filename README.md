# Sistema de Recomendación de Libros - Kaggle In-Class ITBA

Este repositorio contiene la solución desarrollada para la competencia interna de Kaggle de la materia **Sistemas de Recomendación**, correspondiente a la **Maestría en Ciencia de Datos del Instituto Tecnológico de Buenos Aires (ITBA)**.

El objetivo del proyecto es predecir los próximos libros que leerán y calificarán los usuarios basándose en su historial de interacciones, metadatos de los libros y perfiles demográficos de los lectores.

El pipeline está diseñado bajo un enfoque moderno de **etapas**, optimizado para manejar el desbalance de clases y la eficiencia en cómputo:

[ Historial de Interacciones ]
│
▼
┌──────────────────────────────┐
│  1. GENERACIÓN DE CANDIDATOS │ (Filtra de +460.000 a solo ~150-250 ítems por usuario)
└──────────────┬───────────────┘
│ ──> ALS, BPR, Secuencial (Transiciones), Item-CF,
│     Autor Favorito, Demografía (Edad/País/Género).
▼
┌──────────────────────────────┐
│    2. EXTRACCIÓN DE FEATURES │ (Computa ~23 variables de afinidad y contexto)
└──────────────┬───────────────┘
▼
┌──────────────────────────────┐
│    3. RE-RANKING (CatBoost)  │ (Modelo QueryRMSE para ordenar los candidatos)
└──────────────┬───────────────┘
▼
[ Top 20 Recomendaciones ]

## 🚀 Arquitectura del Sistema

### 1. Etapa de Recuperación (Candidate Generation)
Para cada usuario objetivo, se consolida una lista de candidatos únicos provenientes de múltiples fuentes para asegurar tanto **relevancia** como **serendipia**:
* **Modelos Matriciales Implícitos:** Ajuste de matrices dispersas ponderadas por `BM25`. Se entrenan algoritmos de **ALS** (*Alternating Least Squares*) y **BPR** (*Bayesian Personalized Ranking*).
* **Modelo Secuencial:** Matriz de transición basada en ventanas temporales de co-ocurrencia para capturar qué libros se leen inmediatamente después de otros.
* **Item-Based Collaborative Filtering:** Similitud de ítems calculada mediante índices de co-ocurrencia normalizados.
* **Filtros Demográficos & Heurísticas:** Fallbacks basados en popularidad segmentada por rangos de edad, país de residencia (con limpieza geográfica pesada) y género del lector.

### 2. Etapa de Clasificación (Reranking)
Se extraen un total de **23 features manuales** que cruzan los scores de los modelos base, popularidad suavizada (Bayesiana), tasas de consumo del autor, y variables de *matching* categórico (género literario favorito, editorial favorita).

El ordenamiento final lo realiza **CatBoostRanker** utilizando la función de pérdida `QueryRMSE`. El dataset se agrupa por `id_lector` para simular la optimización de listas orientada a métricas de ranking (como NDCG o MAP).

---

## 🛠️ Requisitos Técnicos

El script está optimizado para ejecutarse en entornos de un solo hilo de procesamiento numérico para evitar colisiones de memoria en arquitecturas compartidas (`OPENBLAS_NUM_THREADS=1`).

### Dependencias Principales
* `numpy` & `pandas` (Procesamiento de datos)
* `sqlite3` (Motor de almacenamiento de origen)
* `scipy` (Matrices dispersas)
* `implicit` (Algoritmos ALS y BPR acelerados)
* `catboost` (Motor de Gradient Boosting para Ranking)
* `scikit-learn` (Preprocesamiento)

---

## 📂 Estructura de Datos Requerida

Para ejecutar el script, el directorio raíz debe contar con los siguientes archivos:

* `data.db`: Base de datos SQLite que contiene las tablas `interacciones`, `libros` y `lectores`.
* `ejemplo.csv`: Archivo de muestra de la competencia que define los `id_lector` objetivo a predecir.
* `main.py`: Código principal del algoritmo.

---

## ⚙️ Flujo de Ejecución

El pipeline ejecuta automáticamente las siguientes fases al correr el script:

1.  **Carga y Enriquecimiento:** Lee los datos de SQLite, procesa texto mediante normalización Unicode para remover acentos/caracteres extraños, e imputa géneros de lectores faltantes cruzando estadísticas de nombres propios.
2.  **Split Temporal (Fase A):** Genera un corte con fecha límite (`2023-01-01`). Entrena los generadores de candidatos en el pasado y evalúa contra el "futuro" local para entrenar el `CatBoostRanker` con una estrategia de *Early Stopping*.
3.  **Inferencia Final (Fase B):** Re-entrena las lógicas de candidatos sobre el 100% de los datos históricos disponibles, predice las probabilidades de ranking con el CatBoost guardado y genera el archivo final de *submission*.

Filtrado final de salida:
```python
# Genera el output oficial con formato Kaggle
catboost_sub3.csv
