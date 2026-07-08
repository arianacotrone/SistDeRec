# 📚 Book Recommendation System - Kaggle In-Class Competition

Este repositorio contiene el script principal de modelado, extracción de características e inferencia desarrollado para una competencia interna de **Kaggle** en la materia **Sistemas de Recomendación**, correspondiente a la **Maestría en Ciencia de Datos del ITBA**.

El pipeline implementa un enfoque de **dos etapas (Two-Stage Recommendation System)**:
1. **Generación de Candidatos (Retrieval):** Estrategia multi-modelo para recuperar libros potencialmente relevantes combinando enfoques colaborativos, secuenciales y demográficos.
2. **Re-ranking:** Un modelo de aprendizaje para ordenamiento (**LambdaMART / Ranker**) utilizando **CatBoostRanker** optimizado bajo la función de pérdida `QueryRMSE`.

---

## 🏗️ Arquitectura del Sistema

El pipeline procesa los datos mediante un flujo híbrido estructurado de la siguiente manera:

### 1. Extracción y Limpieza de Metadatos
* **Limpieza de Texto:** Normalización Unicode (NFD) y regex para consolidar géneros literarios.
* **Imputación de Géneros de Lectores:** Sistema de votación cruzada basado en la distribución de nombres propios.
* **Normalización Geográfica:** Reglas heurísticas de limpieza para agrupar variables de residencia a nivel país.

### 2. Estrategia Multi-Candidatos (Retrieval)
Para mitigar el sesgo de popularidad y capturar diferentes intenciones, para cada usuario se genera un pool de candidatos combinando:
* **ALS (Alternating Least Squares):** Filtrado colaborativo implícito ponderado mediante métricas BM25 ($Factors=128$).
* **BPR (Bayesian Personalized Ranking):** Optimización basada en pares para ranking personalizado implícito.
* **Modelo Secuencial:** Cadena de transición temporal basada en ventanas de co-ocurrencia de lectura ($365\text{ días}$).
* **Item-Based CF:** Similitud de ítems calculada mediante co-ocurrencia normalizada de lectores.
* **Filtros de Afinidad Demográfica:** Agrupaciones por rango etario, país de residencia y género del lector.

### 3. Ingeniería de Características (Feature Engineering)
Se computan en batch cerca de **23 variables** predictivas por cada par `(usuario, libro)`, clasificadas en:
* **Scores de Modelos:** Predicciones crudas y normalizadas de ALS y BPR.
* **Features del Ítem:** Popularidad total, popularidad reciente (últimos 90 días), rating bayesiano y puntuación de frescura temporal ajustada por año de edición.
* **Matches de Contexto:** Indicadores binarios de coincidencia entre el ítem y las preferencias históricas del usuario (autor, género, editorial).
* **Afinidad Cardiovascular Embebida:** Producto punto matemático directo entre los vectores de factores latentes de los usuarios (`user_factors`) y los vectores promedio de autores (`autor_vectors`) derivados del espacio latente de ALS.

### 4. Re-Ranking con CatBoost
* **Entrenamiento:** Validación cruzada basada en cortes temporales históricos (Pre-2023 vs. Post-2023).
* **Optimización de Grupo:** Los pools se agrupan mediante `group_id` (`id_lector`) utilizando la función de pérdida `QueryRMSE` para optimizar directamente las métricas de evaluación de ranking del output final.

---

## 🛠️ Requisitos e Instalación

El entorno requiere Python 3.8+ y está configurado explícitamente para procesamiento monohilo (`OPENBLAS_NUM_THREADS=1`) para garantizar reproducibilidad en las operaciones matriciales.

### Dependencias Principales
```text
numpy
pandas
scipy
scikit-learn
implicit
catboost
sqlite3
