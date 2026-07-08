import os, csv, time, warnings
from collections import defaultdict, Counter

warnings.filterwarnings('ignore', category=UserWarning) # Saca el UserWarning: Esto es una advertencia

import numpy as np
import pandas as pd
import sqlite3
import implicit
from catboost import CatBoostRanker, Pool
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import bm25_weight
from sklearn.preprocessing import LabelEncoder
import unicodedata
import re

# Configuración de un solo hilo
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 

# ============================================================
# CONFIG
# ============================================================
FECHA_CORTE      = "2023-01-01"
N_CANDS_ALS      = 100   
N_CANDS_BPR      = 60    
N_CANDS_AUTOR    = 20    
N_CANDS_DEMO     = 20    
MIN_POSITIVOS    = 1     

# ALS / BPR
ALS_FACTORS       = 128
ALS_REG           = 0.1
ALS_ITER          = 50
BPR_FACTORS       = 128
BPR_ITER          = 100
BPR_LR            = 0.01
BPR_REG           = 0.01
BM25_K1           = 100
BM25_B            = 0.8

# Secuencial
SEQ_WINDOW_DIAS  = 365
TOP_K_TRANS      = 50
SEQ_RECENT_K     = 5

# Item-CF
TOP_SIM_ITEMS    = 30
MAX_BOOKS_COOC   = 50

# CatBoost
CB_PARAMS = {
    "iterations":    300,            
    "learning_rate": 0.1,            
    "loss_function": "QueryRMSE",    
    "thread_count":  -1,              
    "random_seed":   42,
    "verbose":       20,             
    "train_dir":     "catboost_info",
    "depth":         7,              
    "rsm":           0.75,           # Dejamos que vea el 75% de las variables. Obliga a usar contexto sin ignorar ALS/BPR
    "l2_leaf_reg":   3.0             # Regularización suave para estabilizar las predicciones en 2024
}
EARLY_STOPPING = 50

# Demográficos
GRUPOS_EDAD_BINS = [0, 25, 35, 50, 65, 200]
GRUPOS_EDAD_LABS = ['<25', '25-35', '35-50', '50-65', '65+']
TOP_PAISES       = 40
RATING_CORTE     = 7

BASE        = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE, "data.db")
EJEMPLO_CSV = os.path.join(BASE, "ejemplo.csv")
OUTPUT_CSV  = os.path.join(BASE, "catboost_sub3.csv")

# ============================================================
# UTILIDADES
# ============================================================
def nuevo_cand():
    return {
        'als_score': 0.0, 'bpr_score': 0.0,
        'from_als': 0, 'from_bpr': 0, 'from_seq': 0,
        'from_itemcf': 0, 'from_autor': 0, 'from_demo': 0,
    }

def build_sparse_bm25(df_sub, u2i, i2i):
    rows = df_sub['id_lector'].map(u2i)
    cols = df_sub['id_libro'].map(i2i)
    mask = rows.notna() & cols.notna()
    sp = csr_matrix(
        (df_sub.loc[mask, 'rating'].values,
         (rows[mask].astype(int), cols[mask].astype(int))),
        shape=(len(u2i), len(i2i))
    )
    return bm25_weight(sp, K1=BM25_K1, B=BM25_B).tocsr()

def batch_recommend(modelo, sparse, user_indices, N, u2i_inv, i2i_inv, filter_sparse=None):
    BATCH = 500
    result = {}
    for start in range(0, len(user_indices), BATCH):
        batch_u = user_indices[start:start + BATCH]
        batch_idx = np.array([u2i_inv[u] for u in batch_u if u in u2i_inv])
        real_u = [u for u in batch_u if u in u2i_inv]
        if len(batch_idx) == 0:
            continue
        sub_sp = sparse[batch_idx]
        ids_rec, scores = modelo.recommend(
            batch_idx, sub_sp, N=N, filter_already_liked_items=True
        )
        for i, u in enumerate(real_u):
            recs = {}
            for j in range(ids_rec.shape[1]):
                iidx = int(ids_rec[i, j])
                if iidx >= 0:
                    recs[i2i_inv[iidx]] = float(scores[i, j])
            if recs:
                mn, mx = min(recs.values()), max(recs.values())
                rng = mx - mn + 1e-8
                recs = {k: (v - mn) / rng for k, v in recs.items()}
            result[u] = recs
    return result

def build_transitions(df_sub):
    df_s = df_sub.sort_values(['id_lector', 'fecha'])
    trans = defaultdict(Counter)
    for uid, grp in df_s.groupby('id_lector', sort=False):
        books   = grp['id_libro'].tolist()
        dates   = grp['fecha'].tolist()
        ratings = grp['rating'].tolist()
        for i in range(len(books) - 1):
            if (dates[i+1] - dates[i]).days <= SEQ_WINDOW_DIAS:
                trans[books[i]][books[i+1]] += ratings[i] / 10.0
    return {b: c.most_common(TOP_K_TRANS) for b, c in trans.items()}

def build_item_sim(lector_libros_pos_sub, libro_lectores_sub):
    co = defaultdict(Counter)
    for uid, bs in lector_libros_pos_sub.items():
        bl = sorted(bs)[:MAX_BOOKS_COOC]
        for i in range(len(bl)):
            for j in range(i+1, len(bl)):
                co[bl[i]][bl[j]] += 1
                co[bl[j]][bl[i]] += 1
    sim = {}
    for ba, co_b in co.items():
        ra = max(len(libro_lectores_sub.get(ba, set())), 1)
        sims = [(bb, cnt / (ra**0.5 * max(len(libro_lectores_sub.get(bb,set())),1)**0.5))
                for bb, cnt in co_b.items()]
        sims.sort(key=lambda x: -x[1])
        sim[ba] = sims[:TOP_SIM_ITEMS]
    return sim

def limpiar_texto(series):
    def remover_acentos(txt):
        if not isinstance(txt, str):
            return ""
        return "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
    series_limpia = series.apply(remover_acentos)
    series_limpia = series_limpia.astype(str).str.replace(r"[^a-zA-Z\s]", " ", regex=True)
    series_limpia = series_limpia.str.lower()
    regex_duplicados = re.compile(r"([a-z])\1+")
    series_limpia = series_limpia.apply(lambda x: regex_duplicados.sub(r"\1", x))
    return series_limpia

def generar_asignador(df_resumen, dict_ref):
    df_asig = df_resumen[df_resumen["genero"].isna() | (df_resumen["genero"] == "-")][["nombre", "genero"]].copy()
    df_asig["nombre_limpio"] = limpiar_texto(df_asig["nombre"]).str.split(r"\s+")
    df_asig = df_asig.explode("nombre_limpio")
    df_asig = df_asig[df_asig["nombre_limpio"].str.len() > 2]
    df_asig = df_asig.merge(dict_ref, on="nombre_limpio", how="inner")
    df_asig = df_asig.rename(columns={"genero_sugerido": "genero_nuevo"})
    votos = df_asig.groupby(["nombre", "genero_nuevo"]).size().reset_index(name="n")
    votos = votos.sort_values("n", ascending=False).drop_duplicates("nombre").reset_index(drop=True)
    return votos[["nombre", "genero_nuevo"]]

def super_limpiador_geografico(texto):
    t = str(texto).lower().strip()
    if t in ['', 'none', 'nan', '!', '.', ':)', 'aa', 'ab', 'a.', 'x', 'xxxx', 'xxxxxxxxxx', 'ninguna', 'no especificado', 'no']:
        return 'Desconocido'
    if any(fanta in t for fanta in ['narnia', 'gotham', 'mordor', 'hogsmeade', 'mundo mundial', 'ciudad perdida', 'unaciudad', 'en otra dimensión']):
        return 'Desconocido'
    ciudades_espana = ['madrid', 'barcelona', 'bcn', 'valencia', 'valència', 'sevilla', 'andalucía', 'galicia', 'asturias', 'alicante', 'alacant', 'murcia', 'málaga', 'malaga', 'zaragoza', 'bilbao', 'coruña', 'vigo', 'granada', 'cadiz']
    if any(c in t for c in ciudades_espana) or 'españa' in t: return 'España'
    ciudades_argentina = ['buenos aires', 'bs as', 'bsas', 'caba', 'capital federal', 'rosario', 'córdoba', 'cordoba', 'mendoza', 'santa fe', 'tucuman', 'salta', 'neuquen', 'mar del plata']
    if any(c in t for c in ciudades_argentina) or 'argentina' in t: return 'Argentina'
    ciudades_mexico = ['mexico', 'méxico', 'cdmx', 'guadalajara', 'monterrey', 'zapopan', 'puebla', 'tijuana']
    if any(c in t for c in ciudades_mexico) or 'mx' == t: return 'México'
    ciudades_colombia = ['bogota', 'bogotá', 'medellin', 'cali', 'barranquilla', 'cartagena']
    if any(c in t for c in ciudades_colombia) or 'colombia' in t: return 'Colombia'
    if any(c in t for c in ['santiago', 'chile', 'viña del mar', 'valparaiso']) or 'stgo' in t: return 'Chile'
    if any(c in t for c in ['caracas', 'maracaibo', 'maracay', 'venezuela']): return 'Venezuela'
    if 'montevideo' in t or 'uruguay' in t: return 'Uruguay'
    if any(c in t for c in ['usa', 'united states', 'washington', 'new york', 'miami']): return 'Estados Unidos'
    if '-' in t:
        posible = t.split('-')[-1].strip()
        if len(posible) > 2: return posible.strip().capitalize()
    if re.match(r'^\d+$', t): return 'Desconocido'
    return t.strip().capitalize()
    
def calcular_afinidad_autores_als(model_als, u2i, i2i, libro_autor_d):
    try:
        item_factors = model_als.item_factors.to_numpy()
        user_factors = model_als.user_factors.to_numpy()
    except AttributeError:
        item_factors = np.asarray(model_als.item_factors)
        user_factors = np.asarray(model_als.user_factors)
        
    autor_to_idxs = defaultdict(list)
    for bid, item_idx in i2i.items():
        autor = libro_autor_d.get(bid, None)
        if autor and item_idx < len(item_factors):
            autor_to_idxs[autor].append(item_idx)
                
    autor_vectors = {}
    for autor, idxs in autor_to_idxs.items():
        if idxs:
            autor_vectors[autor] = item_factors[idxs].mean(axis=0)
    
    return user_factors, autor_vectors

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("1. Cargando datos...")
t0 = time.time()
con = sqlite3.connect(DB_PATH)
df_interacciones = pd.read_sql_query("SELECT id_lector, id_libro, rating, fecha FROM interacciones", con)
df_libros = pd.read_sql_query("SELECT id_libro, autor, genero, editorial, anio_edicion, isbn FROM libros", con)
df_lectores_meta = pd.read_sql_query("SELECT id_lector, nombre, genero, nacimiento, vive_en FROM lectores", con)
con.close()

df_interacciones['fecha'] = pd.to_datetime(df_interacciones['fecha'], dayfirst=True, errors='coerce')
df_interacciones['fecha'] = df_interacciones['fecha'].fillna(df_interacciones['fecha'].dropna().min())
print(f"   {len(df_interacciones)} interacciones en {time.time()-t0:.1f}s")

lectores_objetivo = []
seen_ej = set()
with open(EJEMPLO_CSV) as f:
    for row in csv.DictReader(f):
        if row['id_lector'] not in seen_ej:
            seen_ej.add(row['id_lector'])
            lectores_objetivo.append(row['id_lector'])
print(f"   {len(lectores_objetivo)} lectores objetivo")

# ============================================================
# LIMPIEZA DE ENTRADAS Y ENRIQUECIMIENTO GLOBAL
# ============================================================
print("3. Limpiando e indexando bases de datos globales...")
df_libros['genero'] = df_libros['genero'].astype(str).str.strip().str.lower()

diccionario_generos = {
    'narrativa': 'Narrativa y ficción', 'novela': 'Narrativa y ficción', 'literatura contemporánea': 'Narrativa y ficción',
    'clásicos': 'Clásicos', 'humor': 'Humor y varios', 'romántica, erótica': 'Romántica y erótica',
    'fantasía y ciencia ficción': 'Fantasía y ciencia ficción', 'misterio, terror y suspense': 'Misterio, thriller y terror',
    'novela negra': 'Misterio, thriller y terror', 'histórica y aventuras': 'Historia y novela histórica',
    'infantil y juvenil': 'Infantil y juvenil', 'cómics, novela gráfica': 'Cómics y novela gráfica',
    'poesía, teatro': 'Arte, poesía y teatro', 'biografías y memorias': 'Biografías y no ficción',
    'no ficción': 'Biografías y no ficción', 'ensayo': 'Biografías y no ficción',
    'autoayuda y espiritualidad': 'Autoayuda y desarrollo personal', 'medicina': 'Salud, medicina y nutrición',
    'economía, empresa, marketing': 'Economía y negocios', 'naturaleza y ciencia': 'Ciencia, academia y derecho',
    'cocina': 'Hogar, viajes y deportes', 'varios': 'Humor y varios', '': 'Otros'
}
df_libros['genero'] = df_libros['genero'].fillna('-')
df_libros['genero_limpio'] = df_libros['genero'].map(diccionario_generos).fillna(df_libros['genero'])
df_libros['genero'] = df_libros['genero_limpio'].str.capitalize()
df_libros = df_libros.drop(columns=['genero_limpio'])

df_libros['anio_edicion'] = pd.to_numeric(df_libros['anio_edicion'], errors='coerce')
df_libros['anio_ed_num'] = df_libros['anio_edicion'].fillna(df_libros['anio_edicion'].median())
df_libros['isbn'] = pd.to_numeric(df_libros['isbn'], errors='coerce').astype(float)

# Creación de mapeos de libros estáticos e inmutables (¡AQUÍ OPTIMIZAMOS!)
libro_autor_d = df_libros.set_index('id_libro')['autor'].to_dict()
libro_genero_d = df_libros.set_index('id_libro')['genero'].to_dict()
libro_editorial_d = df_libros.set_index('id_libro')['editorial'].to_dict()
libros_por_autor = df_libros.groupby('autor')['id_libro'].apply(list).to_dict()

# Imputación de géneros de lectores
lectores_palabras = df_lectores_meta[df_lectores_meta["genero"].notna() & (df_lectores_meta["genero"] != "-")][["nombre", "genero"]].copy()
lectores_palabras["nombre_limpio"] = limpiar_texto(lectores_palabras["nombre"])
lectores_palabras["nombre_limpio"] = lectores_palabras["nombre_limpio"].str.split(r"\s+")
lectores_palabras = lectores_palabras.explode("nombre_limpio")
lectores_palabras = lectores_palabras[lectores_palabras["nombre_limpio"].str.len() > 2]

conteos = lectores_palabras.groupby(["nombre_limpio", "genero"]).size().reset_index(name="n")
conteos["total_votos"] = conteos.groupby("nombre_limpio")["n"].transform("sum")
conteos["porcentaje_dominante"] = conteos["n"] / conteos["total_votos"]
palabras_contaminadas = conteos[(conteos["total_votos"] > 50) & (conteos["porcentaje_dominante"] < 0.75)]["nombre_limpio"].unique()

diccionario_filtrado = conteos[~conteos["nombre_limpio"].isin(palabras_contaminadas)].copy()
diccionario_filtrado = diccionario_filtrado.sort_values("n", ascending=False).drop_duplicates("nombre_limpio").reset_index(drop=True)
diccionario_filtrado = diccionario_filtrado.rename(columns={"genero": "genero_sugerido"})[["nombre_limpio", "genero_sugerido"]]

lectores_completo = df_lectores_meta.copy()
asignador = generar_asignador(lectores_completo, diccionario_filtrado)
lectores_completo = lectores_completo.merge(asignador, on="nombre", how="left")
lectores_completo["genero"] = lectores_completo["genero"].replace("-", np.nan).fillna(lectores_completo["genero_nuevo"])
lectores_completo = lectores_completo.drop(columns=["genero_nuevo"])
df_lectores_meta = lectores_completo.copy()

df_lectores_meta['pais'] = df_lectores_meta['vive_en'].apply(super_limpiador_geografico)
top_paises = ['España', 'Argentina', 'México', 'Colombia', 'Chile', 'Venezuela', 'Uruguay', 'Estados Unidos', 'Desconocido']
df_lectores_meta['pais'] = df_lectores_meta['pais'].apply(lambda x: x if x in top_paises else 'Otro')
df_lectores_meta['nacimiento'] = pd.to_numeric(df_lectores_meta['nacimiento'], errors='coerce').fillna(1985)

# Convertir metadatos de lectores a Diccionario Global para acceso O(1) rápido
lectores_meta_dict = df_lectores_meta.set_index('id_lector').to_dict('index')

# ============================================================
# 2. SPLIT TEMPORAL
# ============================================================
print("2. Split temporal...")
CORTE = pd.Timestamp(FECHA_CORTE)
df_pasado  = df_interacciones[df_interacciones['fecha'] < CORTE].copy()
df_futuro  = df_interacciones[df_interacciones['fecha'] >= CORTE].copy()
interacciones_futuras = df_futuro.groupby('id_lector')['id_libro'].apply(set).to_dict()
usuarios_con_futuro = [u for u, s in interacciones_futuras.items() if len(s) >= MIN_POSITIVOS]
futuro_rating_d = df_futuro.groupby(['id_lector', 'id_libro'])['rating'].max().to_dict()

# ============================================================
# 3. PIPELINE DE EXTRACCIÓN DE FEATURES OPTIMIZADO
# ============================================================
def pipeline_features_y_candidatos(df_sub, target_users=None):
    
    print("   -> Generando mapeos y modelos base...")
    anio_max = df_sub['fecha'].dt.year.max()
    
    # Trabajamos localmente con una copia reducida
    df_libros_local = df_libros[['id_libro', 'anio_ed_num']].copy()
    df_libros_local['anios_disp'] = (anio_max - df_libros_local['anio_ed_num']).clip(lower=1)
    
    stats_libros = df_sub.groupby('id_libro').agg(pop_total=('rating', 'count'), rating_mean=('rating', 'mean')).reset_index()
    fecha_limite_reciente = df_sub['fecha'].max() - pd.Timedelta(days=90)
    
    reciente = df_sub[df_sub['fecha'] >= fecha_limite_reciente]
    pop_reciente = reciente.groupby('id_libro').size().rename('pop_reciente')
    stats_libros = stats_libros.merge(pop_reciente, on='id_libro', how='left').fillna(0)
    
    C_g, m_g = df_sub['rating'].mean(), 10
    stats_libros['rating_bay'] = ((stats_libros['pop_total'] / (stats_libros['pop_total'] + m_g)) * stats_libros['rating_mean'] + (m_g / (stats_libros['pop_total'] + m_g)) * C_g)
    stats_libros = stats_libros.merge(df_libros_local[['id_libro','anios_disp']], on='id_libro', how='left').fillna(10)
    stats_libros['inter_ajustadas'] = stats_libros['pop_total'] / np.log1p(stats_libros['anios_disp'])
    stats_libros['score_ct'] = (stats_libros['inter_ajustadas'] / stats_libros['inter_ajustadas'].max()).clip(0, 1)
    
    # Mapeo rápido usando el diccionario global estático de autores
    df_sub_autor = df_sub['id_libro'].map(libro_autor_d)
    pop_autor = df_sub.groupby(df_sub_autor).size().rename('pop_autor_total').reset_index().rename(columns={'id_libro': 'autor'})
    
    libro_pop_d = dict(zip(stats_libros['id_libro'], stats_libros['pop_total']))
    libro_rec_d = dict(zip(stats_libros['id_libro'], stats_libros['pop_reciente']))
    libro_bay_d = dict(zip(stats_libros['id_libro'], stats_libros['rating_bay']))
    libro_ct_d = dict(zip(stats_libros['id_libro'], stats_libros['score_ct']))
    libro_pop_autor_d = dict(zip(pop_autor['autor'], pop_autor['pop_autor_total']))
    
    leidos = df_sub.groupby('id_lector')['id_libro'].apply(set).to_dict()
    pos_sub = df_sub[df_sub['rating'] >= RATING_CORTE]
    libro_lectores_sub = defaultdict(set)
    lector_libros_pos_sub = defaultdict(set)
    for r in pos_sub.itertuples(index=False):
        libro_lectores_sub[r.id_libro].add(r.id_lector)
        lector_libros_pos_sub[r.id_lector].add(r.id_libro)
        
    actividad_lector = df_sub.groupby('id_lector').size().to_dict()
    avg_rating_user = df_sub.groupby('id_lector')['rating'].mean().to_dict()
    
    pop_of_read_pop = df_sub['id_libro'].map(libro_pop_d).fillna(0)
    afinidad_mainstream = df_sub.groupby('id_lector').apply(lambda x: pop_of_read_pop.loc[x.index].mean()).to_dict()
    
    ultima_fecha_u = df_sub.groupby('id_lector')['fecha'].max().to_dict()
    fecha_ref = df_sub['fecha'].max()
    dias_desde_ultima = {u: (fecha_ref - f).days for u, f in ultima_fecha_u.items()}
    
    # Optimización del cálculo por autor por usuario
    df_sub_autores = df_sub['id_libro'].map(libro_autor_d)
    user_autor_stats = df_sub.groupby(['id_lector', df_sub_autores]).agg(n_libros_autor=('id_libro','count'), rating_promedio_autor=('rating','mean')).reset_index().rename(columns={'id_libro': 'autor'})
    user_autor_d = {(r.id_lector, r.autor): (int(r.n_libros_autor), float(r.rating_promedio_autor)) for r in user_autor_stats.itertuples()}
    
    ultimo_libro_d = df_sub.sort_values('fecha').groupby('id_lector')['id_libro'].last().to_dict()
    ultimo_autor_d = {u: libro_autor_d.get(b) for u, b in ultimo_libro_d.items()}
    
    # User Favoritos optimizado con .map()
    df_con_meta_gen = df_sub['id_libro'].map(libro_genero_d)
    df_con_meta_edi = df_sub['id_libro'].map(libro_editorial_d)
    user_gen_fav = df_sub.groupby(['id_lector', df_con_meta_gen]).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('id_lector').set_index('id_lector')['id_libro'].to_dict()
    user_edi_fav = df_sub.groupby(['id_lector', df_con_meta_edi]).size().reset_index(name='c').sort_values('c', ascending=False).drop_duplicates('id_lector').set_index('id_lector')['id_libro'].to_dict()
    
    trans_d = build_transitions(df_sub)
    sim_d = build_item_sim(lector_libros_pos_sub, libro_lectores_sub)
    
    # Demografía agregada usando mapeo sobre vectores de df_sub
    df_m = pd.DataFrame({'id_lector': df_sub['id_lector'], 'id_libro': df_sub['id_libro']})
    df_m['nacimiento'] = df_m['id_lector'].apply(lambda x: lectores_meta_dict.get(x, {}).get('nacimiento', 1985))
    df_m['pais'] = df_m['id_lector'].apply(lambda x: lectores_meta_dict.get(x, {}).get('pais', 'Desconocido'))
    df_m['genero'] = df_m['id_libro'].map(libro_genero_d)
    
    df_m['edad'] = anio_max - df_m['nacimiento']
    df_m['grupo_edad'] = pd.cut(df_m['edad'], bins=GRUPOS_EDAD_BINS, labels=GRUPOS_EDAD_LABS).astype(str)
    
    pop_gen = df_m.groupby(['genero', 'id_libro']).size().reset_index(name='c').sort_values(['genero', 'c'], ascending=False).groupby('genero').head(N_CANDS_DEMO)
    demo_gen_d = pop_gen.groupby('genero')['id_libro'].apply(list).to_dict()
    
    pop_pais = df_m.groupby(['pais', 'id_libro']).size().reset_index(name='c').sort_values(['pais', 'c'], ascending=False).groupby('pais').head(N_CANDS_DEMO)
    demo_pais_d = pop_pais.groupby('pais')['id_libro'].apply(list).to_dict()
    
    pop_edad = df_m.groupby(['grupo_edad', 'id_libro']).size().reset_index(name='c').sort_values(['grupo_edad', 'c'], ascending=False).groupby('grupo_edad').head(N_CANDS_DEMO)
    demo_edad_d = pop_edad.groupby('grupo_edad')['id_libro'].apply(list).to_dict()
    
    # Modelos Matriciales
    print("   -> Seteando e indexando matrices implícitas...")
    unique_users = df_sub['id_lector'].unique()
    unique_items = df_libros['id_libro'].unique()
    u2i = {u: i for i, u in enumerate(unique_users)}
    i2i = {b: i for i, b in enumerate(unique_items)}
    u2i_inv = {i: u for u, i in u2i.items()}
    i2i_inv = {i: b for b, i in i2i.items()}
    
    fallback_books = stats_libros.sort_values('score_ct', ascending=False)['id_libro'].head(50).tolist()
    
    sparse_bm25 = build_sparse_bm25(df_sub, u2i, i2i)
    
    print("   -> Ajustando ALS...")
    model_als = implicit.als.AlternatingLeastSquares(factors=ALS_FACTORS, regularization=ALS_REG, iterations=ALS_ITER, random_state=42, num_threads=1)
    model_als.fit(sparse_bm25, show_progress=False)

    user_factors, autor_vectors = calcular_afinidad_autores_als(model_als, u2i, i2i, libro_autor_d)
    
    print("   -> Ajustando BPR...")
    model_bpr = implicit.bpr.BayesianPersonalizedRanking(factors=BPR_FACTORS, learning_rate=BPR_LR, regularization=BPR_REG, iterations=BPR_ITER, random_state=42, num_threads=1)
    model_bpr.fit(sparse_bm25, show_progress=False)
    
    users_to_process = target_users if target_users is not None else unique_users
    print(f"   -> Recuperando candidatos para {len(users_to_process)} usuarios...")
    
    recs_als = batch_recommend(model_als, sparse_bm25, users_to_process, N_CANDS_ALS, u2i, i2i_inv)
    recs_bpr = batch_recommend(model_bpr, sparse_bm25, users_to_process, N_CANDS_BPR, u2i, i2i_inv)
    
    rows_out = []
    
    for uid in users_to_process:
        u_leidos = leidos.get(uid, set())
        cands = defaultdict(nuevo_cand)
        
        # 1. ALS
        for b, sc in recs_als.get(uid, {}).items():
            cands[b]['als_score'] = sc
            cands[b]['from_als'] = 1
            
        # 2. BPR
        for b, sc in recs_bpr.get(uid, {}).items():
            cands[b]['bpr_score'] = sc
            cands[b]['from_bpr'] = 1
            
        # 3. Secuencial
        u_hist = df_sub[df_sub['id_lector'] == uid].sort_values('fecha', ascending=False).head(SEQ_RECENT_K)['id_libro'].tolist()
        for b_orig in u_hist:
            for b_next, r_w in trans_d.get(b_orig, []):
                if b_next not in u_leidos:
                    cands[b_next]['from_seq'] = 1
                    
        # 4. Item-CF
        u_pos = lector_libros_pos_sub.get(uid, set())
        for b_pos in u_pos:
            for b_sim, _ in sim_d.get(b_pos, []):
                if b_sim not in u_leidos:
                    cands[b_sim]['from_itemcf'] = 1
                    
        # 5. Autor Favorito
        u_autores = {libro_autor_d.get(b) for b in u_leidos if libro_autor_d.get(b)}
        for aut in u_autores:
            for b_aut in libros_por_autor.get(aut, [])[:N_CANDS_AUTOR]:
                if b_aut not in u_leidos:
                    cands[b_aut]['from_autor'] = 1
                    
        # 6. Demográficos 
        meta_u = lectores_meta_dict.get(uid)
        if meta_u:
            g_u = meta_u.get('genero', '-')
            p_u = meta_u.get('pais', 'Desconocido')
            n_u = meta_u.get('nacimiento', 1985)
            ed_u = anio_max - n_u
            ge_u = str(pd.cut([ed_u], bins=GRUPOS_EDAD_BINS, labels=GRUPOS_EDAD_LABS)[0])
            
            for b_d in demo_gen_d.get(g_u, []):
                if b_d not in u_leidos: cands[b_d]['from_demo'] = 1
            for b_d in demo_pais_d.get(p_u, []):
                if b_d not in u_leidos: cands[b_d]['from_demo'] = 1
            for b_d in demo_edad_d.get(ge_u, []):
                if b_d not in u_leidos: cands[b_d]['from_demo'] = 1
                
        if not cands:
            for b_f in fallback_books[:30]:
                if b_f not in u_leidos: cands[b_f]['from_demo'] = 1
                
        # Construcción compacta de las ~25 features manuales
        act = actividad_lector.get(uid, 0)
        avg_r = avg_rating_user.get(uid, 7.0)
        main_u = afinidad_mainstream.get(uid, 0)
        dias_u = dias_desde_ultima.get(uid, 999)
        u_gen_f = user_gen_fav.get(uid, '-')
        u_edi_f = user_edi_fav.get(uid, '-')
        u_last_a = ultimo_autor_d.get(uid, '-')
        
        u_idx = u2i.get(uid, None)
        if u_idx is not None and u_idx < len(user_factors):
            u_vector = user_factors[u_idx]
        else:
            u_vector = np.zeros(user_factors.shape[1])

        for libro, info in cands.items():
            aut_l = libro_autor_d.get(libro, '-')
            n_aut_u, r_aut_u = user_autor_d.get((uid, aut_l), (0, 0.0))

            afinidad_co_autor = 0.0
            if aut_l in autor_vectors:
                afinidad_co_autor = float(np.dot(u_vector, autor_vectors[aut_l]))
            
            rows_out.append({
                'id_lector': uid,
                'id_libro': libro,
                'als_score': info['als_score'],
                'bpr_score': info['bpr_score'],
                'from_als': info['from_als'],
                'from_bpr': info['from_bpr'],
                'from_seq': info['from_seq'],
                'from_itemcf': info['from_itemcf'],
                'from_autor': info['from_autor'],
                'from_demo': info['from_demo'],
                'pop_total': libro_pop_d.get(libro, 0),
                'pop_reciente': libro_rec_d.get(libro, 0),
                'rating_bay': libro_bay_d.get(libro, 5.0),
                'score_ct': libro_ct_d.get(libro, 0.0),
                'pop_autor_total': libro_pop_autor_d.get(aut_l, 0),
                'match_genero': int(libro_genero_d.get(libro, '') == u_gen_f),
                'match_editorial': int(libro_editorial_d.get(libro, '') == u_edi_f),
                'match_ultimo_autor': int(aut_l == u_last_a),
                'user_books_autor': n_aut_u,
                'user_rating_autor': r_aut_u,
                'actividad_lector': act,
                'avg_rating_user': avg_r,
                'afinidad_mainstream': main_u,
                'dias_desde_ultima_lectura': dias_u,
                'afinidad_co_autor': afinidad_co_autor
            })
            
    return pd.DataFrame(rows_out), fallback_books

# Variables fijas
FEATURES = [
    'als_score', 'bpr_score', 'from_als', 'from_bpr', 'from_seq', 'from_itemcf',
    'from_autor', 'from_demo', 'pop_total', 'pop_reciente', 'rating_bay', 'score_ct',
    'pop_autor_total', 'match_genero', 'match_editorial', 'match_ultimo_autor',
    'user_books_autor', 'user_rating_autor', 'actividad_lector', 'avg_rating_user',
    'afinidad_mainstream', 'dias_desde_ultima_lectura', 'afinidad_co_autor'
]

# ============================================================
# FASE A — ENTRENAMIENTO LOCAL (DATOS PRE-2023)
# ============================================================
print("\n=== FASE A: Armando Dataset de Entrenamiento (Pre-2023) ===")
df_train, _ = pipeline_features_y_candidatos(df_pasado, target_users=usuarios_con_futuro)

def asignar_target(row):
    rating = futuro_rating_d.get((row['id_lector'], row['id_libro']), 0)
    if rating == 0: return 0          
    elif rating >= 8: return 2          
    else: return 1          

df_train['target'] = df_train.apply(asignar_target, axis=1)

print(f"   Matriz de entrenamiento generada: {df_train.shape}")
print(f"   Distribución de targets:\n{df_train['target'].value_counts().head(5)}")

np.random.seed(42)
usuarios_train_pool = df_train['id_lector'].unique()
mask_tr_u = np.random.rand(len(usuarios_train_pool)) < 0.9
set_tr_u = set(usuarios_train_pool[mask_tr_u])

df_tr_split = df_train[df_train['id_lector'].isin(set_tr_u)].copy()
df_va_split = df_train[~df_train['id_lector'].isin(set_tr_u)].copy()

df_tr_split = df_tr_split.sort_values('id_lector').reset_index(drop=True)
df_va_split = df_va_split.sort_values('id_lector').reset_index(drop=True)

pool_train = Pool(data=df_tr_split[FEATURES], label=df_tr_split['target'], group_id=df_tr_split['id_lector'])
pool_val   = Pool(data=df_va_split[FEATURES], label=df_va_split['target'], group_id=df_va_split['id_lector'])

print(f"   Ajustando CatBoostRanker en {len(df_tr_split)} filas (Val: {len(df_va_split)})...")
ranker = CatBoostRanker(**CB_PARAMS)
ranker.fit(pool_train, eval_set=pool_val, early_stopping_rounds=EARLY_STOPPING)

imp = pd.DataFrame({
    'feature': FEATURES, 
    'importance': ranker.get_feature_importance(data=pool_train)  
}).sort_values('importance', ascending=False)
print("\n   [CatBoost Feature Importance]:")
print(imp.head(10).to_string(index=False))

# ============================================================
# FASE B — INFERENCIA FINAL (DF COMPLETO)
# ============================================================
print("\n=== FASE B: Reentrenamiento Final e Inferencia (Todo el Historial) ===")
df_completo_train, fallback_final = pipeline_features_y_candidatos(df_interacciones, target_users=lectores_objetivo)

df_completo_train = df_completo_train.sort_values('id_lector').reset_index(drop=True)
pool_completo = Pool(
    data=df_completo_train[FEATURES], 
    group_id=df_completo_train['id_lector']
)

print("   -> Calculando scores de CatBoost en batch...")
df_completo_train['score_pred'] = ranker.predict(pool_completo)

print(f"   -> Generando el archivo de submission para {len(lectores_objetivo)} usuarios...")
recomendaciones = []
leidos_totales = df_interacciones.groupby('id_lector')['id_libro'].apply(set).to_dict()

for uid, grp in df_completo_train.groupby('id_lector', sort=False):
    grp_sorted = grp.sort_values('score_pred', ascending=False)
    libros_ordenados = grp_sorted['id_libro'].tolist()
    
    top_20 = libros_ordenados[:20]
    for lib in top_20:
        recomendaciones.append({'id_lector': uid, 'id_libro': lib})
        
    if len(top_20) < 20:
        ya_recomendados = set(top_20) | leidos_totales.get(uid, set())
        for lib_f in fallback_final:
            if lib_f not in ya_recomendados:
                recomendaciones.append({'id_lector': uid, 'id_libro': lib_f})
                ya_recomendados.add(lib_f)
            if len(ya_recomendados) - len(leidos_totales.get(uid, set())) >= 20:
                break

df_sub = pd.DataFrame(recomendaciones)
df_sub.to_csv(OUTPUT_CSV, index=False)
print(f"\n[PROCESO COMPLETADO] Archivo listo en: {OUTPUT_CSV}")
