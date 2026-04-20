import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import math

# ===================== CONFIGURACIÓN MODIFICABLE =====================
X_LIMIT = 470           # Variable para identificar cruce del canal
UMBRAL_DISTANCIA = 10   # Umbral de distancia para vecino más cercano
TOLERANCIA_ROI = 5      # Tolerancia para puntos cercanos a la ROI (en píxeles)
FPS_VIDEO = 40          # Frames por segundo para los videos de seguimiento
DIA_ANALIZAR = "2024-07-10"  # DÍA ESPECÍFICO para generar videos
# =====================================================================

# Configuración de ROI (Región de Interés) - cuadrilátero
ROI1_POINTS = [(439, 100), (558, 100), (516, 600), (180, 600)]

def distancia_punto_a_segmento_optimizada(p, a, b):
    """
    Calcula la distancia mínima de un punto p a un segmento de recta entre a y b
    Versión optimizada para mayor velocidad
    """
    # Convertir a arrays numpy para cálculos vectoriales
    p = np.array(p, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    
    # Vector del segmento
    ab = b - a
    # Vector desde a hasta p
    ap = p - a
    
    # Proyección escalar de ap sobre ab
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    # Clamp t al rango [0, 1] para quedarnos en el segmento
    t = np.clip(t, 0, 1)
    
    # Punto más cercano en el segmento
    punto_mas_cercano = a + t * ab
    
    # Distancia entre p y el punto más cercano
    distancia = np.linalg.norm(p - punto_mas_cercano)
    
    return distancia, punto_mas_cercano

def encontrar_todos_puntos_roi_cercanos(x_puntos, y_puntos, roi_points, tolerancia=TOLERANCIA_ROI):
    """
    Encuentra TODOS los puntos del perímetro que están cerca del perímetro de la ROI poligonal
    Versión optimizada y garantizada para encontrar TODOS los puntos cercanos
    """
    # Crear lista de segmentos de la ROI
    segmentos = []
    n_vertices = len(roi_points)
    for i in range(n_vertices):
        a = roi_points[i]
        b = roi_points[(i + 1) % n_vertices]
        segmentos.append((a, b))
    
    # Lista para almacenar TODOS los puntos cercanos
    indices_cercanos = []
    
    # Precalcular bounding box de la ROI con margen de tolerancia
    roi_x_min = min(p[0] for p in roi_points) - tolerancia
    roi_x_max = max(p[0] for p in roi_points) + tolerancia
    roi_y_min = min(p[1] for p in roi_points) - tolerancia
    roi_y_max = max(p[1] for p in roi_points) + tolerancia
    
    # Verificar cada punto del perímetro
    for i in range(len(x_puntos)):
        x, y = x_puntos[i], y_puntos[i]
        
        # Primera verificación rápida: ¿está dentro del bounding box extendido?
        if not (roi_x_min <= x <= roi_x_max and roi_y_min <= y <= roi_y_max):
            continue
        
        p = (x, y)
        dist_min = float('inf')
        
        # Calcular distancia a cada segmento
        for segmento in segmentos:
            a, b = segmento
            distancia, _ = distancia_punto_a_segmento_optimizada(p, a, b)
            if distancia < dist_min:
                dist_min = distancia
            # Si ya está dentro de la tolerancia, podemos salir del bucle
            if dist_min <= tolerancia:
                break
        
        # Si la distancia mínima está dentro de la tolerancia, agregar el punto
        if dist_min <= tolerancia:
            indices_cercanos.append(i)
    
    return indices_cercanos

def identificar_zonas_concentradas(x_puntos, y_puntos, indices_roi, eps=15):
    """
    Identifica zonas concentradas de puntos ROI cercanos usando DBSCAN
    Retorna una lista de zonas, cada zona contiene los índices de puntos en esa zona
    """
    if len(indices_roi) == 0:
        return []
    
    # Extraer coordenadas de los puntos ROI cercanos
    puntos_roi = np.column_stack([x_puntos[indices_roi], y_puntos[indices_roi]])
    
    # Aplicar DBSCAN para identificar clusters
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(puntos_roi)
    
    # Crear lista de zonas
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    zonas = []
    
    for i in range(n_clusters):
        # Índices de puntos en este cluster (en el array de índices ROI)
        cluster_indices = np.where(labels == i)[0]
        # Convertir a índices originales
        zona_indices = [indices_roi[idx] for idx in cluster_indices]
        zonas.append(zona_indices)
    
    return zonas

def seleccionar_punto_mas_cercano_a_roi(x_puntos, y_puntos, zona_indices, roi_points):
    """
    Selecciona el punto más cercano a la ROI dentro de una zona
    """
    if len(zona_indices) == 0:
        return None
    
    # Calcular distancia mínima de cada punto a la ROI
    distancias = []
    for idx in zona_indices:
        p = (x_puntos[idx], y_puntos[idx])
        dist_min = float('inf')
        
        # Calcular distancia a cada segmento de la ROI
        n_vertices = len(roi_points)
        for i in range(n_vertices):
            a = roi_points[i]
            b = roi_points[(i + 1) % n_vertices]
            distancia, _ = distancia_punto_a_segmento_optimizada(p, a, b)
            if distancia < dist_min:
                dist_min = distancia
        
        distancias.append(dist_min)
    
    # Seleccionar el punto con la distancia mínima
    idx_mas_cercano = zona_indices[np.argmin(distancias)]
    
    return idx_mas_cercano

def seguir_perimetro_desde_punto(x_puntos, y_puntos, punto_inicio_idx, umbral_distancia=UMBRAL_DISTANCIA, max_puntos=None, direccion='una'):
    """
    Sigue el perímetro desde un punto en una o ambas direcciones
    """
    n_puntos = len(x_puntos)
    visitados = np.zeros(n_puntos, dtype=bool)
    secuencia = []
    
    idx_actual = punto_inicio_idx
    contador_puntos = 0
    
    # Seguimiento en dirección principal
    while True:
        if max_puntos is not None and contador_puntos >= max_puntos:
            break
        
        visitados[idx_actual] = True
        secuencia.append(idx_actual)
        contador_puntos += 1
        
        # Encontrar vecinos no visitados
        distancias = np.sqrt((x_puntos - x_puntos[idx_actual])**2 + 
                           (y_puntos - y_puntos[idx_actual])**2)
        distancias[visitados] = np.inf
        
        if np.all(distancias == np.inf):
            break
        
        # Encontrar el vecino más cercano
        idx_siguiente = np.argmin(distancias)
        
        # Verificar si la distancia al vecino más cercano supera el umbral
        if distancias[idx_siguiente] > umbral_distancia:
            break
        
        idx_actual = idx_siguiente
    
    # Si se requiere seguimiento en ambas direcciones
    if direccion == 'ambas' and len(secuencia) > 1:
        # Para seguir en dirección contraria, empezamos desde el punto inicial nuevamente
        # pero evitando la dirección ya tomada
        idx_actual = punto_inicio_idx
        visitados_contrario = visitados.copy()
        
        # Marcar toda la secuencia anterior como visitada para no volver
        for idx in secuencia:
            visitados_contrario[idx] = True
        
        # Encontrar otro vecino (diferente al primero)
        distancias = np.sqrt((x_puntos - x_puntos[punto_inicio_idx])**2 + 
                           (y_puntos - y_puntos[punto_inicio_idx])**2)
        distancias[visitados_contrario] = np.inf
        
        if not np.all(distancias == np.inf):
            idx_otro_vecino = np.argmin(distancias)
            
            if distancias[idx_otro_vecino] <= umbral_distancia:
                # Seguimiento en dirección contraria
                secuencia_contraria = [punto_inicio_idx]
                idx_actual = idx_otro_vecino
                contador_contrario = 0
                
                while True:
                    if max_puntos is not None and contador_contrario >= max_puntos:
                        break
                    
                    visitados_contrario[idx_actual] = True
                    secuencia_contraria.append(idx_actual)
                    contador_contrario += 1
                    
                    # Encontrar vecinos no visitados
                    distancias = np.sqrt((x_puntos - x_puntos[idx_actual])**2 + 
                                       (y_puntos - y_puntos[idx_actual])**2)
                    distancias[visitados_contrario] = np.inf
                    
                    if np.all(distancias == np.inf):
                        break
                    
                    idx_siguiente = np.argmin(distancias)
                    
                    if distancias[idx_siguiente] > umbral_distancia:
                        break
                    
                    idx_actual = idx_siguiente
                
                # Combinar ambas secuencias (invertir la contraria y agregar)
                secuencia_contraria_invertida = list(reversed(secuencia_contraria))
                secuencia_completa = secuencia_contraria_invertida[:-1] + secuencia  # Evitar duplicar punto_inicio
                return secuencia_completa
    
    return secuencia

def verificar_cruce_x470(secuencia, x_puntos, x_limit=X_LIMIT):
    """
    Verifica si la secuencia cruza la línea X=470
    """
    if len(secuencia) < 2:
        return False
    
    # Verificar cambios de lado de X=470
    lados = []
    for idx in secuencia:
        x = x_puntos[idx]
        if x < x_limit:
            lados.append(-1)  # Izquierda
        else:
            lados.append(1)   # Derecha
    
    # Buscar cambios de signo
    for i in range(1, len(lados)):
        if lados[i] != lados[i-1]:
            return True
    
    return False

def encontrar_curvas_perimetrales(x_puntos, y_puntos, umbral_distancia=UMBRAL_DISTANCIA):
    """
    Identifica todas las curvas perimetrales en los puntos dados
    """
    n_puntos = len(x_puntos)
    visitados = np.zeros(n_puntos, dtype=bool)
    curvas = []
    
    for i in range(n_puntos):
        if not visitados[i]:
            # Seguir una curva desde este punto en ambas direcciones
            secuencia = seguir_perimetro_desde_punto(x_puntos, y_puntos, i, umbral_distancia, direccion='ambas')
            
            # Marcar como visitados
            for idx in secuencia:
                visitados[idx] = True
            
            if len(secuencia) > 1:
                curvas.append(secuencia)
    
    return curvas

def identificar_extremos_curva_con_roi(curva, x_puntos, y_puntos, indices_roi):
    """
    Identifica los extremos de una curva que están cerca de la ROI
    Retorna listas de índices para cada extremo (puede haber múltiples puntos por extremo)
    """
    extremos = []
    
    # Considerar puntos cerca del inicio y final de la curva
    # Usar una ventana de los primeros y últimos 10 puntos
    ventana = min(10, len(curva) // 4)
    
    inicio_curva = curva[:ventana]
    final_curva = curva[-ventana:]
    
    # Encontrar puntos ROI en cada extremo
    puntos_inicio = [idx for idx in inicio_curva if idx in indices_roi]
    puntos_final = [idx for idx in final_curva if idx in indices_roi]
    
    if puntos_inicio:
        extremos.append(puntos_inicio)
    if puntos_final:
        extremos.append(puntos_final)
    
    return extremos

def encontrar_seguimientos_desde_extremos(x_puntos, y_puntos, indices_roi, x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA):
    """
    Realiza seguimientos desde los extremos de las curvas
    (Mismo método que en el análisis principal)
    """
    n_puntos = len(x_puntos)
    
    # 1. Identificar todas las curvas perimetrales
    curvas = encontrar_curvas_perimetrales(x_puntos, y_puntos, umbral_distancia)
    
    # 2. Identificar zonas concentradas de puntos ROI
    zonas = identificar_zonas_concentradas(x_puntos, y_puntos, indices_roi, eps=15)
    
    # Diccionario para almacenar seguimientos por curva
    seguimientos_por_curva = {}
    
    # 3. Para cada curva, identificar extremos con ROI y realizar seguimientos
    for curva_idx, curva in enumerate(curvas):
        extremos_curva = identificar_extremos_curva_con_roi(curva, x_puntos, y_puntos, indices_roi)
        
        # Necesitamos al menos 2 extremos con ROI para esta curva
        if len(extremos_curva) >= 2:
            # Para cada extremo, encontrar la zona correspondiente
            extremos_con_zona = []
            
            for puntos_extremo in extremos_curva:
                # Encontrar a qué zona pertenece este extremo
                zona_encontrada = None
                for zona_idx, zona in enumerate(zonas):
                    if any(p in zona for p in puntos_extremo):
                        zona_encontrada = zona_idx
                        break
                
                if zona_encontrada is not None:
                    # Seleccionar el punto más cercano a la ROI dentro de este extremo
                    punto_mas_cercano = seleccionar_punto_mas_cercano_a_roi(
                        x_puntos, y_puntos, puntos_extremo, ROI1_POINTS
                    )
                    
                    if punto_mas_cercano is not None:
                        extremos_con_zona.append({
                            'zona': zona_encontrada,
                            'punto_inicio': punto_mas_cercano,
                            'puntos_extremo': puntos_extremo
                        })
            
            # Si tenemos al menos 2 extremos con zonas distintas
            if len(extremos_con_zona) >= 2:
                # Realizar seguimientos desde cada extremo
                seguimientos_extremos = []
                
                for extremo_info in extremos_con_zona:
                    punto_inicio = extremo_info['punto_inicio']
                    
                    # Realizar seguimiento desde este extremo
                    secuencia = seguir_perimetro_desde_punto(
                        x_puntos, y_puntos, punto_inicio, umbral_distancia, 
                        max_puntos=1000, direccion='una'
                    )
                    
                    # Verificar si el seguimiento cruza X=470
                    cruza_x470 = verificar_cruce_x470(secuencia, x_puntos, x_limit)
                    
                    # Verificar si termina en otra zona ROI
                    idx_final = secuencia[-1] if secuencia else punto_inicio
                    termina_en_zona_roi = False
                    zona_termina = None
                    
                    for zona_idx, zona in enumerate(zonas):
                        if idx_final in zona and idx_final != punto_inicio:
                            termina_en_zona_roi = True
                            zona_termina = zona_idx
                            break
                    
                    seguimientos_extremos.append({
                        'indice_inicio': punto_inicio,
                        'indice_final': idx_final,
                        'secuencia': secuencia,
                        'termina_en_zona_roi': termina_en_zona_roi,
                        'zona_termina': zona_termina,
                        'cruza_x470': cruza_x470,
                        'longitud': len(secuencia),
                        'zona_inicio': extremo_info['zona']
                    })
                
                # Almacenar seguimientos para esta curva
                if len(seguimientos_extremos) >= 2:
                    seguimientos_por_curva[curva_idx] = {
                        'curva': curva,
                        'seguimientos': seguimientos_extremos
                    }
    
    return seguimientos_por_curva, zonas, curvas

def identificar_encuentros_seguimientos(seguimientos_por_curva):
    """
    Identifica encuentros entre seguimientos (si se "topan" o comparten puntos)
    """
    encuentros = []
    
    for curva_idx, info_curva in seguimientos_por_curva.items():
        seguimientos = info_curva['seguimientos']
        
        # Comparar cada par de seguimientos de la misma curva
        for i in range(len(seguimientos)):
            for j in range(i+1, len(seguimientos)):
                seg_i = seguimientos[i]
                seg_j = seguimientos[j]
                
                # Verificar si son de zonas distintas (extremos opuestos)
                if seg_i['zona_inicio'] == seg_j['zona_inicio']:
                    continue
                
                # Verificar si comparten puntos en común (excluyendo extremos)
                puntos_i = set(seg_i['secuencia'])
                puntos_j = set(seg_j['secuencia'])
                
                # Excluir los puntos iniciales (están en zonas ROI)
                puntos_i_sin_extremo = puntos_i - {seg_i['indice_inicio']}
                puntos_j_sin_extremo = puntos_j - {seg_j['indice_inicio']}
                
                puntos_comunes = puntos_i_sin_extremo.intersection(puntos_j_sin_extremo)
                
                if len(puntos_comunes) > 5:  # Umbral para considerar encuentro
                    encuentros.append({
                        'curva_idx': curva_idx,
                        'seguimiento_i': i,
                        'seguimiento_j': j,
                        'puntos_comunes': list(puntos_comunes),
                        'cantidad_comunes': len(puntos_comunes),
                        'cumple_criterio_1': True  # Criterio 1: seguimientos se encuentran
                    })
    
    return encuentros

def clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_puntos, x_limit=X_LIMIT):
    """
    Clasifica las curvas según los 3 criterios
    (Mismo método que en el análisis principal)
    """
    curvas_clasificadas = []
    
    for encuentro in encuentros:
        curva_idx = encuentro['curva_idx']
        info_curva = seguimientos_por_curva[curva_idx]
        seg_i = info_curva['seguimientos'][encuentro['seguimiento_i']]
        seg_j = info_curva['seguimientos'][encuentro['seguimiento_j']]
        
        # Criterio 1: Seguimientos se encuentran (ya verificado)
        criterio1 = encuentro['cumple_criterio_1']
        
        # Criterio 2: Eliminar si es ROI a ROI (ambos extremos en la misma zona)
        # En realidad, ya filtramos que sean de zonas distintas, pero verificamos
        criterio2 = seg_i['zona_inicio'] != seg_j['zona_inicio']
        
        # Criterio 3: Ambos seguimientos cruzan X=470
        criterio3 = seg_i['cruza_x470'] and seg_j['cruza_x470']
        
        # Determinar tipo de curva
        if criterio1 and criterio2 and criterio3:
            tipo = "limite_real_canal"
        elif criterio1 and not criterio2:
            tipo = "otro_limite"
        else:
            tipo = "no_cumple"
        
        # Crear curva combinada (unión de ambos seguimientos)
        curva_combinada = list(set(seg_i['secuencia'] + seg_j['secuencia']))
        curva_combinada.sort(key=lambda idx: seg_i['secuencia'].index(idx) if idx in seg_i['secuencia'] else 
                            len(seg_i['secuencia']) + seg_j['secuencia'].index(idx) if idx in seg_j['secuencia'] else 0)
        
        curvas_clasificadas.append({
            'curva_idx': curva_idx,
            'tipo': tipo,
            'cumple_3_criterios': (criterio1 and criterio2 and criterio3),
            'longitud': len(curva_combinada),
            'curva_puntos': curva_combinada,
            'seguimiento_i': seg_i,
            'seguimiento_j': seg_j,
            'criterios': {
                'encuentro': criterio1,
                'distintas_zonas': criterio2,
                'cruza_x470': criterio3
            }
        })
    
    return curvas_clasificadas

def obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_deseado_str):
    """
    Obtiene el estado de apertura directamente del archivo NetCDF generado
    por el código de análisis
    """
    ds = xr.open_dataset(archivo_aperturas)
    
    # Encontrar tiempo más cercano
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    
    # Obtener estado del archivo
    estado = int(ds.estado_apertura.values[idx_tiempo])
    
    # Obtener límites reales
    if 'cantidad_limites_reales' in ds:
        limites_reales = int(ds.cantidad_limites_reales.values[idx_tiempo])
    else:
        limites_reales = 2 if estado == 1 else 0
    
    ds.close()
    
    return estado, limites_reales

def generar_video_seguimiento_desde_extremos(archivo_aperturas, tiempo_deseado_str, archivo_video, 
                                           x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA):
    """
    Genera video del seguimiento usando EXACTAMENTE el mismo método que el código de análisis
    (Seguimiento desde extremos con 3 criterios)
    """
    print(f"\nGenerando video de seguimiento DESDE EXTREMOS para {tiempo_deseado_str}")
    print(f"Usando EXACTAMENTE el mismo método que el análisis principal")
    print(f"Usando X_LIMIT = {x_limit}")
    print(f"Usando UMBRAL_DISTANCIA = {umbral_distancia}")
    print(f"Usando TOLERANCIA_ROI = {TOLERANCIA_ROI} px")
    print(f"FPS Video: {FPS_VIDEO}")
    print(f"ROI: {ROI1_POINTS}")
    
    # Primero obtener el estado del archivo NetCDF
    estado_archivo, limites_reales_archivo = obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_deseado_str)
    
    print(f"Estado del archivo NetCDF: {'ABIERTO' if estado_archivo == 1 else 'CERRADO'}")
    print(f"Límites reales en archivo: {limites_reales_archivo}")
    
    # Ahora procesar para generar el video
    ds = xr.open_dataset(archivo_aperturas)
    
    # Encontrar tiempo más cercano
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    tiempo_real = ds.time.values[idx_tiempo]
    tiempo_real_str = pd.to_datetime(tiempo_real).strftime('%Y-%m-%d %H:%M')
    
    print(f"Tiempo encontrado: {tiempo_real_str} (índice: {idx_tiempo})")
    
    # Verificar si existe máscara de agua
    if ds.existe_mascara_agua.values[idx_tiempo] == 0:
        print("ERROR: No existe máscara de agua para este tiempo")
        ds.close()
        return
    
    # Obtener puntos del perímetro
    perimetro = ds.perimetro_arena_agua.values[idx_tiempo, :]
    indices_activos = np.where(perimetro == 1)[0]
    
    if len(indices_activos) == 0:
        print("ERROR: No hay puntos de perímetro activos")
        ds.close()
        return
    
    # Coordenadas originales
    x_activos = ds.coordenada_x.values[indices_activos]
    y_activos = ds.coordenada_y.values[indices_activos]
    
    # 1. Encontrar TODOS los puntos cercanos a la ROI (igual que en el análisis)
    indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, ROI1_POINTS, TOLERANCIA_ROI)
    
    print(f"Puntos ROI cercanos encontrados: {len(indices_roi)}")
    
    # 2. Encontrar seguimientos desde extremos (mismo método que análisis)
    seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(
        x_activos, y_activos, indices_roi, x_limit, umbral_distancia
    )
    
    print(f"Curvas perimetrales identificadas: {len(curvas)}")
    print(f"Zonas ROI identificadas: {len(zonas)}")
    print(f"Curvas con seguimientos desde extremos: {len(seguimientos_por_curva)}")
    
    # 3. Identificar encuentros entre seguimientos
    encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
    
    print(f"Encuentros encontrados entre seguimientos: {len(encuentros)}")
    
    # 4. Clasificar curvas según criterios
    curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
    
    print(f"Curvas clasificadas: {len(curvas_clasificadas)}")
    
    # Contar curvas por tipo
    curvas_reales = [c for c in curvas_clasificadas if c['tipo'] == 'limite_real_canal']
    otros_limites = [c for c in curvas_clasificadas if c['tipo'] == 'otro_limite']
    
    print(f"Límites reales del canal (3 criterios): {len(curvas_reales)}")
    print(f"Otros límites (solo encuentro): {len(otros_limites)}")
    
    # 5. Determinar estado y seleccionar límites para mostrar
    estado_final = estado_archivo  # Usar el estado del archivo por consistencia
    
    # Seleccionar límites para mostrar según el estado
    if estado_final == 1 and len(curvas_reales) >= 2:
        # Estado ABIERTO: mostrar las 2 curvas reales más largas
        curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
        limite1_info = curvas_reales[0] if len(curvas_reales) > 0 else None
        limite2_info = curvas_reales[1] if len(curvas_reales) > 1 else None
        print(f"Estado: ABIERTO - Mostrando 2 límites reales")
    else:
        # Estado CERRADO: mostrar solo seguimientos (sin límites reales)
        limite1_info = None
        limite2_info = None
        print(f"Estado: CERRADO - No se muestran límites reales")
    
    # Configurar animación
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Configurar gráfico
    ax.set_xlim(np.min(x_activos) - 20, np.max(x_activos) + 20)
    ax.set_ylim(np.max(y_activos) + 20, np.min(y_activos) - 20)
    ax.set_xlabel('Coordenada X', fontsize=12)
    ax.set_ylabel('Coordenada Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f'Estado de Conexión - Seguimiento desde Extremos\n{tiempo_real_str}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Dibujar ROI
    roi_poly = Polygon(ROI1_POINTS, closed=True, linewidth=3, 
                      edgecolor='orange', facecolor='none', 
                      linestyle='--', alpha=0.7, label='ROI')
    ax.add_patch(roi_poly)
    
    # Puntos del perímetro en AZUL
    ax.scatter(x_activos, y_activos, c='blue', s=10, alpha=0.3, label='Perímetro')
    
    # Dibujar zonas ROI con diferentes colores (SIN LEYENDA)
    colores_zonas = plt.cm.tab20(np.linspace(0, 1, len(zonas)))
    for i, zona in enumerate(zonas):
        if len(zona) > 0:
            x_zona = x_activos[zona]
            y_zona = y_activos[zona]
            ax.scatter(x_zona, y_zona, c=[colores_zonas[i]], s=80, marker='s', 
                      alpha=0.8, edgecolors='black', linewidths=1.5, 
                      zorder=10)  # sin label
    
    # Línea de referencia X=x_limit
    ax.axvline(x=x_limit, color='red', linestyle='--', alpha=0.7, 
               linewidth=2, label=f'X={x_limit}')
    
    # Preparar animación de seguimientos desde extremos
    seguimientos_anim = []
    
    # Colores para seguimientos (un color por curva)
    colores_curvas = plt.cm.Set3(np.linspace(0, 1, len(seguimientos_por_curva)))
    
    for curva_idx, (curva_num, info_curva) in enumerate(list(seguimientos_por_curva.items())[:8]):  # Máximo 8 curvas
        seguimientos = info_curva['seguimientos']
        
        # Para cada seguimiento de esta curva
        for seg_idx, seg in enumerate(seguimientos[:2]):  # Máximo 2 seguimientos por curva
            if len(seg['secuencia']) > 0:
                # Punto inicial
                x_inicio = x_activos[seg['indice_inicio']]
                y_inicio = y_activos[seg['indice_inicio']]
                
                # Crear círculo para seguimiento
                import matplotlib.patches as patches
                circulo = patches.Circle((x_inicio, y_inicio), radius=8, 
                                color=colores_curvas[curva_idx], 
                                alpha=0.9, fill=True, zorder=20, linewidth=2, 
                                edgecolor='black')
                ax.add_patch(circulo)
                
                seguimientos_anim.append({
                    'circulo': circulo,
                    'secuencia': seg['secuencia'],
                    'x_vals': [],
                    'y_vals': [],
                    'linea': ax.plot([], [], '-', 
                                   color=colores_curvas[curva_idx], 
                                   alpha=0.7, linewidth=2.5)[0],  # sin label
                    'cruza_x470': seg['cruza_x470'],
                    'curva_idx': curva_num,
                    'seg_idx': seg_idx,
                    'indice_actual': 0,
                    'zona_inicio': seg['zona_inicio']
                })
    
    # Variables para límites reales (si existen)
    limite_sup_linea, = ax.plot([], [], 'lime', linewidth=6, alpha=0.9, 
                                label='Límite Real Superior')
    limite_inf_linea, = ax.plot([], [], 'cyan', linewidth=6, alpha=0.9, 
                                label='Límite Real Inferior')
    
    # Variables para puntos de encuentro
    encuentro_points = ax.plot([], [], 'ro', markersize=10, alpha=0.9, 
                              label='Puntos Encuentro')[0]
    
    # Variables para otros límites (solo encuentro) - SIN LEYENDA
    otros_limites_lineas = []
    for i in range(min(2, len(otros_limites))):
        linea, = ax.plot([], [], 'yellow', linewidth=4, alpha=0.7)[0]  # sin label
        otros_limites_lineas.append(linea)
    
    def init():
        limite_sup_linea.set_data([], [])
        limite_inf_linea.set_data([], [])
        encuentro_points.set_data([], [])
        for linea in otros_limites_lineas:
            linea.set_data([], [])
        for seg_anim in seguimientos_anim:
            seg_anim['x_vals'] = []
            seg_anim['y_vals'] = []
            seg_anim['linea'].set_data([], [])
            seg_anim['indice_actual'] = 0
        elementos = [limite_sup_linea, limite_inf_linea, encuentro_points] + otros_limites_lineas
        elementos.extend([s['circulo'] for s in seguimientos_anim])
        elementos.extend([s['linea'] for s in seguimientos_anim])
        return elementos
    
    def animate(frame):
        nonlocal seguimientos_anim
        
        # Encontrar la secuencia más larga para normalizar el progreso
        if seguimientos_anim:
            longitudes = [len(s['secuencia']) for s in seguimientos_anim]
            max_longitud = max(longitudes) if longitudes else 1
        else:
            max_longitud = 1
        
        # Calcular progreso normalizado (0 a 1)
        progreso = min(frame / (max_longitud * 1.3), 1.0)  # Agregamos 30% extra para mostrar límites
        
        # Actualizar seguimientos
        for seg_anim in seguimientos_anim:
            if len(seg_anim['secuencia']) > 0:
                # Calcular cuántos puntos mostrar basado en el progreso
                puntos_a_mostrar = int(progreso * len(seg_anim['secuencia']))
                puntos_a_mostrar = max(1, puntos_a_mostrar)  # Al menos 1 punto
                
                # Si hay puntos para mostrar
                if puntos_a_mostrar > 0:
                    # Obtener el último punto mostrado
                    idx = puntos_a_mostrar - 1
                    x = x_activos[seg_anim['secuencia'][idx]]
                    y = y_activos[seg_anim['secuencia'][idx]]
                    seg_anim['circulo'].center = (x, y)
                    
                    # Agregar puntos a la línea
                    seg_anim['x_vals'] = x_activos[seg_anim['secuencia'][:puntos_a_mostrar]].tolist()
                    seg_anim['y_vals'] = y_activos[seg_anim['secuencia'][:puntos_a_mostrar]].tolist()
                    seg_anim['linea'].set_data(seg_anim['x_vals'], seg_anim['y_vals'])
                    
                    seg_anim['indice_actual'] = idx
        
        # Actualizar límites y encuentros (aparecen después de cierto progreso)
        if progreso >= 0.6:  # Mostrar después del 60% de la animación
            # Límites reales (si existen)
            if limite1_info is not None:
                x_lim1 = x_activos[limite1_info['curva_puntos']]
                y_lim1 = y_activos[limite1_info['curva_puntos']]
                limite_sup_linea.set_data(x_lim1, y_lim1)
            
            if limite2_info is not None:
                x_lim2 = x_activos[limite2_info['curva_puntos']]
                y_lim2 = y_activos[limite2_info['curva_puntos']]
                limite_inf_linea.set_data(x_lim2, y_lim2)
            
            # Otros límites (solo encuentro)
            for i, linea in enumerate(otros_limites_lineas):
                if i < len(otros_limites):
                    curva_info = otros_limites[i]
                    x_otro = x_activos[curva_info['curva_puntos']]
                    y_otro = y_activos[curva_info['curva_puntos']]
                    linea.set_data(x_otro, y_otro)
            
            # Mostrar puntos de encuentro
            if len(encuentros) > 0:
                # Combinar todos los puntos de encuentro
                todos_puntos_encuentro = []
                for encuentro in encuentros:
                    todos_puntos_encuentro.extend(encuentro['puntos_comunes'])
                
                if len(todos_puntos_encuentro) > 0:
                    # Tomar solo algunos puntos para no saturar
                    puntos_a_mostrar = min(20, len(todos_puntos_encuentro))
                    indices_mostrar = np.linspace(0, len(todos_puntos_encuentro)-1, puntos_a_mostrar, dtype=int)
                    puntos_seleccionados = [todos_puntos_encuentro[i] for i in indices_mostrar]
                    
                    x_encuentro = x_activos[puntos_seleccionados]
                    y_encuentro = y_activos[puntos_seleccionados]
                    encuentro_points.set_data(x_encuentro, y_encuentro)
        
        elementos = [limite_sup_linea, limite_inf_linea, encuentro_points] + otros_limites_lineas
        elementos.extend([s['circulo'] for s in seguimientos_anim])
        elementos.extend([s['linea'] for s in seguimientos_anim])
        
        return elementos
    
    # Calcular frames totales
    if seguimientos_anim:
        longitudes = [len(s['secuencia']) for s in seguimientos_anim]
        max_longitud = max(longitudes) if longitudes else 1
        max_frames = int(max_longitud * 1.3) + 40  # 30% extra + 40 frames adicionales
    else:
        max_frames = 100
    
    print(f"Duración de animación: {max_frames} frames ({max_frames/FPS_VIDEO:.1f} segundos)")
    
    # Texto de estado e información 
    estado_color = 'lightgreen' if estado_final == 1 else 'lightcoral'
    
    info_detallada = (
        f'ESTADO: {"ABIERTO" if estado_final == 1 else "CERRADO"}\n'
        f'Curvas perimetrales: {len(curvas)}\n'
        f'Zonas ROI: {len(zonas)}\n'
        f'Seguimientos activos: {len(seguimientos_anim)}\n'
        f'Límites (3 criterios): {len(curvas_reales)}\n'
        f'Otros límites: {len(otros_limites)}\n'
        f'Encuentros: {len(encuentros)}'
    )
    
    estado_texto = ax.text(0.02, 0.98, 
                          info_detallada,
                          transform=ax.transAxes,
                          verticalalignment='top',
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor=estado_color, alpha=0.9))
    
    # Leyenda mejorada - SOLO los elementos solicitados
    legend_elements = [
        Line2D([0], [0], color='orange', lw=3, linestyle='--', label='ROI'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'X={x_limit}'),
        Line2D([0], [0], color='lime', lw=6, label='Límite Real Superior'),
        Line2D([0], [0], color='cyan', lw=6, label='Límite Real Inferior'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', 
               markersize=6, alpha=0.3, label='Perímetro'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', 
               markersize=8, label='Puntos Encuentro')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
              framealpha=0.9, ncol=2)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Crear animación
    print("Generando animación desde extremos...")
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=max_frames, interval=1000/FPS_VIDEO, blit=True)
    
    # Guardar video
    print(f"Guardando video con FPS={FPS_VIDEO}: {archivo_video}")
    estado_str = "ABIERTO" if estado_final == 1 else "CERRADO"
    ani.save(archivo_video, writer='ffmpeg', fps=FPS_VIDEO, dpi=150,
            metadata={'title': f'Seguimiento desde Extremos {tiempo_real_str}',
                     'comment': f'Estado: {estado_str}, Método IDÉNTICO al análisis'})
    
    plt.close(fig)
    ds.close()
    
    print("✓ Video generado exitosamente")
    print(f"✓ Estado final: {'ABIERTO' if estado_final == 1 else 'CERRADO'}")
    print(f"✓ Curvas reales identificadas: {len(curvas_reales)}")
    print(f"✓ Consistencia con archivo: {'SÍ' if len(curvas_reales) >= 2 == (estado_final == 1) else 'VERIFICAR'}")
    
    return estado_final, len(curvas_reales)


def generar_imagen_subplots_dia(archivo_aperturas, dia_analizar, carpeta_salida, 
                                x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA,
                                tolerancia_roi=TOLERANCIA_ROI, roi_points=ROI1_POINTS):
    """
    Genera una imagen con subplots en disposición 4 columnas x 3 filas (máx 12 instantes).
    Todos los subplots comparten la misma escala y relación de aspecto 1:1.
    """
    print(f"\n{'─'*50}")
    print("GENERANDO IMAGEN COMPUESTA DE SUBPLOTS (4x3)")
    print(f"{'─'*50}")
    
    # Abrir dataset y filtrar instantes del día
    ds = xr.open_dataset(archivo_aperturas)
    fecha_analizar = np.datetime64(dia_analizar)
    mascara_dia = pd.to_datetime(ds.time.values).date == pd.to_datetime(fecha_analizar).date()
    indices_dia = np.where(mascara_dia)[0]
    
    if len(indices_dia) == 0:
        print(f"No hay datos para el día {dia_analizar}")
        ds.close()
        return
    
    n_instantes = len(indices_dia)
    print(f"Generando subplots para {n_instantes} instantes en un grid 4x3...")
    
    # --- Calcular límites globales para todos los instantes (escala común) ---
    x_min_global, x_max_global = np.inf, -np.inf
    y_min_global, y_max_global = np.inf, -np.inf
    
    for idx in indices_dia:
        if ds.existe_mascara_agua.values[idx] == 0:
            continue
        perimetro = ds.perimetro_arena_agua.values[idx, :]
        idx_act = np.where(perimetro == 1)[0]
        if len(idx_act) == 0:
            continue
        x_tmp = ds.coordenada_x.values[idx_act]
        y_tmp = ds.coordenada_y.values[idx_act]
        x_min_global = min(x_min_global, np.min(x_tmp))
        x_max_global = max(x_max_global, np.max(x_tmp))
        y_min_global = min(y_min_global, np.min(y_tmp))
        y_max_global = max(y_max_global, np.max(y_tmp))
    
    # Si no hay datos, salir
    if x_min_global == np.inf:
        print("No se encontraron datos válidos para calcular límites.")
        ds.close()
        return
    
    # Añadir márgenes
    margen = 20
    x_min_global -= margen
    x_max_global += margen
    y_min_global -= margen
    y_max_global += margen
    
    # Forzar misma escala en X e Y (relación 1:1)
    dx = x_max_global - x_min_global
    dy = y_max_global - y_min_global
    centro_x = (x_min_global + x_max_global) / 2
    centro_y = (y_min_global + y_max_global) / 2
    lado_max = max(dx, dy) / 2
    
    x_lims = (centro_x - lado_max, centro_x + lado_max)
    y_lims = (centro_y + lado_max, centro_y - lado_max)  # invertido por coordenadas imagen
    
    # Configuración de colores globales
    colores_zonas_global = plt.cm.tab20(np.linspace(0, 1, 10))
    colores_curvas_global = plt.cm.Set3(np.linspace(0, 1, 8))
    
    # Crear figura con 4 columnas y 3 filas (máx 12 subplots)
    n_filas = 3
    n_columnas = 4
    fig, axes = plt.subplots(n_filas, n_columnas, figsize=(20, 15))
    axes = axes.flatten()
    
    # Ocultar todos los ejes inicialmente, luego activaremos los necesarios
    for ax in axes:
        ax.axis('off')
    
    # Para cada instante, generar el subplot
    for i, idx_tiempo in enumerate(indices_dia):
        if i >= len(axes):
            print(f"Advertencia: solo se mostrarán los primeros {len(axes)} instantes.")
            break
        
        ax = axes[i]
        ax.axis('on')  # activar el eje
        
        tiempo = ds.time.values[idx_tiempo]
        tiempo_str = pd.to_datetime(tiempo).strftime('%Y-%m-%d %H:%M')
        
        # --- INICIO: Copia del análisis (sin animación) ---
        if ds.existe_mascara_agua.values[idx_tiempo] == 0:
            ax.text(0.5, 0.5, 'Sin máscara', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_aspect('equal')
            continue
        
        perimetro = ds.perimetro_arena_agua.values[idx_tiempo, :]
        indices_activos = np.where(perimetro == 1)[0]
        if len(indices_activos) == 0:
            ax.text(0.5, 0.5, 'Sin perímetro', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_aspect('equal')
            continue
        
        x_activos = ds.coordenada_x.values[indices_activos]
        y_activos = ds.coordenada_y.values[indices_activos]
        
        # 1. Puntos ROI
        indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, roi_points, tolerancia_roi)
        
        # 2. Seguimientos desde extremos
        seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(
            x_activos, y_activos, indices_roi, x_limit, umbral_distancia
        )
        
        # 3. Encuentros
        encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
        
        # 4. Clasificación
        curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
        curvas_reales = [c for c in curvas_clasificadas if c['tipo'] == 'limite_real_canal']
        otros_limites = [c for c in curvas_clasificadas if c['tipo'] == 'otro_limite']
        
        # Estado (del archivo)
        estado_archivo, _ = obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_str)
        estado_final = estado_archivo
        
        # Seleccionar límites reales si corresponde
        if estado_final == 1 and len(curvas_reales) >= 2:
            curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
            limite1_info = curvas_reales[0]
            limite2_info = curvas_reales[1]
        else:
            limite1_info = None
            limite2_info = None
        # --- FIN del análisis ---
        
        # ---- PLOTEO ESTÁTICO con límites globales y aspecto igual ----
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # ROI
        roi_poly = Polygon(roi_points, closed=True, linewidth=2, 
                          edgecolor='orange', facecolor='none', 
                          linestyle='--', alpha=0.7)
        ax.add_patch(roi_poly)
        
        # Perímetro
        ax.scatter(x_activos, y_activos, c='blue', s=4, alpha=0.3)
        
        # Zonas ROI (colores, sin leyenda)
        for j, zona in enumerate(zonas[:10]):
            if len(zona) > 0:
                x_zona = x_activos[zona]
                y_zona = y_activos[zona]
                color = colores_zonas_global[j % len(colores_zonas_global)]
                ax.scatter(x_zona, y_zona, c=[color], s=40, marker='s',
                          alpha=0.8, edgecolors='black', linewidths=1, zorder=10)
        
        # Línea X=470
        ax.axvline(x=x_limit, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Seguimientos (líneas completas) - máximo 8 curvas, 2 seguimientos por curva
        for k, (curva_num, info_curva) in enumerate(list(seguimientos_por_curva.items())[:8]):
            seguimientos = info_curva['seguimientos']
            color_curva = colores_curvas_global[k % len(colores_curvas_global)]
            for seg_idx, seg in enumerate(seguimientos[:2]):
                if len(seg['secuencia']) > 0:
                    x_seq = x_activos[seg['secuencia']]
                    y_seq = y_activos[seg['secuencia']]
                    ax.plot(x_seq, y_seq, '-', color=color_curva, alpha=0.7, linewidth=1.5)
                    # Círculo inicial
                    x_ini = x_activos[seg['indice_inicio']]
                    y_ini = y_activos[seg['indice_inicio']]
                    ax.scatter(x_ini, y_ini, s=40, color=color_curva, edgecolors='black', 
                              linewidth=1.5, alpha=0.9, zorder=20)
        
        # Límites reales
        if limite1_info is not None:
            x_lim1 = x_activos[limite1_info['curva_puntos']]
            y_lim1 = y_activos[limite1_info['curva_puntos']]
            ax.plot(x_lim1, y_lim1, 'lime', linewidth=4, alpha=0.9)
        if limite2_info is not None:
            x_lim2 = x_activos[limite2_info['curva_puntos']]
            y_lim2 = y_activos[limite2_info['curva_puntos']]
            ax.plot(x_lim2, y_lim2, 'cyan', linewidth=4, alpha=0.9)
        
        # Otros límites (amarillo)
        for idx_otro, otro in enumerate(otros_limites[:2]):
            x_otro = x_activos[otro['curva_puntos']]
            y_otro = y_activos[otro['curva_puntos']]
            ax.plot(x_otro, y_otro, 'yellow', linewidth=3, alpha=0.7)
        
        # Puntos de encuentro (solo algunos)
        if len(encuentros) > 0:
            todos_puntos = []
            for enc in encuentros:
                todos_puntos.extend(enc['puntos_comunes'])
            if todos_puntos:
                n_puntos = min(10, len(todos_puntos))
                indices_puntos = np.linspace(0, len(todos_puntos)-1, n_puntos, dtype=int)
                puntos_enc = [todos_puntos[i] for i in indices_puntos]
                ax.scatter(x_activos[puntos_enc], y_activos[puntos_enc], 
                          c='red', s=20, alpha=0.9, zorder=15)
        
        # Título del subplot: fecha/hora y estado
        estado_str = 'ABIERTO' if estado_final == 1 else 'CERRADO'
        ax.set_title(f"{tiempo_str}\n{estado_str}", fontsize=9, fontweight='bold')
        
        # Quitar etiquetas de ejes para ahorrar espacio
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Leyenda global para toda la figura
    legend_elements_global = [
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='ROI'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label=f'X={x_limit}'),
        Line2D([0], [0], color='lime', lw=3, label='Límite Real Superior'),
        Line2D([0], [0], color='cyan', lw=3, label='Límite Real Inferior'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', 
               markersize=4, alpha=0.3, label='Perímetro'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', 
               markersize=5, label='Puntos Encuentro'),
        Line2D([0], [0], color='gray', marker='s', linestyle='None',
               markersize=5, label='Zonas ROI'),
        Line2D([0], [0], color='purple', lw=1.5, label='Seguimientos')
    ]
    fig.legend(handles=legend_elements_global, loc='upper center', 
               ncol=4, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    
    # Título general (sin "Parte")
    fecha_titulo = pd.to_datetime(dia_analizar).strftime('%Y-%m-%d')
    fig.suptitle(f"Seguimiento desde Extremos - {fecha_titulo}", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Ajustar layout para dejar espacio a la leyenda
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Guardar imagen
    fecha_str = dia_analizar.replace('-', '')
    nombre_imagen = f"Subplots_Extremos_{fecha_str}.png"
    ruta_imagen = os.path.join(carpeta_salida, nombre_imagen)
    plt.savefig(ruta_imagen, dpi=200, bbox_inches='tight')
    print(f"Imagen guardada: {ruta_imagen}")
    plt.close(fig)
    
    ds.close()
    print(f"✓ Imagen compuesta generada para {dia_analizar}")

# ===================== NUEVA FUNCIÓN: IMAGEN INDIVIDUAL =====================
def generar_imagen_instante(archivo_aperturas, tiempo_deseado_str, carpeta_salida,
                            x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA,
                            tolerancia_roi=TOLERANCIA_ROI, roi_points=ROI1_POINTS):
    """
    Genera una imagen en alta definición del instante especificado,
    con el mismo formato de leyenda, colores y escritos que en los videos.
    Los ejes se escalan correctamente para ese instante (con margen).
    """
    print(f"\n{'─'*50}")
    print(f"GENERANDO IMAGEN INDIVIDUAL PARA {tiempo_deseado_str}")
    print(f"{'─'*50}")

    # Abrir dataset
    ds = xr.open_dataset(archivo_aperturas)

    # Encontrar tiempo más cercano
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    tiempo_real = ds.time.values[idx_tiempo]
    tiempo_real_str = pd.to_datetime(tiempo_real).strftime('%Y-%m-%d %H:%M')

    print(f"Tiempo encontrado: {tiempo_real_str} (índice: {idx_tiempo})")

    if ds.existe_mascara_agua.values[idx_tiempo] == 0:
        print("ERROR: No existe máscara de agua para este tiempo")
        ds.close()
        return

    # Obtener puntos del perímetro
    perimetro = ds.perimetro_arena_agua.values[idx_tiempo, :]
    indices_activos = np.where(perimetro == 1)[0]

    if len(indices_activos) == 0:
        print("ERROR: No hay puntos de perímetro activos")
        ds.close()
        return

    x_activos = ds.coordenada_x.values[indices_activos]
    y_activos = ds.coordenada_y.values[indices_activos]

    # --- Análisis completo (igual que en el video) ---
    # 1. Puntos ROI cercanos
    indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, roi_points, tolerancia_roi)
    # 2. Seguimientos desde extremos
    seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(
        x_activos, y_activos, indices_roi, x_limit, umbral_distancia
    )
    # 3. Encuentros
    encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
    # 4. Clasificación
    curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
    curvas_reales = [c for c in curvas_clasificadas if c['tipo'] == 'limite_real_canal']
    otros_limites = [c for c in curvas_clasificadas if c['tipo'] == 'otro_limite']

    # Estado (del archivo)
    estado_archivo, _ = obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_real_str)
    estado_final = estado_archivo

    # Seleccionar límites reales si corresponde
    if estado_final == 1 and len(curvas_reales) >= 2:
        curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
        limite1_info = curvas_reales[0]
        limite2_info = curvas_reales[1]
    else:
        limite1_info = None
        limite2_info = None

    # --- Configuración de la figura ---
    fig, ax = plt.subplots(figsize=(16, 12))

    # Calcular límites basados en los puntos activos (con margen)
    x_min, x_max = np.min(x_activos), np.max(x_activos)
    y_min, y_max = np.min(y_activos), np.max(y_activos)
    margen = 20
    x_min -= margen
    x_max += margen
    y_min -= margen
    y_max += margen

    # Forzar relación de aspecto 1:1 (ajustando al rango más grande)
    dx = x_max - x_min
    dy = y_max - y_min
    centro_x = (x_min + x_max) / 2
    centro_y = (y_min + y_max) / 2
    lado_max = max(dx, dy) / 2
    x_lims = (centro_x - lado_max, centro_x + lado_max)
    y_lims = (centro_y + lado_max, centro_y - lado_max)  # invertido porque y crece hacia abajo en imagen

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Coordenada X', fontsize=12)
    ax.set_ylabel('Coordenada Y', fontsize=12)
    ax.set_title(f'Estado de Conexión - Seguimiento desde Extremos\n{tiempo_real_str}',
                 fontsize=16, fontweight='bold', pad=20)

    # --- Dibujar elementos (estilo video) ---
    # ROI
    roi_poly = Polygon(roi_points, closed=True, linewidth=3,
                       edgecolor='orange', facecolor='none',
                       linestyle='--', alpha=0.7)
    ax.add_patch(roi_poly)

    # Perímetro (puntos azules)
    ax.scatter(x_activos, y_activos, c='blue', s=10, alpha=0.3, label='Perímetro')

    # Zonas ROI (colores dinámicos, sin etiqueta en leyenda)
    colores_zonas = plt.cm.tab20(np.linspace(0, 1, len(zonas)))
    for i, zona in enumerate(zonas):
        if len(zona) > 0:
            x_zona = x_activos[zona]
            y_zona = y_activos[zona]
            ax.scatter(x_zona, y_zona, c=[colores_zonas[i]], s=80, marker='s',
                       alpha=0.8, edgecolors='black', linewidths=1.5, zorder=10)

    # Línea X = x_limit
    ax.axvline(x=x_limit, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=f'X={x_limit}')

    # Seguimientos (líneas de colores con círculo inicial)
    colores_curvas = plt.cm.Set3(np.linspace(0, 1, len(seguimientos_por_curva)))
    for k, (curva_num, info_curva) in enumerate(list(seguimientos_por_curva.items())[:8]):
        seguimientos = info_curva['seguimientos']
        color_curva = colores_curvas[k % len(colores_curvas)]
        for seg_idx, seg in enumerate(seguimientos[:2]):
            if len(seg['secuencia']) > 0:
                x_seq = x_activos[seg['secuencia']]
                y_seq = y_activos[seg['secuencia']]
                ax.plot(x_seq, y_seq, '-', color=color_curva, alpha=0.7, linewidth=2.5)
                # Círculo en el punto inicial
                x_ini = x_activos[seg['indice_inicio']]
                y_ini = y_activos[seg['indice_inicio']]
                ax.scatter(x_ini, y_ini, s=80, color=color_curva, edgecolors='black',
                           linewidths=1.5, alpha=0.9, zorder=20)

    # Límites reales
    if limite1_info is not None:
        x_lim1 = x_activos[limite1_info['curva_puntos']]
        y_lim1 = y_activos[limite1_info['curva_puntos']]
        ax.plot(x_lim1, y_lim1, 'lime', linewidth=6, alpha=0.9, label='Límite Real Superior')
    if limite2_info is not None:
        x_lim2 = x_activos[limite2_info['curva_puntos']]
        y_lim2 = y_activos[limite2_info['curva_puntos']]
        ax.plot(x_lim2, y_lim2, 'cyan', linewidth=6, alpha=0.9, label='Límite Real Inferior')

    # Otros límites (amarillo, sin etiqueta)
    for idx_otro, otro in enumerate(otros_limites[:2]):
        x_otro = x_activos[otro['curva_puntos']]
        y_otro = y_activos[otro['curva_puntos']]
        ax.plot(x_otro, y_otro, 'yellow', linewidth=4, alpha=0.7)

    # Puntos de encuentro (solo algunos para no saturar)
    if len(encuentros) > 0:
        todos_puntos = []
        for enc in encuentros:
            todos_puntos.extend(enc['puntos_comunes'])
        if todos_puntos:
            n_puntos = min(20, len(todos_puntos))
            indices_puntos = np.linspace(0, len(todos_puntos)-1, n_puntos, dtype=int)
            puntos_enc = [todos_puntos[i] for i in indices_puntos]
            ax.scatter(x_activos[puntos_enc], y_activos[puntos_enc],
                       c='red', s=50, alpha=0.9, zorder=15, label='Puntos Encuentro')

    # --- Cuadro de texto informativo (igual que en video) ---
    estado_color = 'lightgreen' if estado_final == 1 else 'lightcoral'
    info_detallada = (
        f'ESTADO: {"ABIERTO" if estado_final == 1 else "CERRADO"}\n'
        f'Curvas perimetrales: {len(curvas)}\n'
        f'Zonas ROI: {len(zonas)}\n'
        f'Seguimientos activos: {sum(len(info["seguimientos"]) for info in seguimientos_por_curva.values())}\n'
        f'Límites (3 criterios): {len(curvas_reales)}\n'
        f'Otros límites: {len(otros_limites)}\n'
        f'Encuentros: {len(encuentros)}'
    )
    ax.text(0.02, 0.98, info_detallada,
            transform=ax.transAxes, verticalalignment='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=estado_color, alpha=0.9))

    # --- Leyenda (exactamente como en video) ---
    legend_elements = [
        Line2D([0], [0], color='orange', lw=3, linestyle='--', label='ROI'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'X={x_limit}'),
        Line2D([0], [0], color='lime', lw=6, label='Límite Real Superior'),
        Line2D([0], [0], color='cyan', lw=6, label='Límite Real Inferior'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None',
               markersize=6, alpha=0.3, label='Perímetro'),
        Line2D([0], [0], color='red', marker='o', linestyle='None',
               markersize=8, label='Puntos Encuentro')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, ncol=2)

    # Ajustar layout y guardar
    plt.tight_layout()

    # Crear nombre de archivo
    fecha_hora_str = tiempo_real_str.replace('-', '').replace(' ', '_').replace(':', '')
    nombre_imagen = f"Instante_{fecha_hora_str}_HD.png"
    ruta_imagen = os.path.join(carpeta_salida, nombre_imagen)
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    print(f"Imagen guardada: {ruta_imagen}")
    plt.close(fig)

    ds.close()
    print("✓ Imagen individual generada exitosamente")


# ===================== MAIN MODIFICADO =====================
def main():
    """Función principal - Genera videos para día específico y la imagen del instante 10:00"""
    # Configuración de rutas
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    
    # Buscar el archivo más reciente de aperturas (usar la misma carpeta que el análisis)
    carpeta_aperturas = os.path.join(carpeta_base, 'Resultados_Aperturas_ROI1-FINAL')
    
    if not os.path.exists(carpeta_aperturas):
        print(f"ERROR: No existe la carpeta de resultados: {carpeta_aperturas}")
        print("Ejecuta primero el código de análisis completo de aperturas.")
        return
    
    # Buscar archivo más reciente
    archivos_nc = [f for f in os.listdir(carpeta_aperturas) if f.endswith('.nc') and 'COMPLETO' in f]
    
    if not archivos_nc:
        print("ERROR: No se encontraron archivos NetCDF de aperturas en la carpeta")
        return
    
    # Tomar el más reciente
    archivo_aperturas = os.path.join(carpeta_aperturas, sorted(archivos_nc)[-1])
    
    print("="*80)
    print("GENERACIÓN DE VIDEOS CON MÉTODO DESDE EXTREMOS")
    print("="*80)
    print(f"• Día a procesar: {DIA_ANALIZAR}")
    print(f"• Archivo de aperturas: {archivo_aperturas}")
    print(f"• FPS Video: {FPS_VIDEO}")
    print(f"• Método: IDÉNTICO al análisis (Seguimiento desde extremos)")
    print("="*80)
    
    # Leer dataset para obtener los tiempos del día específico
    ds = xr.open_dataset(archivo_aperturas)
    
    # Filtrar por día
    fecha_analizar = np.datetime64(DIA_ANALIZAR)
    mascara_dia = pd.to_datetime(ds.time.values).date == pd.to_datetime(fecha_analizar).date()
    indices_dia = np.where(mascara_dia)[0]
    
    if len(indices_dia) == 0:
        print(f"ERROR: No hay datos para el día {DIA_ANALIZAR}")
        ds.close()
        return
    
    print(f" Encontrados {len(indices_dia)} instantes para el día {DIA_ANALIZAR}")
    
    # Crear carpeta para videos
    fecha_str = DIA_ANALIZAR.replace('-', '')
    carpeta_videos = os.path.join(carpeta_base, f'Videos_Aperturas_{fecha_str}-FINAL')
    os.makedirs(carpeta_videos, exist_ok=True)
    
    # Estadísticas
    estadisticas = {
        'total': len(indices_dia),
        'abiertos': 0,
        'cerrados': 0,
        'con_curvas_reales': 0,
        'sin_curvas_reales': 0
    }
    
    # Generar video para cada instante del día
    print(f"\nGenerando videos con método desde extremos para cada instante del día {DIA_ANALIZAR}...")
    
    for idx in indices_dia:
        tiempo = ds.time.values[idx]
        tiempo_str = pd.to_datetime(tiempo).strftime('%Y-%m-%d %H:%M')
        hora_str = pd.to_datetime(tiempo).strftime('%H%M')
        
        # Obtener estado del archivo
        estado_archivo = int(ds.estado_apertura.values[idx])
        estado_str = "ABIERTO" if estado_archivo == 1 else "CERRADO"
        
        if estado_archivo == 1:
            estadisticas['abiertos'] += 1
        else:
            estadisticas['cerrados'] += 1
        
        # Crear nombre de archivo de video
        archivo_video = os.path.join(
            carpeta_videos, 
            f'Video_Extremos_{fecha_str}_{hora_str}_{estado_str}_FPS{FPS_VIDEO}.mp4'
        )
        
        print(f"\n  Procesando: {tiempo_str} ({estado_str})")
        
        try:
            estado_calculado, curvas_reales = generar_video_seguimiento_desde_extremos(
                archivo_aperturas, tiempo_str, archivo_video, 
                x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA
            )
            
            if curvas_reales >= 2:
                estadisticas['con_curvas_reales'] += 1
                print(f"  ✓ {curvas_reales} curvas reales identificadas")
            else:
                estadisticas['sin_curvas_reales'] += 1
                print(f"  • Solo {curvas_reales} curva(s) real(es)")
                
        except Exception as e:
            print(f"  ✗ Error generando video para {tiempo_str}: {e}")
    
    ds.close()
    
    # Calcular porcentajes
    if estadisticas['total'] > 0:
        porc_abiertos = estadisticas['abiertos'] / estadisticas['total'] * 100
        porc_cerrados = estadisticas['cerrados'] / estadisticas['total'] * 100
        porc_con_curvas = estadisticas['con_curvas_reales'] / estadisticas['total'] * 100
    else:
        porc_abiertos = porc_cerrados = porc_con_curvas = 0
    
    print("\n" + "="*80)
    print("RESUMEN DE GENERACIÓN DE VIDEOS CON MÉTODO DESDE EXTREMOS")
    print("="*80)
    print(f"• Día procesado: {DIA_ANALIZAR}")
    print(f"• Instantes procesados: {estadisticas['total']}")
    print(f"• Distribución de estados:")
    print(f"  - ABIERTOS: {estadisticas['abiertos']} ({porc_abiertos:.1f}%)")
    print(f"  - CERRADOS: {estadisticas['cerrados']} ({porc_cerrados:.1f}%)")
    print(f"• Curvas reales identificadas:")
    print(f"  - Con ≥2 curvas reales: {estadisticas['con_curvas_reales']} ({porc_con_curvas:.1f}%)")
    print(f"  - Con <2 curvas reales: {estadisticas['sin_curvas_reales']} ({100-porc_con_curvas:.1f}%)")
    print(f"• Carpeta de videos: {carpeta_videos}")
    print(f"• FPS utilizado: {FPS_VIDEO}")
    print(f"• Características del método:")
    print(f"  - IDÉNTICO al análisis principal")
    print(f"  - Seguimiento desde extremos de curvas")
    print(f"  - 2 seguimientos por curva (desde cada extremo)")
    print(f"  - 3 criterios: Encuentro + Zonas distintas + Cruce X=470")
    print("="*80)
    
    # Mostrar recomendaciones
    print("\nINTERPRETACIÓN DE RESULTADOS:")
    print("• Estado ABIERTO: ≥2 curvas cumplen los 3 criterios")
    print("• Límites reales: Curvas que cumplen los 3 criterios")
    print("• Otros límites: Curvas que solo cumplen el criterio 1 (encuentro)")
    print("• Sin límites reales: Estado CERRADO automáticamente")
    
    # --- Generar imagen compuesta con subplots (4 columnas, 3 filas) ---
    generar_imagen_subplots_dia(
        archivo_aperturas, DIA_ANALIZAR, carpeta_videos,
        x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA,
        tolerancia_roi=TOLERANCIA_ROI, roi_points=ROI1_POINTS
    )
    
    # --- NUEVO: Generar imagen individual para el instante 2024-07-10 10:00 ---
    instante_deseado = "2024-07-10 10:00"
    generar_imagen_instante(
        archivo_aperturas, instante_deseado, carpeta_videos,
        x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA,
        tolerancia_roi=TOLERANCIA_ROI, roi_points=ROI1_POINTS
    )
    
    print("\nPROCESO COMPLETADO")

if __name__ == "__main__":
    main()