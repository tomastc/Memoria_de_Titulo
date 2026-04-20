import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# ===================== CONFIGURACIÓN MODIFICABLE =====================
X_LIMIT = 470                # Línea de cruce del canal
UMBRAL_DISTANCIA = 10        # Umbral para vecino más cercano
TOLERANCIA_ROI = 5           # Tolerancia para puntos cerca de la ROI
FPS_VIDEO = 40               # Frames por segundo para videos
# =====================================================================

# Instantes a procesar (modificar aquí)
INSTANTE_1 = "2024-07-10 10:00"   # Ejemplo: instante abierto
INSTANTE_2 = "2024-01-01 16:00"   # Ejemplo: instante cerrado

# ROI fija
ROI1_POINTS = [(439, 100), (558, 100), (516, 600), (180, 600)]

# ===================== FUNCIONES AUXILIARES (ANÁLISIS) =====================
def distancia_punto_a_segmento_optimizada(p, a, b):
    p = np.array(p, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    punto_mas_cercano = a + t * ab
    distancia = np.linalg.norm(p - punto_mas_cercano)
    return distancia, punto_mas_cercano

def encontrar_todos_puntos_roi_cercanos(x_puntos, y_puntos, roi_points, tolerancia):
    segmentos = []
    n_vertices = len(roi_points)
    for i in range(n_vertices):
        a = roi_points[i]
        b = roi_points[(i + 1) % n_vertices]
        segmentos.append((a, b))
    indices_cercanos = []
    roi_x_min = min(p[0] for p in roi_points) - tolerancia
    roi_x_max = max(p[0] for p in roi_points) + tolerancia
    roi_y_min = min(p[1] for p in roi_points) - tolerancia
    roi_y_max = max(p[1] for p in roi_points) + tolerancia
    for i in range(len(x_puntos)):
        x, y = x_puntos[i], y_puntos[i]
        if not (roi_x_min <= x <= roi_x_max and roi_y_min <= y <= roi_y_max):
            continue
        p = (x, y)
        dist_min = float('inf')
        for segmento in segmentos:
            a, b = segmento
            distancia, _ = distancia_punto_a_segmento_optimizada(p, a, b)
            if distancia < dist_min:
                dist_min = distancia
            if dist_min <= tolerancia:
                break
        if dist_min <= tolerancia:
            indices_cercanos.append(i)
    return indices_cercanos

def identificar_zonas_concentradas(x_puntos, y_puntos, indices_roi, eps=15):
    if len(indices_roi) == 0:
        return []
    puntos_roi = np.column_stack([x_puntos[indices_roi], y_puntos[indices_roi]])
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(puntos_roi)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    zonas = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        zona_indices = [indices_roi[idx] for idx in cluster_indices]
        zonas.append(zona_indices)
    return zonas

def seleccionar_punto_mas_cercano_a_roi(x_puntos, y_puntos, zona_indices, roi_points):
    if len(zona_indices) == 0:
        return None
    distancias = []
    for idx in zona_indices:
        p = (x_puntos[idx], y_puntos[idx])
        dist_min = float('inf')
        n_vertices = len(roi_points)
        for i in range(n_vertices):
            a = roi_points[i]
            b = roi_points[(i + 1) % n_vertices]
            distancia, _ = distancia_punto_a_segmento_optimizada(p, a, b)
            if distancia < dist_min:
                dist_min = distancia
        distancias.append(dist_min)
    idx_mas_cercano = zona_indices[np.argmin(distancias)]
    return idx_mas_cercano

def seguir_perimetro_desde_punto(x_puntos, y_puntos, punto_inicio_idx, umbral_distancia, max_puntos=None, direccion='una'):
    n_puntos = len(x_puntos)
    visitados = np.zeros(n_puntos, dtype=bool)
    secuencia = []
    idx_actual = punto_inicio_idx
    contador_puntos = 0
    while True:
        if max_puntos is not None and contador_puntos >= max_puntos:
            break
        visitados[idx_actual] = True
        secuencia.append(idx_actual)
        contador_puntos += 1
        distancias = np.sqrt((x_puntos - x_puntos[idx_actual])**2 + (y_puntos - y_puntos[idx_actual])**2)
        distancias[visitados] = np.inf
        if np.all(distancias == np.inf):
            break
        idx_siguiente = np.argmin(distancias)
        if distancias[idx_siguiente] > umbral_distancia:
            break
        idx_actual = idx_siguiente
    if direccion == 'ambas' and len(secuencia) > 1:
        idx_actual = punto_inicio_idx
        visitados_contrario = visitados.copy()
        for idx in secuencia:
            visitados_contrario[idx] = True
        distancias = np.sqrt((x_puntos - x_puntos[punto_inicio_idx])**2 + (y_puntos - y_puntos[punto_inicio_idx])**2)
        distancias[visitados_contrario] = np.inf
        if not np.all(distancias == np.inf):
            idx_otro_vecino = np.argmin(distancias)
            if distancias[idx_otro_vecino] <= umbral_distancia:
                secuencia_contraria = [punto_inicio_idx]
                idx_actual = idx_otro_vecino
                contador_contrario = 0
                while True:
                    if max_puntos is not None and contador_contrario >= max_puntos:
                        break
                    visitados_contrario[idx_actual] = True
                    secuencia_contraria.append(idx_actual)
                    contador_contrario += 1
                    distancias = np.sqrt((x_puntos - x_puntos[idx_actual])**2 + (y_puntos - y_puntos[idx_actual])**2)
                    distancias[visitados_contrario] = np.inf
                    if np.all(distancias == np.inf):
                        break
                    idx_siguiente = np.argmin(distancias)
                    if distancias[idx_siguiente] > umbral_distancia:
                        break
                    idx_actual = idx_siguiente
                secuencia_contraria_invertida = list(reversed(secuencia_contraria))
                secuencia_completa = secuencia_contraria_invertida[:-1] + secuencia
                return secuencia_completa
    return secuencia

def verificar_cruce_x470(secuencia, x_puntos, x_limit):
    if len(secuencia) < 2:
        return False
    lados = []
    for idx in secuencia:
        x = x_puntos[idx]
        lados.append(-1 if x < x_limit else 1)
    for i in range(1, len(lados)):
        if lados[i] != lados[i-1]:
            return True
    return False

def encontrar_curvas_perimetrales(x_puntos, y_puntos, umbral_distancia):
    n_puntos = len(x_puntos)
    visitados = np.zeros(n_puntos, dtype=bool)
    curvas = []
    for i in range(n_puntos):
        if not visitados[i]:
            secuencia = seguir_perimetro_desde_punto(x_puntos, y_puntos, i, umbral_distancia, direccion='ambas')
            for idx in secuencia:
                visitados[idx] = True
            if len(secuencia) > 1:
                curvas.append(secuencia)
    return curvas

def identificar_extremos_curva_con_roi(curva, x_puntos, y_puntos, indices_roi):
    extremos = []
    ventana = min(10, len(curva) // 4)
    inicio_curva = curva[:ventana]
    final_curva = curva[-ventana:]
    puntos_inicio = [idx for idx in inicio_curva if idx in indices_roi]
    puntos_final = [idx for idx in final_curva if idx in indices_roi]
    if puntos_inicio:
        extremos.append(puntos_inicio)
    if puntos_final:
        extremos.append(puntos_final)
    return extremos

def encontrar_seguimientos_desde_extremos(x_puntos, y_puntos, indices_roi, x_limit, umbral_distancia):
    curvas = encontrar_curvas_perimetrales(x_puntos, y_puntos, umbral_distancia)
    zonas = identificar_zonas_concentradas(x_puntos, y_puntos, indices_roi, eps=15)
    seguimientos_por_curva = {}
    for curva_idx, curva in enumerate(curvas):
        extremos_curva = identificar_extremos_curva_con_roi(curva, x_puntos, y_puntos, indices_roi)
        if len(extremos_curva) >= 2:
            extremos_con_zona = []
            for puntos_extremo in extremos_curva:
                zona_encontrada = None
                for zona_idx, zona in enumerate(zonas):
                    if any(p in zona for p in puntos_extremo):
                        zona_encontrada = zona_idx
                        break
                if zona_encontrada is not None:
                    punto_mas_cercano = seleccionar_punto_mas_cercano_a_roi(x_puntos, y_puntos, puntos_extremo, ROI1_POINTS)
                    if punto_mas_cercano is not None:
                        extremos_con_zona.append({'zona': zona_encontrada, 'punto_inicio': punto_mas_cercano, 'puntos_extremo': puntos_extremo})
            if len(extremos_con_zona) >= 2:
                seguimientos_extremos = []
                for extremo_info in extremos_con_zona:
                    punto_inicio = extremo_info['punto_inicio']
                    secuencia = seguir_perimetro_desde_punto(x_puntos, y_puntos, punto_inicio, umbral_distancia, max_puntos=1000, direccion='una')
                    cruza_x470 = verificar_cruce_x470(secuencia, x_puntos, x_limit)
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
                if len(seguimientos_extremos) >= 2:
                    seguimientos_por_curva[curva_idx] = {'curva': curva, 'seguimientos': seguimientos_extremos}
    return seguimientos_por_curva, zonas, curvas

def identificar_encuentros_seguimientos(seguimientos_por_curva):
    encuentros = []
    for curva_idx, info_curva in seguimientos_por_curva.items():
        seguimientos = info_curva['seguimientos']
        for i in range(len(seguimientos)):
            for j in range(i+1, len(seguimientos)):
                seg_i = seguimientos[i]
                seg_j = seguimientos[j]
                if seg_i['zona_inicio'] == seg_j['zona_inicio']:
                    continue
                puntos_i = set(seg_i['secuencia'])
                puntos_j = set(seg_j['secuencia'])
                puntos_i_sin_extremo = puntos_i - {seg_i['indice_inicio']}
                puntos_j_sin_extremo = puntos_j - {seg_j['indice_inicio']}
                puntos_comunes = puntos_i_sin_extremo.intersection(puntos_j_sin_extremo)
                if len(puntos_comunes) > 5:
                    encuentros.append({
                        'curva_idx': curva_idx,
                        'seguimiento_i': i,
                        'seguimiento_j': j,
                        'puntos_comunes': list(puntos_comunes),
                        'cantidad_comunes': len(puntos_comunes),
                        'cumple_criterio_1': True
                    })
    return encuentros

def clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_puntos, x_limit):
    curvas_clasificadas = []
    for encuentro in encuentros:
        curva_idx = encuentro['curva_idx']
        info_curva = seguimientos_por_curva[curva_idx]
        seg_i = info_curva['seguimientos'][encuentro['seguimiento_i']]
        seg_j = info_curva['seguimientos'][encuentro['seguimiento_j']]
        criterio1 = encuentro['cumple_criterio_1']
        criterio2 = seg_i['zona_inicio'] != seg_j['zona_inicio']
        criterio3 = seg_i['cruza_x470'] and seg_j['cruza_x470']
        if criterio1 and criterio2 and criterio3:
            tipo = "limite_real_canal"
        elif criterio1 and not criterio2:
            tipo = "otro_limite"
        else:
            tipo = "no_cumple"
        curva_combinada = list(set(seg_i['secuencia'] + seg_j['secuencia']))
        curva_combinada.sort(key=lambda idx: seg_i['secuencia'].index(idx) if idx in seg_i['secuencia'] else len(seg_i['secuencia']) + seg_j['secuencia'].index(idx) if idx in seg_j['secuencia'] else 0)
        curvas_clasificadas.append({
            'curva_idx': curva_idx,
            'tipo': tipo,
            'cumple_3_criterios': (criterio1 and criterio2 and criterio3),
            'longitud': len(curva_combinada),
            'curva_puntos': curva_combinada,
            'seguimiento_i': seg_i,
            'seguimiento_j': seg_j,
            'criterios': {'encuentro': criterio1, 'distintas_zonas': criterio2, 'cruza_x470': criterio3}
        })
    return curvas_clasificadas

def obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_deseado_str):
    ds = xr.open_dataset(archivo_aperturas)
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    estado = int(ds.estado_apertura.values[idx_tiempo])
    if 'cantidad_limites_reales' in ds:
        limites_reales = int(ds.cantidad_limites_reales.values[idx_tiempo])
    else:
        limites_reales = 2 if estado == 1 else 0
    ds.close()
    return estado, limites_reales

# ===================== FUNCIÓN PARA GENERAR VIDEO =====================
def generar_video_seguimiento_desde_extremos(archivo_aperturas, tiempo_deseado_str, archivo_video, 
                                             x_limit, umbral_distancia):
    print(f"\nGenerando video para {tiempo_deseado_str}")
    estado_archivo, _ = obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_deseado_str)
    ds = xr.open_dataset(archivo_aperturas)
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    tiempo_real = ds.time.values[idx_tiempo]
    tiempo_real_str = pd.to_datetime(tiempo_real).strftime('%Y-%m-%d %H:%M')
    if ds.existe_mascara_agua.values[idx_tiempo] == 0:
        print("  No hay máscara de agua")
        ds.close()
        return
    perimetro = ds.perimetro_arena_agua.values[idx_tiempo, :]
    indices_activos = np.where(perimetro == 1)[0]
    if len(indices_activos) == 0:
        print("  No hay perímetro")
        ds.close()
        return
    x_activos = ds.coordenada_x.values[indices_activos]
    y_activos = ds.coordenada_y.values[indices_activos]
    indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, ROI1_POINTS, TOLERANCIA_ROI)
    seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(x_activos, y_activos, indices_roi, x_limit, umbral_distancia)
    encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
    curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
    curvas_reales = [c for c in curvas_clasificadas if c['tipo'] == 'limite_real_canal']
    otros_limites = [c for c in curvas_clasificadas if c['tipo'] == 'otro_limite']
    estado_final = estado_archivo
    if estado_final == 1 and len(curvas_reales) >= 2:
        curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
        limite1_info = curvas_reales[0]
        limite2_info = curvas_reales[1]
    else:
        limite1_info = None
        limite2_info = None
    # Configurar figura
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(np.min(x_activos)-20, np.max(x_activos)+20)
    ax.set_ylim(np.max(y_activos)+20, np.min(y_activos)-20)
    ax.set_xlabel('Coordenada X', fontsize=12)
    ax.set_ylabel('Coordenada Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f'Estado de Conexión - Seguimiento desde Extremos\n{tiempo_real_str}', fontsize=16, fontweight='bold', pad=20)
    # ROI
    roi_poly = Polygon(ROI1_POINTS, closed=True, linewidth=3, edgecolor='orange', facecolor='none', linestyle='--', alpha=0.7, label='ROI')
    ax.add_patch(roi_poly)
    # Perímetro
    ax.scatter(x_activos, y_activos, c='blue', s=10, alpha=0.3, label='Perímetro')
    # Zonas ROI
    colores_zonas = plt.cm.tab20(np.linspace(0, 1, len(zonas)))
    for i, zona in enumerate(zonas):
        if len(zona) > 0:
            ax.scatter(x_activos[zona], y_activos[zona], c=[colores_zonas[i]], s=80, marker='s', alpha=0.8, edgecolors='black', linewidths=1.5, zorder=10)
    # Línea X
    ax.axvline(x=x_limit, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'X={x_limit}')
    # Preparar animación
    seguimientos_anim = []
    colores_curvas = plt.cm.Set3(np.linspace(0, 1, len(seguimientos_por_curva)))
    for curva_idx, (curva_num, info_curva) in enumerate(list(seguimientos_por_curva.items())[:8]):
        seguimientos = info_curva['seguimientos']
        for seg_idx, seg in enumerate(seguimientos[:2]):
            if len(seg['secuencia']) > 0:
                x_inicio = x_activos[seg['indice_inicio']]
                y_inicio = y_activos[seg['indice_inicio']]
                import matplotlib.patches as patches
                circulo = patches.Circle((x_inicio, y_inicio), radius=8, color=colores_curvas[curva_idx], alpha=0.9, fill=True, zorder=20, linewidth=2, edgecolor='black')
                ax.add_patch(circulo)
                seguimientos_anim.append({
                    'circulo': circulo,
                    'secuencia': seg['secuencia'],
                    'x_vals': [],
                    'y_vals': [],
                    'linea': ax.plot([], [], '-', color=colores_curvas[curva_idx], alpha=0.7, linewidth=2.5)[0],
                    'cruza_x470': seg['cruza_x470'],
                    'curva_idx': curva_num,
                    'seg_idx': seg_idx,
                    'indice_actual': 0,
                    'zona_inicio': seg['zona_inicio']
                })
    # Límites reales y encuentros
    limite_sup_linea, = ax.plot([], [], 'lime', linewidth=6, alpha=0.9, label='Límite Real Superior')
    limite_inf_linea, = ax.plot([], [], 'cyan', linewidth=6, alpha=0.9, label='Límite Real Inferior')
    encuentro_points = ax.plot([], [], 'ro', markersize=10, alpha=0.9, label='Puntos Encuentro')[0]
    otros_limites_lineas = []
    for i in range(min(2, len(otros_limites))):
        linea, = ax.plot([], [], 'yellow', linewidth=4, alpha=0.7)
        otros_limites_lineas.append(linea)
    # Funciones de animación
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
        if seguimientos_anim:
            longitudes = [len(s['secuencia']) for s in seguimientos_anim]
            max_longitud = max(longitudes) if longitudes else 1
        else:
            max_longitud = 1
        progreso = min(frame / (max_longitud * 1.3), 1.0)
        for seg_anim in seguimientos_anim:
            if len(seg_anim['secuencia']) > 0:
                puntos_a_mostrar = int(progreso * len(seg_anim['secuencia']))
                puntos_a_mostrar = max(1, puntos_a_mostrar)
                if puntos_a_mostrar > 0:
                    idx = puntos_a_mostrar - 1
                    x = x_activos[seg_anim['secuencia'][idx]]
                    y = y_activos[seg_anim['secuencia'][idx]]
                    seg_anim['circulo'].center = (x, y)
                    seg_anim['x_vals'] = x_activos[seg_anim['secuencia'][:puntos_a_mostrar]].tolist()
                    seg_anim['y_vals'] = y_activos[seg_anim['secuencia'][:puntos_a_mostrar]].tolist()
                    seg_anim['linea'].set_data(seg_anim['x_vals'], seg_anim['y_vals'])
                    seg_anim['indice_actual'] = idx
        if progreso >= 0.6:
            if limite1_info is not None:
                limite_sup_linea.set_data(x_activos[limite1_info['curva_puntos']], y_activos[limite1_info['curva_puntos']])
            if limite2_info is not None:
                limite_inf_linea.set_data(x_activos[limite2_info['curva_puntos']], y_activos[limite2_info['curva_puntos']])
            for i, linea in enumerate(otros_limites_lineas):
                if i < len(otros_limites):
                    curva_info = otros_limites[i]
                    linea.set_data(x_activos[curva_info['curva_puntos']], y_activos[curva_info['curva_puntos']])
            if len(encuentros) > 0:
                todos_puntos_encuentro = []
                for enc in encuentros:
                    todos_puntos_encuentro.extend(enc['puntos_comunes'])
                if todos_puntos_encuentro:
                    puntos_a_mostrar = min(20, len(todos_puntos_encuentro))
                    indices_mostrar = np.linspace(0, len(todos_puntos_encuentro)-1, puntos_a_mostrar, dtype=int)
                    puntos_seleccionados = [todos_puntos_encuentro[i] for i in indices_mostrar]
                    encuentro_points.set_data(x_activos[puntos_seleccionados], y_activos[puntos_seleccionados])
        elementos = [limite_sup_linea, limite_inf_linea, encuentro_points] + otros_limites_lineas
        elementos.extend([s['circulo'] for s in seguimientos_anim])
        elementos.extend([s['linea'] for s in seguimientos_anim])
        return elementos
    # Frames totales
    if seguimientos_anim:
        longitudes = [len(s['secuencia']) for s in seguimientos_anim]
        max_longitud = max(longitudes) if longitudes else 1
        max_frames = int(max_longitud * 1.3) + 40
    else:
        max_frames = 100
    # Cuadro de texto informativo
    estado_color = 'lightgreen' if estado_final == 1 else 'lightcoral'
    info_detallada = (f'ESTADO: {"ABIERTO" if estado_final == 1 else "CERRADO"}\n'
                      f'Curvas perimetrales: {len(curvas)}\n'
                      f'Zonas ROI: {len(zonas)}\n'
                      f'Seguimientos activos: {len(seguimientos_anim)}\n'
                      f'Límites (3 criterios): {len(curvas_reales)}\n'
                      f'Otros límites: {len(otros_limites)}\n'
                      f'Encuentros: {len(encuentros)}')
    ax.text(0.02, 0.98, info_detallada, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=estado_color, alpha=0.9))
    # Leyenda
    legend_elements = [
        Line2D([0], [0], color='orange', lw=3, linestyle='--', label='ROI'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'X={x_limit}'),
        Line2D([0], [0], color='lime', lw=6, label='Límite Real Superior'),
        Line2D([0], [0], color='cyan', lw=6, label='Límite Real Inferior'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, alpha=0.3, label='Perímetro'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=8, label='Puntos Encuentro')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    plt.tight_layout()
    # Crear animación y guardar
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=max_frames, interval=1000/FPS_VIDEO, blit=True)
    estado_str = "ABIERTO" if estado_final == 1 else "CERRADO"
    ani.save(archivo_video, writer='ffmpeg', fps=FPS_VIDEO, dpi=150,
             metadata={'title': f'Seguimiento {tiempo_real_str}', 'comment': f'Estado: {estado_str}'})
    plt.close(fig)
    ds.close()
    print(f"  Video guardado: {archivo_video}")

# ===================== FUNCIÓN PARA GENERAR IMAGEN ESTÁTICA =====================
def generar_imagen_instante(archivo_aperturas, tiempo_deseado_str, archivo_imagen,
                            x_limit, umbral_distancia):
    print(f"\nGenerando imagen para {tiempo_deseado_str}")
    ds = xr.open_dataset(archivo_aperturas)
    tiempo_deseado = np.datetime64(datetime.strptime(tiempo_deseado_str, "%Y-%m-%d %H:%M"))
    idx_tiempo = np.argmin(np.abs(ds.time.values - tiempo_deseado))
    tiempo_real = ds.time.values[idx_tiempo]
    tiempo_real_str = pd.to_datetime(tiempo_real).strftime('%Y-%m-%d %H:%M')
    if ds.existe_mascara_agua.values[idx_tiempo] == 0:
        print("  No hay máscara de agua")
        ds.close()
        return
    perimetro = ds.perimetro_arena_agua.values[idx_tiempo, :]
    indices_activos = np.where(perimetro == 1)[0]
    if len(indices_activos) == 0:
        print("  No hay perímetro")
        ds.close()
        return
    x_activos = ds.coordenada_x.values[indices_activos]
    y_activos = ds.coordenada_y.values[indices_activos]
    indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, ROI1_POINTS, TOLERANCIA_ROI)
    seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(x_activos, y_activos, indices_roi, x_limit, umbral_distancia)
    encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
    curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
    curvas_reales = [c for c in curvas_clasificadas if c['tipo'] == 'limite_real_canal']
    otros_limites = [c for c in curvas_clasificadas if c['tipo'] == 'otro_limite']
    estado_archivo, _ = obtener_estado_apertura_desde_archivo(archivo_aperturas, tiempo_real_str)
    estado_final = estado_archivo
    if estado_final == 1 and len(curvas_reales) >= 2:
        curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
        limite1_info = curvas_reales[0]
        limite2_info = curvas_reales[1]
    else:
        limite1_info = None
        limite2_info = None
    # Configurar figura
    fig, ax = plt.subplots(figsize=(16, 12))
    # Calcular límites con margen y aspecto 1:1
    x_min, x_max = np.min(x_activos), np.max(x_activos)
    y_min, y_max = np.min(y_activos), np.max(y_activos)
    margen = 20
    x_min -= margen
    x_max += margen
    y_min -= margen
    y_max += margen
    dx = x_max - x_min
    dy = y_max - y_min
    centro_x = (x_min + x_max) / 2
    centro_y = (y_min + y_max) / 2
    lado_max = max(dx, dy) / 2
    x_lims = (centro_x - lado_max, centro_x + lado_max)
    y_lims = (centro_y + lado_max, centro_y - lado_max)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Coordenada X', fontsize=12)
    ax.set_ylabel('Coordenada Y', fontsize=12)
    ax.set_title(f'Estado de Conexión - Seguimiento desde Extremos\n{tiempo_real_str}', fontsize=16, fontweight='bold', pad=20)
    # ROI
    roi_poly = Polygon(ROI1_POINTS, closed=True, linewidth=3, edgecolor='orange', facecolor='none', linestyle='--', alpha=0.7)
    ax.add_patch(roi_poly)
    # Perímetro
    ax.scatter(x_activos, y_activos, c='blue', s=10, alpha=0.3, label='Perímetro')
    # Zonas ROI
    colores_zonas = plt.cm.tab20(np.linspace(0, 1, len(zonas)))
    for i, zona in enumerate(zonas):
        if len(zona) > 0:
            ax.scatter(x_activos[zona], y_activos[zona], c=[colores_zonas[i]], s=80, marker='s', alpha=0.8, edgecolors='black', linewidths=1.5, zorder=10)
    # Línea X
    ax.axvline(x=x_limit, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'X={x_limit}')
    # Seguimientos (líneas completas)
    colores_curvas = plt.cm.Set3(np.linspace(0, 1, len(seguimientos_por_curva)))
    for k, (curva_num, info_curva) in enumerate(list(seguimientos_por_curva.items())[:8]):
        seguimientos = info_curva['seguimientos']
        color_curva = colores_curvas[k % len(colores_curvas)]
        for seg_idx, seg in enumerate(seguimientos[:2]):
            if len(seg['secuencia']) > 0:
                x_seq = x_activos[seg['secuencia']]
                y_seq = y_activos[seg['secuencia']]
                ax.plot(x_seq, y_seq, '-', color=color_curva, alpha=0.7, linewidth=2.5)
                ax.scatter(x_activos[seg['indice_inicio']], y_activos[seg['indice_inicio']],
                           s=80, color=color_curva, edgecolors='black', linewidths=1.5, alpha=0.9, zorder=20)
    # Límites reales
    if limite1_info is not None:
        ax.plot(x_activos[limite1_info['curva_puntos']], y_activos[limite1_info['curva_puntos']],
                'lime', linewidth=6, alpha=0.9, label='Límite Real Superior')
    if limite2_info is not None:
        ax.plot(x_activos[limite2_info['curva_puntos']], y_activos[limite2_info['curva_puntos']],
                'cyan', linewidth=6, alpha=0.9, label='Límite Real Inferior')
    # Otros límites
    for idx_otro, otro in enumerate(otros_limites[:2]):
        ax.plot(x_activos[otro['curva_puntos']], y_activos[otro['curva_puntos']],
                'yellow', linewidth=4, alpha=0.7)
    # Puntos de encuentro
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
    # Cuadro de texto
    estado_color = 'lightgreen' if estado_final == 1 else 'lightcoral'
    info_detallada = (f'ESTADO: {"ABIERTO" if estado_final == 1 else "CERRADO"}\n'
                      f'Curvas perimetrales: {len(curvas)}\n'
                      f'Zonas ROI: {len(zonas)}\n'
                      f'Seguimientos activos: {sum(len(info["seguimientos"]) for info in seguimientos_por_curva.values())}\n'
                      f'Límites (3 criterios): {len(curvas_reales)}\n'
                      f'Otros límites: {len(otros_limites)}\n'
                      f'Encuentros: {len(encuentros)}')
    ax.text(0.02, 0.98, info_detallada, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor=estado_color, alpha=0.9))
    # Leyenda
    legend_elements = [
        Line2D([0], [0], color='orange', lw=3, linestyle='--', label='ROI'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'X={x_limit}'),
        Line2D([0], [0], color='lime', lw=6, label='Límite Real Superior'),
        Line2D([0], [0], color='cyan', lw=6, label='Límite Real Inferior'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, alpha=0.3, label='Perímetro'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=8, label='Puntos Encuentro')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    plt.tight_layout()
    plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight')
    plt.close(fig)
    ds.close()
    print(f"  Imagen guardada: {archivo_imagen}")

# ===================== MAIN =====================
def main():
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    carpeta_aperturas = os.path.join(carpeta_base, 'Resultados_Aperturas_ROI1-FINAL')
    if not os.path.exists(carpeta_aperturas):
        print(f"ERROR: No existe la carpeta de resultados: {carpeta_aperturas}")
        return
    archivos_nc = [f for f in os.listdir(carpeta_aperturas) if f.endswith('.nc') and 'COMPLETO' in f]
    if not archivos_nc:
        print("ERROR: No se encontraron archivos NetCDF")
        return
    archivo_aperturas = os.path.join(carpeta_aperturas, sorted(archivos_nc)[-1])
    print("="*60)
    print("GENERACIÓN PARA DOS INSTANTES")
    print("="*60)
    print(f"Archivo de aperturas: {archivo_aperturas}")
    print(f"Instante 1: {INSTANTE_1}")
    print(f"Instante 2: {INSTANTE_2}")
    print("-"*60)

    # Crear carpeta de salida
    carpeta_salida = os.path.join(carpeta_base, 'Resultados_Dos_Instantes')
    os.makedirs(carpeta_salida, exist_ok=True)

    instantes = [INSTANTE_1, INSTANTE_2]

    for instante in instantes:
        # Obtener estado para nombre de archivo
        estado, _ = obtener_estado_apertura_desde_archivo(archivo_aperturas, instante)
        estado_str = "ABIERTO" if estado == 1 else "CERRADO"
        fecha_hora = instante.replace('-', '').replace(' ', '_').replace(':', '')
        # Video
        nombre_video = f"Video_{fecha_hora}_{estado_str}_FPS{FPS_VIDEO}.mp4"
        ruta_video = os.path.join(carpeta_salida, nombre_video)
        generar_video_seguimiento_desde_extremos(archivo_aperturas, instante, ruta_video, X_LIMIT, UMBRAL_DISTANCIA)
        # Imagen
        nombre_imagen = f"Imagen_{fecha_hora}_{estado_str}_HD.png"
        ruta_imagen = os.path.join(carpeta_salida, nombre_imagen)
        generar_imagen_instante(archivo_aperturas, instante, ruta_imagen, X_LIMIT, UMBRAL_DISTANCIA)

    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print(f"Resultados guardados en: {carpeta_salida}")
    print("="*60)

if __name__ == "__main__":
    main()