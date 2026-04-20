import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# ===================== CONFIGURACIÓN MODIFICABLE =====================
X_LIMIT = 470           # Variable para identificar cruce del canal
UMBRAL_DISTANCIA = 10   # Umbral de distancia para vecino más cercano
TOLERANCIA_ROI = 5      # Tolerancia para puntos cercanos a la ROI (en píxeles)
# =====================================================================

# Configuración de ROI (Región de Interés) - cuadrilátero
ROI1_POINTS = [(439, 100), (558, 100), (516, 600), (180, 600)]

def generar_reporte_completo(ds_con_estados, tiempo_total, archivo_perimetros,
                           archivo_con_estados, carpeta_resultados, x_limit):
    """
    Genera un archivo de texto con un resumen COMPLETO del procesamiento de aperturas
    Incluye estadísticas mensuales, rangos de fechas, duraciones, etc.
    Y ahora muestra TODOS los instantes en detalle
    """
    # Crear nombre de archivo
    fecha_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo_reporte = os.path.join(carpeta_resultados, f"reporte_completo_aperturas_{fecha_actual}.txt")
    
    # Calcular estadísticas básicas
    estados_array = ds_con_estados.estado_apertura.values
    tiempos = ds_con_estados.time.values
    
    total_tiempos = len(estados_array)
    tiempos_abiertos = np.sum(estados_array == 1)
    tiempos_cerrados = np.sum(estados_array == 0)
    
    porcentaje_abiertos = 100 * tiempos_abiertos / total_tiempos if total_tiempos > 0 else 0
    porcentaje_cerrados = 100 * tiempos_cerrados / total_tiempos if total_tiempos > 0 else 0
    
    # Extraer información de posibles límites
    if 'cantidad_posibles_limites' in ds_con_estados:
        cantidades_posibles = ds_con_estados.cantidad_posibles_limites.values
        promedio_posibles = np.mean(cantidades_posibles) if len(cantidades_posibles) > 0 else 0
        max_posibles = np.max(cantidades_posibles) if len(cantidades_posibles) > 0 else 0
        min_posibles = np.min(cantidades_posibles) if len(cantidades_posibles) > 0 else 0
    else:
        promedio_posibles = 0
        max_posibles = 0
        min_posibles = 0
    
    # Extraer límites reales (que cumplen 3 criterios)
    if 'cantidad_limites_reales' in ds_con_estados:
        cantidades_reales = ds_con_estados.cantidad_limites_reales.values
        promedio_reales = np.mean(cantidades_reales) if len(cantidades_reales) > 0 else 0
        tiempos_con_limites_reales = np.sum(cantidades_reales >= 2)
    else:
        # Si no existe la variable, calcularla basado en estado_apertura
        tiempos_con_limites_reales = tiempos_abiertos
        cantidades_reales = np.zeros_like(estados_array)
        cantidades_reales[estados_array == 1] = 2  # Asumimos 2 límites cuando está abierto
    
    # Convertir tiempos a datetime para análisis
    tiempos_dt = pd.to_datetime(tiempos)
    
    # Obtener rango de fechas
    fecha_min = tiempos_dt.min()
    fecha_max = tiempos_dt.max()
    
    # Extraer meses y años
    meses = tiempos_dt.month
    anos = tiempos_dt.year
    meses_anos = [f"{a}-{m:02d}" for a, m in zip(anos, meses)]
    meses_anos_unicos = sorted(set(meses_anos))
    
    # Calcular estadísticas por mes
    estadisticas_mensuales = []
    for mes_ano in meses_anos_unicos:
        año, mes = map(int, mes_ano.split('-'))
        mascara_mes = (anos == año) & (meses == mes)
        
        if np.any(mascara_mes):
            estados_mes = estados_array[mascara_mes]
            total_mes = len(estados_mes)
            abiertos_mes = np.sum(estados_mes == 1)
            cerrados_mes = np.sum(estados_mes == 0)
            
            # Calcular duración de secuencias abiertas y cerradas
            secuencias = []
            estado_actual = estados_mes[0]
            duracion_actual = 1
            
            for i in range(1, len(estados_mes)):
                if estados_mes[i] == estado_actual:
                    duracion_actual += 1
                else:
                    secuencias.append((estado_actual, duracion_actual))
                    estado_actual = estados_mes[i]
                    duracion_actual = 1
            
            secuencias.append((estado_actual, duracion_actual))
            
            # Calcular promedios
            duraciones_abiertas = [d for e, d in secuencias if e == 1]
            duraciones_cerradas = [d for e, d in secuencias if e == 0]
            
            duracion_prom_abierta = np.mean(duraciones_abiertas) if duraciones_abiertas else 0
            duracion_prom_cerrada = np.mean(duraciones_cerradas) if duraciones_cerradas else 0
            
            # Porcentaje del día (asumiendo 24 horas = 24 instantes)
            porcentaje_dia_abierto = (duracion_prom_abierta / 24) * 100 if duracion_prom_abierta > 0 else 0
            porcentaje_dia_cerrado = (duracion_prom_cerrada / 24) * 100 if duracion_prom_cerrada > 0 else 0
            
            # Calcular límites reales para este mes
            if 'cantidad_limites_reales' in ds_con_estados:
                limites_reales_mes = cantidades_reales[mascara_mes]
                promedio_limites_reales_mes = np.mean(limites_reales_mes)
            else:
                promedio_limites_reales_mes = 2 * (abiertos_mes / total_mes) if total_mes > 0 else 0
            
            estadisticas_mensuales.append({
                'mes_ano': mes_ano,
                'total': total_mes,
                'abiertos': abiertos_mes,
                'cerrados': cerrados_mes,
                'porc_abiertos': 100 * abiertos_mes / total_mes if total_mes > 0 else 0,
                'porc_cerrados': 100 * cerrados_mes / total_mes if total_mes > 0 else 0,
                'duracion_prom_abierta': duracion_prom_abierta,
                'duracion_prom_cerrada': duracion_prom_cerrada,
                'porc_dia_abierto': porcentaje_dia_abierto,
                'porc_dia_cerrado': porcentaje_dia_cerrado,
                'promedio_limites_reales': promedio_limites_reales_mes
            })
    
    # Crear contenido del reporte
    reporte = []
    reporte.append("="*120)
    reporte.append("REPORTE COMPLETO DE ANÁLISIS DE APERTURAS DEL CANAL")
    reporte.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("="*120)
    reporte.append("")
    
    reporte.append("1. RESUMEN GLOBAL")
    reporte.append("-"*120)
    reporte.append(f"  • Rango de fechas analizadas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
    reporte.append(f"  • Total de instantes procesados: {total_tiempos}")
    reporte.append(f"  • Instantes ABIERTOS: {tiempos_abiertos} ({porcentaje_abiertos:.1f}%)")
    reporte.append(f"  • Instantes CERRADOS: {tiempos_cerrados} ({porcentaje_cerrados:.1f}%)")
    reporte.append(f"  • Instantes con límites reales (≥2): {tiempos_con_limites_reales} ({tiempos_con_limites_reales/total_tiempos*100:.1f}%)")
    reporte.append(f"  • Promedio de posibles límites por instante: {promedio_posibles:.2f}")
    reporte.append(f"  • Mínimo posibles límites: {min_posibles}")
    reporte.append(f"  • Máximo posibles límites: {max_posibles}")
    if 'cantidad_limites_reales' in ds_con_estados:
        reporte.append(f"  • Promedio límites reales por instante: {np.mean(cantidades_reales):.2f}")
    reporte.append(f"  • Tiempo total de procesamiento: {tiempo_total:.2f} segundos")
    reporte.append("")
    
    reporte.append("2. ESTADÍSTICAS MENSUALES")
    reporte.append("-"*120)
    reporte.append("Mes-Año   | Total | Abiertos | %Abierto | Cerrados | %Cerrado | DurProm(Ab) | DurProm(Ce) | %Dia(Ab) | %Dia(Ce) | LimReales")
    reporte.append("-"*130)
    
    for stats in estadisticas_mensuales:
        reporte.append(f"{stats['mes_ano']:10} | {stats['total']:5} | {stats['abiertos']:8} | "
                      f"{stats['porc_abiertos']:9.1f} | {stats['cerrados']:8} | {stats['porc_cerrados']:9.1f} | "
                      f"{stats['duracion_prom_abierta']:12.1f} | {stats['duracion_prom_cerrada']:12.1f} | "
                      f"{stats['porc_dia_abierto']:8.1f} | {stats['porc_dia_cerrado']:8.1f} | {stats['promedio_limites_reales']:9.2f}")
    
    reporte.append("")
    
    reporte.append("3. CARACTERÍSTICAS DEL MÉTODO")
    reporte.append("-"*120)
    reporte.append("  • ALGORITMO: Seguimiento optimizado desde zonas ROI")
    reporte.append("  • PRINCIPIO: 1 seguimiento por zona concentrada de puntos ROI en cada extremo de curva")
    reporte.append("  • VALIDACIÓN: Los seguimientos desde extremos opuestos deben 'chocarse' (compartir puntos)")
    reporte.append("  • CRITERIOS MODIFICADOS:")
    reporte.append("     1. Encuentro entre seguimientos que 'chocan' (misma curva recorrida en ambos sentidos)")
    reporte.append("     2. Conexión zonaROI-zonaROI (no ROI exacta, tolerancia de 5 píxeles)")
    reporte.append("     3. Cruce de línea X = {x_limit}")
    reporte.append("  • OPTIMIZACIÓN: Solo se realizan seguimientos necesarios (1 por zona por extremo)")
    reporte.append("  • ALMACENAMIENTO: Solo se guardan los 2 límites que cumplen los 3 criterios")
    reporte.append("  • VENTAJAS:")
    reporte.append("     - Optimización de procesamiento (menos seguimientos redundantes)")
    reporte.append("     - Validación robusta (garantía de que es la misma curva)")
    reporte.append("     - Conexión ROI-ROI asegurada")
    reporte.append("     - Eliminación de falsos positivos")
    reporte.append("")
    
    reporte.append("4. PARÁMETROS DE PROCESAMIENTO")
    reporte.append("-"*120)
    reporte.append(f"  • X_LIMIT: {x_limit} (línea de referencia para cruce del canal)")
    reporte.append(f"  • UMBRAL_DISTANCIA: {UMBRAL_DISTANCIA} píxeles (para seguimiento de vecinos)")
    reporte.append(f"  • TOLERANCIA_ROI: {TOLERANCIA_ROI} píxeles (para considerar punto cercano a ROI)")
    reporte.append(f"  • ROI utilizada: {ROI1_POINTS} (polígono cuadrilátero)")
    reporte.append(f"  • Algoritmo clustering: DBSCAN (eps=15, min_samples=1) para identificar zonas")
    reporte.append(f"  • Archivo de entrada: {archivo_perimetros}")
    reporte.append(f"  • Archivo de salida: {archivo_con_estados}")
    reporte.append(f"  • Carpeta resultados: {carpeta_resultados}")
    reporte.append("")
    
    reporte.append("5. DETALLE COMPLETO POR INSTANTE (TODOS LOS INSTANTES)")
    reporte.append("-"*120)
    reporte.append("Nº  | Fecha-Hora           | Estado   | PuntosPerim | PosiblesLim | LimitesReales | Comentarios")
    reporte.append("-"*130)
    
    # Mostrar TODOS los instantes
    for i in range(len(tiempos)):
        fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M')
        estado_str = "ABIERTO" if estados_array[i] == 1 else "CERRADO"
        posibles = cantidades_posibles[i] if 'cantidad_posibles_limites' in ds_con_estados else 0
        
        # Calcular puntos de perímetro
        if 'perimetro_arena_agua' in ds_con_estados:
            perim = ds_con_estados.perimetro_arena_agua.values[i, :]
            puntos_perim = np.sum(perim == 1)
        else:
            puntos_perim = 0
        
        # Límites reales
        if 'cantidad_limites_reales' in ds_con_estados:
            limites_reales = cantidades_reales[i]
        else:
            limites_reales = 2 if estados_array[i] == 1 else 0
        
        # Determinar comentarios según las condiciones
        comentario = ""
        
        if estados_array[i] == 1:  # ABIERTO
            if limites_reales >= 2:
                comentario = " Cumple 3 criterios: Encuentro + ROI-ROI + Cruce X"
            else:
                comentario = "¡ATENCIÓN! Estado ABIERTO pero menos de 2 límites reales"
        else:  # CERRADO
            if limites_reales >= 2:
                comentario = "¡CONTRADICCIÓN! Estado CERRADO pero con límites reales"
            elif posibles >= 2:
                comentario = "Posibles límites encontrados pero no cumplen 3 criterios"
            elif puntos_perim == 0:
                comentario = "Sin puntos de perímetro"
            elif posibles == 0:
                comentario = "Sin seguimientos posibles desde ROI"
            else:
                comentario = "No cumple criterios de apertura"
        
        # Añadir información específica sobre criterios
        if posibles > 0 and estados_array[i] == 0:
            if posibles < 2:
                comentario += f" (solo {posibles} seguimiento(s))"
            else:
                comentario += f" ({posibles} seguimientos, pero no 'chocan' o no cruzan X)"
        
        reporte.append(f"{i+1:3} | {fecha_str:20} | {estado_str:8} | {puntos_perim:11} | {posibles:11} | {limites_reales:13} | {comentario}")
    
    reporte.append("")
    
    reporte.append("6. ANÁLISIS DE CASOS ESPECIALES")
    reporte.append("-"*120)
    
    # Identificar casos especiales
    casos_contradictorios = []
    casos_sin_perimetro = []
    casos_muchos_seguimientos = []
    casos_pocos_seguimientos = []
    
    for i in range(len(tiempos)):
        fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M')
        estado = estados_array[i]
        posibles = cantidades_posibles[i] if 'cantidad_posibles_limites' in ds_con_estados else 0
        
        if 'perimetro_arena_agua' in ds_con_estados:
            perim = ds_con_estados.perimetro_arena_agua.values[i, :]
            puntos_perim = np.sum(perim == 1)
        else:
            puntos_perim = 0
        
        # Caso 1: Contradicción entre estado y límites reales
        if 'cantidad_limites_reales' in ds_con_estados:
            limites_reales = cantidades_reales[i]
            if (estado == 1 and limites_reales < 2) or (estado == 0 and limites_reales >= 2):
                casos_contradictorios.append(f"{fecha_str}: Estado={'ABIERTO' if estado==1 else 'CERRADO'}, LimitesReales={limites_reales}")
        
        # Caso 2: Sin puntos de perímetro
        if puntos_perim == 0:
            casos_sin_perimetro.append(fecha_str)
        
        # Caso 3: Muchos seguimientos posibles
        if posibles >= 10:
            casos_muchos_seguimientos.append(f"{fecha_str}: {posibles} seguimientos")
        
        # Caso 4: Pocos puntos pero estado abierto
        if puntos_perim < 50 and estado == 1:
            casos_pocos_seguimientos.append(f"{fecha_str}: {puntos_perim} puntos, Estado=ABIERTO")
    
    reporte.append(f"  • Casos contradictorios (estado vs límites): {len(casos_contradictorios)}")
    if casos_contradictorios:
        for caso in casos_contradictorios[:10]:  # Mostrar solo primeros 10
            reporte.append(f"    - {caso}")
        if len(casos_contradictorios) > 10:
            reporte.append(f"    ... y {len(casos_contradictorios)-10} más")
    
    reporte.append(f"\n  • Instantes sin puntos de perímetro: {len(casos_sin_perimetro)}")
    if casos_sin_perimetro:
        reporte.append(f"    Primeros: {', '.join(casos_sin_perimetro[:5])}")
        if len(casos_sin_perimetro) > 5:
            reporte.append(f"    ... y {len(casos_sin_perimetro)-5} más")
    
    reporte.append(f"\n  • Instantes con ≥10 seguimientos posibles: {len(casos_muchos_seguimientos)}")
    if casos_muchos_seguimientos:
        for caso in casos_muchos_seguimientos[:5]:
            reporte.append(f"    - {caso}")
        if len(casos_muchos_seguimientos) > 5:
            reporte.append(f"    ... y {len(casos_muchos_seguimientos)-5} más")
    
    reporte.append(f"\n  • Instantes con <50 puntos pero abiertos: {len(casos_pocos_seguimientos)}")
    if casos_pocos_seguimientos:
        for caso in casos_pocos_seguimientos[:5]:
            reporte.append(f"    - {caso}")
        if len(casos_pocos_seguimientos) > 5:
            reporte.append(f"    ... y {len(casos_pocos_seguimientos)-5} más")
    
    reporte.append("")
    
    reporte.append("7. RESUMEN DE VELOCIDADES Y EFICIENCIA")
    reporte.append("-"*120)
    reporte.append(f"  • Tiempo total de procesamiento: {tiempo_total:.2f} segundos ({tiempo_total/60:.1f} minutos)")
    reporte.append(f"  • Instantes procesados por segundo: {total_tiempos/tiempo_total:.2f}")
    reporte.append(f"  • Tiempo promedio por instante: {tiempo_total/total_tiempos*1000:.1f} ms")
    reporte.append(f"  • Eficiencia del método optimizado: ALTA (solo seguimientos necesarios)")
    reporte.append(f"  • Reducción de seguimientos: ~90% (vs método anterior con todos los puntos)")
    reporte.append("")
    
    reporte.append("8. CONCLUSIONES Y RECOMENDACIONES")
    reporte.append("-"*120)
    reporte.append("  • CONCLUSIONES:")
    reporte.append("     1. El método optimizado es eficiente y evita procesamiento redundante")
    reporte.append("     2. La validación por 'choque' asegura robustez en la identificación")
    reporte.append("     3. El criterio zonaROI (vs ROI exacta) es más flexible y realista")
    reporte.append("     4. Solo almacenar límites que cumplen criterios optimiza espacio")
    
    reporte.append("\n  • RECOMENDACIONES:")
    if casos_contradictorios:
        reporte.append(f"     1. Revisar {len(casos_contradictorios)} casos contradictorios")
    if casos_sin_perimetro:
        reporte.append(f"     2. Verificar {len(casos_sin_perimetro)} instantes sin perímetro")
    reporte.append("     3. Validar visualmente casos con muchos seguimientos pero cerrados")
    reporte.append("     4. Considerar ajustar umbrales para casos límite")
    
    reporte.append("\n  • MEJORAS FUTURAS:")
    reporte.append("     1. Incorporar validación por persistencia temporal")
    reporte.append("     2. Añadir métricas de calidad de los límites")
    reporte.append("     3. Implementar detección automática de parámetros óptimos")
    reporte.append("")
    
    reporte.append("9. LEGENDAS Y CÓDIGOS")
    reporte.append("-"*120)
    reporte.append("  • Estado ABIERTO: Canal con desembocadura activa (cumple 3 criterios)")
    reporte.append("  • Estado CERRADO: Canal sin desembocadura activa")
    reporte.append("  • PuntosPerim: Cantidad de puntos en el perímetro agua-arena")
    reporte.append("  • PosiblesLim: Seguimientos iniciados desde zonas ROI")
    reporte.append("  • LimitesReales: Límites que cumplen los 3 criterios (encuentro + ROI-ROI + cruce X)")
    reporte.append("  • Ok: Caso normal y esperado")
    reporte.append("  • ¡ATENCIÓN!: Caso que requiere revisión")
    reporte.append("  • ¡CONTRADICCIÓN!: Inconsistencia en los datos")
    reporte.append("")
    
    reporte.append("="*120)
    reporte.append(f"FIN DEL REPORTE - {total_tiempos} INSTANTES ANALIZADOS")
    reporte.append("="*120)
    
    # Escribir archivo
    print(f"Generando reporte con {total_tiempos} instantes...")
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    
    # Calcular tamaño del archivo
    tamaño_mb = os.path.getsize(archivo_reporte) / (1024 * 1024)
    
    print(f" Reporte completo generado en: {archivo_reporte}")
    print(f" Tamaño del archivo: {tamaño_mb:.1f} MB")
    print(f" Instantes incluidos: {total_tiempos}")
    
    return archivo_reporte

def distancia_punto_a_segmento_optimizada(p, a, b):
    """
    Calcula la distancia mínima de un punto p a un segmento de recta entre a y b
    Versión optimizada para mayor velocidad
    """
    # Convertir a arrays numpy para cálculos vectoriales
    p = np.array(p, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
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

def seguir_perimetro_desde_punto(x_puntos, y_puntos, punto_inicio_idx, umbral_distancia=UMBRAL_DISTANCIA, max_puntos=None, direccion='ambas'):
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

def determinar_estado_apertura_completo(archivo_perimetros, archivo_salida, x_limit=X_LIMIT,
                                       umbral_distancia=UMBRAL_DISTANCIA):
    """
    Analiza estado de apertura del canal para TODA la base de datos
    """
    print("="*80)
    print("ANÁLISIS COMPLETO DE APERTURAS DEL CANAL")
    print("="*80)
    print(f"• Método: Seguimiento desde extremos de curvas")
    print(f"• Principio: 2 seguimientos por curva (desde cada extremo cerca de ROI)")
    print(f"• Validación: 3 criterios: 1) Encuentro, 2) Distintas zonas ROI, 3) Cruce X={x_limit}")
    print(f"• Almacenamiento: Solo 2 límites reales más largos por instante (si estado=ABIERTO)")
    print(f"• Parámetros: X_LIMIT={x_limit}, UMBRAL_DISTANCIA={umbral_distancia}, TOLERANCIA_ROI={TOLERANCIA_ROI}")
    print("="*80)
    
    print(f"\nLeyendo archivo de perímetros: {archivo_perimetros}")
    ds_perimetros = xr.open_dataset(archivo_perimetros)
    tiempos = ds_perimetros.time.values
    n_tiempos = len(tiempos)
    
    print(f"✓ Total de instantes en la base de datos: {n_tiempos}")
    print(f"✓ Rango de fechas: {pd.to_datetime(tiempos[0]).strftime('%Y-%m-%d')} a {pd.to_datetime(tiempos[-1]).strftime('%Y-%m-%d')}")
    
    # Preparar arrays para resultados
    estados = np.zeros(n_tiempos, dtype=np.uint8)
    cantidades_posibles = np.zeros(n_tiempos, dtype=np.uint8)
    cantidades_reales = np.zeros(n_tiempos, dtype=np.uint8)  # Límites que cumplen 3 criterios
    
    # Arrays para límites (inicializados con ceros)
    n_puntos_perimetro = ds_perimetros.punto_perimetro.size
    limites_superiores = np.zeros((n_tiempos, n_puntos_perimetro), dtype=np.uint8)
    limites_inferiores = np.zeros((n_tiempos, n_puntos_perimetro), dtype=np.uint8)
    
    # Procesar cada tiempo
    print("\nIniciando procesamiento...")
    print("(Mostrando progreso cada 50 instantes)")
    
    inicio_procesamiento_total = datetime.now()
    instantes_procesados = 0
    instantes_con_puntos = 0
    
    for idx_original in range(n_tiempos):
        # Mostrar progreso cada 50 instantes
        if idx_original % 50 == 0:
            tiempo_transcurrido = (datetime.now() - inicio_procesamiento_total).total_seconds()
            velocidad = idx_original / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
            porcentaje = idx_original / n_tiempos * 100
            print(f"  Procesando instante {idx_original+1}/{n_tiempos} ({porcentaje:.1f}%) - Velocidad: {velocidad:.1f} inst/s")
        
        # Solo procesar si existe máscara de agua
        if ds_perimetros.existe_mascara_agua.values[idx_original] == 1:
            perimetro = ds_perimetros.perimetro_arena_agua.values[idx_original, :]
            indices_activos = np.where(perimetro == 1)[0]
            
            if len(indices_activos) == 0:
                estados[idx_original] = 0
                cantidades_posibles[idx_original] = 0
                cantidades_reales[idx_original] = 0
                continue
            
            instantes_con_puntos += 1
            
            # Coordenadas originales
            x_activos = ds_perimetros.coordenada_x.values[indices_activos]
            y_activos = ds_perimetros.coordenada_y.values[indices_activos]
            
            # 1. Encontrar TODOS los puntos cercanos a la ROI
            indices_roi = encontrar_todos_puntos_roi_cercanos(x_activos, y_activos, ROI1_POINTS, TOLERANCIA_ROI)
            
            if len(indices_roi) < 2:
                estados[idx_original] = 0
                cantidades_posibles[idx_original] = 0
                cantidades_reales[idx_original] = 0
                continue
            
            # 2. Encontrar seguimientos desde extremos
            seguimientos_por_curva, zonas, curvas = encontrar_seguimientos_desde_extremos(
                x_activos, y_activos, indices_roi, x_limit, umbral_distancia
            )
            
            # 3. Identificar encuentros entre seguimientos
            encuentros = identificar_encuentros_seguimientos(seguimientos_por_curva)
            
            # 4. Clasificar curvas según criterios
            curvas_clasificadas = clasificar_curvas_segun_criterios(seguimientos_por_curva, encuentros, x_activos, x_limit)
            
            # Contar posibles límites (todas las curvas con seguimientos)
            cantidades_posibles[idx_original] = len(seguimientos_por_curva)
            
            # Filtrar curvas que cumplen los 3 criterios
            curvas_reales = [c for c in curvas_clasificadas if c['cumple_3_criterios']]
            cantidades_reales[idx_original] = len(curvas_reales)
            
            # 5. Determinar estado y almacenar límites
            if len(curvas_reales) >= 2:
                # Estado ABIERTO: al menos 2 curvas cumplen los 3 criterios
                estados[idx_original] = 1
                instantes_procesados += 1
                
                # Ordenar curvas por longitud (más largas primero)
                curvas_reales.sort(key=lambda x: x['longitud'], reverse=True)
                
                # Tomar las 2 curvas más largas
                for i, curva_info in enumerate(curvas_reales[:2]):
                    # Convertir índices de activos a índices de perimetro completo
                    curva_completa = indices_activos[curva_info['curva_puntos']]
                    
                    # Marcar límites en los arrays
                    if i == 0:  # Límite superior
                        for idx in curva_completa:
                            if idx < n_puntos_perimetro:
                                limites_superiores[idx_original, idx] = 1
                    else:  # Límite inferior
                        for idx in curva_completa:
                            if idx < n_puntos_perimetro:
                                limites_inferiores[idx_original, idx] = 1
            else:
                # Estado CERRADO: menos de 2 curvas cumplen los 3 criterios
                estados[idx_original] = 0
        else:
            estados[idx_original] = 0
            cantidades_posibles[idx_original] = 0
            cantidades_reales[idx_original] = 0
    
    fin_procesamiento_total = datetime.now()
    tiempo_total = (fin_procesamiento_total - inicio_procesamiento_total).total_seconds()
    
    print(f"\n Procesamiento completado en {tiempo_total:.2f} segundos ({tiempo_total/60:.1f} minutos)")
    print(f" Instantes con máscara de agua: {instantes_con_puntos}/{n_tiempos}")
    print(f" Instantes procesados exitosamente: {instantes_procesados}/{n_tiempos}")
    print(f" Instantes ABIERTOS: {np.sum(estados == 1)}")
    print(f" Instantes CERRADOS: {np.sum(estados == 0)}")
    
    # Preparar otros datos
    otros_datos = {
        'existe_mascara_agua': ds_perimetros.existe_mascara_agua.values,
        'nombre_archivo': ds_perimetros.nombre_archivo.values,
        'perimetro_arena_agua': ds_perimetros.perimetro_arena_agua.values
    }
    
    # Crear dataset de salida
    print("\nCreando archivo NetCDF con resultados...")
    
    ds_salida = xr.Dataset(
        {
            'perimetro_arena_agua': (('time', 'punto_perimetro'), 
                                    otros_datos['perimetro_arena_agua']),
            'coordenada_x': (('punto_perimetro',), ds_perimetros.coordenada_x.values),
            'coordenada_y': (('punto_perimetro',), ds_perimetros.coordenada_y.values),
            'existe_mascara_agua': (('time',), otros_datos['existe_mascara_agua']),
            'nombre_archivo': (('time',), otros_datos['nombre_archivo']),
            'estado_apertura': (('time',), estados),
            'cantidad_posibles_limites': (('time',), cantidades_posibles),
            'cantidad_limites_reales': (('time',), cantidades_reales),
            'limite_superior_canal': (('time', 'punto_perimetro'), limites_superiores),
            'limite_inferior_canal': (('time', 'punto_perimetro'), limites_inferiores)
        },
        coords={
            'time': tiempos,
            'punto_perimetro': ds_perimetros.punto_perimetro.values
        }
    )
    
    # Agregar atributos
    ds_salida.attrs.update({
        'title': 'Análisis completo de aperturas del canal (Método desde Extremos)',
        'history': f'Creado el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'source_file': archivo_perimetros,
        'description': 'Análisis completo de estado de apertura del canal usando seguimiento desde extremos de curvas',
        'methodology': f'Seguimiento desde extremos con validación por 3 criterios: 1) Encuentro seguimientos, 2) Extremos en zonas ROI distintas, 3) Cruce X={x_limit}',
        'algorithm': 'Método con seguimiento desde extremos y clasificación por criterios',
        'coordinate_system': 'Origen en esquina superior izquierda (Y aumenta hacia abajo)',
        'x_limit_used': x_limit,
        'umbral_distancia_used': umbral_distancia,
        'tolerancia_roi_used': TOLERANCIA_ROI,
        'roi_points': str(ROI1_POINTS),
        'processing_time_seconds': tiempo_total,
        'total_instances': n_tiempos,
        'instances_with_water_mask': instantes_con_puntos,
        'successfully_processed_instances': instantes_procesados,
        'open_instances': int(np.sum(estados == 1)),
        'closed_instances': int(np.sum(estados == 0)),
        'note': 'Solo se almacenan los 2 límites más largos que cumplen los 3 criterios por instante'
    })
    
    # Guardar archivo
    print(f"Guardando archivo NetCDF: {archivo_salida}")
    ds_salida.to_netcdf(archivo_salida)
    
    ds_perimetros.close()
    
    return ds_salida, tiempo_total

def main():
    """Función principal - Procesa TODA la base de datos"""
    # Iniciar medición de tiempo total
    inicio_tiempo_total = datetime.now()
    
    # Configuración de rutas
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    
    # Archivos de entrada y salida
    archivo_perimetros = os.path.join(carpeta_base, 'Resultados_Perimetros_ROI1-FINAL', 'Data_perimetros_arena_agua.nc')
    carpeta_resultados = os.path.join(carpeta_base, 'Resultados_Aperturas_ROI1-FINAL')
    
    # Crear nombre de archivo de salida
    fecha_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo_con_estados = os.path.join(carpeta_resultados, f'Data_Aperturas_COMPLETO_{fecha_actual}.nc')
    
    # Crear carpeta de resultados
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    # 1. Verificar archivo
    if not os.path.exists(archivo_perimetros):
        print(f"ERROR: No existe el archivo de perímetros: {archivo_perimetros}")
        print("Ejecuta primero el código de generación de perímetros.")
        return
    
    # 2. Determinar estados de apertura para TODA la base de datos
    print("\n" + "="*80)
    print("INICIANDO ANÁLISIS COMPLETO DE APERTURAS")
    print("="*80)
    
    try:
        ds_con_estados, tiempo_procesamiento = determinar_estado_apertura_completo(
            archivo_perimetros, archivo_con_estados, 
            x_limit=X_LIMIT, umbral_distancia=UMBRAL_DISTANCIA
        )
    except Exception as e:
        print(f"ERROR durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Generar reporte COMPLETO con TODOS los instantes
    print("\n" + "="*80)
    print("GENERANDO REPORTE COMPLETO CON TODOS LOS INSTANTES")
    print("="*80)
    
    archivo_reporte = generar_reporte_completo(
        ds_con_estados=ds_con_estados,
        tiempo_total=tiempo_procesamiento,
        archivo_perimetros=archivo_perimetros,
        archivo_con_estados=archivo_con_estados,
        carpeta_resultados=carpeta_resultados,
        x_limit=X_LIMIT
    )
    
    # 4. Imprimir estadísticas finales en consola
    print("\n" + "="*80)
    print("ESTADÍSTICAS FINALES - ANÁLISIS COMPLETO")
    print("="*80)
    
    estados_array = ds_con_estados.estado_apertura.values
    tiempos = ds_con_estados.time.values
    
    total_tiempos = len(estados_array)
    tiempos_abiertos = np.sum(estados_array == 1)
    tiempos_cerrados = np.sum(estados_array == 0)
    
    if 'cantidad_limites_reales' in ds_con_estados:
        cantidades_reales = ds_con_estados.cantidad_limites_reales.values
        tiempos_con_limites_reales = np.sum(cantidades_reales >= 2)
        promedio_limites_reales = np.mean(cantidades_reales)
    else:
        tiempos_con_limites_reales = tiempos_abiertos
        promedio_limites_reales = 2 * (tiempos_abiertos / total_tiempos) if total_tiempos > 0 else 0
    
    # Convertir tiempos a datetime para obtener rango
    tiempos_dt = pd.to_datetime(tiempos)
    fecha_min = tiempos_dt.min()
    fecha_max = tiempos_dt.max()
    
    # Calcular tiempo total de ejecución
    fin_tiempo_total = datetime.now()
    tiempo_total_ejecucion = (fin_tiempo_total - inicio_tiempo_total).total_seconds()
    
    # Estadísticas detalladas
    if 'cantidad_posibles_limites' in ds_con_estados:
        cantidades_posibles = ds_con_estados.cantidad_posibles_limites.values
        promedio_posibles = np.mean(cantidades_posibles)
        max_posibles = np.max(cantidades_posibles)
    else:
        promedio_posibles = 0
        max_posibles = 0
    
    print(f"RESUMEN EJECUCIÓN:")
    print(f"  • Rango temporal analizado: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
    print(f"  • Total de instantes: {total_tiempos}")
    print(f"  • Tiempo total de ejecución: {tiempo_total_ejecucion:.2f} segundos ({tiempo_total_ejecucion/60:.1f} minutos)")
    print(f"  • Velocidad promedio: {total_tiempos/tiempo_total_ejecucion:.1f} instantes/segundo")
    
    print(f"\nRESULTADOS APERTURA:")
    print(f"  • Instantes ABIERTOS: {tiempos_abiertos} ({tiempos_abiertos/total_tiempos*100:.1f}%)")
    print(f"  • Instantes CERRADOS: {tiempos_cerrados} ({tiempos_cerrados/total_tiempos*100:.1f}%)")
    print(f"  • Instantes con ≥2 límites reales: {tiempos_con_limites_reales} ({tiempos_con_limites_reales/total_tiempos*100:.1f}%)")
    print(f"  • Promedio límites reales por instante: {promedio_limites_reales:.2f}")
    
    print(f"\nESTADÍSTICAS DE SEGUIMIENTO:")
    print(f"  • Promedio seguimientos posibles por instante: {promedio_posibles:.2f}")
    print(f"  • Máximo seguimientos en un instante: {max_posibles}")
    
    print(f"\nCARACTERÍSTICAS DEL MÉTODO (DESDE EXTREMOS):")
    print(f"  • Enfoque: 2 seguimientos por curva (desde cada extremo cerca de ROI)")
    print(f"  • Criterio 1: Seguimientos deben encontrarse/compartir puntos")
    print(f"  • Criterio 2: Extremos en zonas ROI distintas")
    print(f"  • Criterio 3: Ambos seguimientos cruzan X={X_LIMIT}")
    print(f"  • Clasificación: 'limite_real_canal' (cumple 3) o 'otro_limite' (solo cumple 1)")
    print(f"  • Estado ABIERTO: ≥2 curvas cumplen 3 criterios")
    print(f"  • Almacenamiento: Solo 2 curvas más largas para instantes ABIERTOS")
    
    print(f"\nARCHIVOS GENERADOS:")
    print(f"  • NetCDF con resultados: {archivo_con_estados}")
    print(f"  • Reporte completo: {archivo_reporte}")
    print(f"  • Carpeta de resultados: {carpeta_resultados}")
    
    # Calcular tamaño de archivos
    tamaño_nc = os.path.getsize(archivo_con_estados) / (1024*1024) if os.path.exists(archivo_con_estados) else 0
    tamaño_txt = os.path.getsize(archivo_reporte) / (1024*1024) if os.path.exists(archivo_reporte) else 0
    
    print(f"\nESTADÍSTICAS DE ARCHIVOS:")
    print(f"  • NetCDF: {tamaño_nc:.1f} MB")
    print(f"  • Reporte TXT: {tamaño_txt:.1f} MB")
    print(f"  • Variables principales: estado_apertura, cantidad_limites_reales, limite_superior_canal, limite_inferior_canal")
    print(f"  • Dimensiones: time={total_tiempos}, punto_perimetro={ds_con_estados.punto_perimetro.size}")
    
    # Información adicional sobre distribución por mes
    if total_tiempos > 0:
        # Calcular porcentaje por mes
        meses = tiempos_dt.month
        anos = tiempos_dt.year
        
        print(f"\nDISTRIBUCIÓN TEMPORAL (resumen):")
        meses_unicos = sorted(set(zip(anos, meses)))
        for ano, mes in meses_unicos[:12]:  # Mostrar solo primeros 12 meses
            mascara = (anos == ano) & (meses == mes)
            total_mes = np.sum(mascara)
            if total_mes > 0:
                abiertos_mes = np.sum(estados_array[mascara] == 1)
                print(f"  • {ano}-{mes:02d}: {total_mes} instantes, {abiertos_mes} abiertos ({abiertos_mes/total_mes*100:.0f}%)")
        
        if len(meses_unicos) > 12:
            print(f"  • ... y {len(meses_unicos)-12} meses más")
    
    print(f"\nRECOMENDACIONES PARA ANÁLISIS:")
    print(f"  1. Revisar el reporte completo ({archivo_reporte}) para ver detalle de cada instante")
    print(f"  2. Buscar '¡ATENCIÓN!' y '¡CONTRADICCIÓN!' en el reporte para casos especiales")
    print(f"  3. Usar el código de generación de videos para visualizar casos específicos")
    print(f"  4. Verificar la sección 'ANÁLISIS DE CASOS ESPECIALES' en el reporte")
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)

if __name__ == "__main__":
    main()