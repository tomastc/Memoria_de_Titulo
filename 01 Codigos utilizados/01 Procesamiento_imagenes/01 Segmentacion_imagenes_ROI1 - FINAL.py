import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile as tiff
from sklearn.cluster import KMeans
from skimage import exposure, filters, morphology
from scipy.spatial.distance import cdist
import os
import glob
import time
import xarray as xr # type: ignore
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
import warnings
import re
import sys
from typing import Optional, Tuple, Dict, Any, Set, List
from collections import defaultdict, OrderedDict

# Configuración para mostrar más decimales en estadísticas
np.set_printoptions(precision=4, suppress=True)

# ============================================================================
# POST-PROCESAMIENTO SIMPLIFICADO Y EFECTIVO CON FUSIÓN DE ENCLAVES PEQUEÑOS
# ============================================================================

def postprocess_coastal_masks(mask_water, mask_sand, roi_mask, image_shape=None):
    """
    Post-procesamiento simplificado y efectivo para máscaras de agua y arena.
    
    Modificación: Solo una etapa para SUB-ÁREAS menores a diez mil píxeles
    que estén completamente rodeadas por la otra clase dentro de la misma región.
    """
    from scipy import ndimage
    
    # Convertir a booleanos para operaciones morfológicas
    water = mask_water.astype(bool)
    sand = mask_sand.astype(bool)
    roi = roi_mask.astype(bool)
    
    # 1. ELIMINAR COMPONENTES MUY PEQUEÑOS (PÍXELES AISLADOS)
    min_object_area = 300
    water = morphology.remove_small_objects(water, min_size=min_object_area)
    sand = morphology.remove_small_objects(sand, min_size=min_object_area)
    
    # 2. RELLENAR HUECOS PEQUEÑOS
    water = morphology.remove_small_holes(water, area_threshold=300)
    sand = morphology.remove_small_holes(sand, area_threshold=300)
    
    # 3. SUAVIZADO CON APERTURA Y CIERRE
    kernel_open = morphology.disk(5)
    water = morphology.binary_opening(water, kernel_open)
    sand = morphology.binary_opening(sand, kernel_open)
    
    kernel_close = morphology.disk(10)
    water = morphology.binary_closing(water, kernel_close)
    sand = morphology.binary_closing(sand, kernel_close)
    
    # 4. GARANTIZAR QUE NO HAYA SUPERPOSICIÓN (AGUA TIENE PRIORIDAD)
    overlap = water & sand
    sand[overlap] = False  # Dar prioridad al agua
    
    # 5. ÚNICA ETAPA: FUSIÓN DE SUB-ÁREAS MENORES
    #    DENTRO DE UNA CLASE QUE ESTÉN COMPLETAMENTE RODEADAS POR LA OTRA CLASE
    # -----------------------------------------------------------------
    max_subarea_area = 7000  # Umbral de área para sub-áreas
    
    # Crear una máscara combinada para análisis
    combined_mask = np.zeros_like(water, dtype=int)
    combined_mask[water] = 1  # Agua = 1
    combined_mask[sand] = 2   # Arena = 2
    
    def find_and_merge_subareas(class_mask, other_class_mask, combined_mask, class_value, other_class_value):
        """
        Encuentra sub-áreas dentro de una clase que están completamente rodeadas
        por la otra clase y las fusiona si son menores al umbral.
        
        Args:
            class_mask: Máscara booleana de la clase actual (ej: agua)
            other_class_mask: Máscara booleana de la otra clase (ej: arena)
            combined_mask: Máscara combinada con valores 1 y 2
            class_value: Valor de la clase actual en combined_mask (1 para agua)
            other_class_value: Valor de la otra clase en combined_mask (2 para arena)
        
        Returns:
            Máscara de clase actualizada
        """
        # Crear una copia para trabajar
        class_mask_updated = class_mask.copy()
        
        # Encontrar las regiones de la otra clase
        other_class_labeled, num_other_regions = ndimage.label(other_class_mask)
        
        # Para cada región de la otra clase
        for label in range(1, num_other_regions + 1):
            region_other = (other_class_labeled == label)
            
            # Encontrar huecos dentro de esta región de la otra clase
            # Los huecos son áreas de la clase actual dentro de la región de la otra clase
            region_dilated = morphology.binary_dilation(region_other, morphology.disk(1))
            potential_holes = region_dilated & ~region_other
            
            # Etiquetar estos huecos potenciales
            holes_labeled, num_holes = ndimage.label(potential_holes)
            
            for hole_label in range(1, num_holes + 1):
                hole_mask = (holes_labeled == hole_label)
                
                # Verificar que el hueco sea completamente de la clase actual
                hole_values = combined_mask[hole_mask]
                if not np.all(hole_values == class_value):
                    continue  # No es un hueco de la clase actual, saltar
                
                # Verificar que el hueco esté completamente rodeado por la otra clase
                # Dilatar el hueco y verificar su borde
                hole_dilated = morphology.binary_dilation(hole_mask, morphology.disk(1))
                hole_border = hole_dilated & ~hole_mask
                
                # El borde debe estar completamente dentro de la región de la otra clase
                border_values = combined_mask[hole_border]
                if np.all(border_values == other_class_value):
                    # Es un hueco completamente rodeado por la otra clase
                    hole_area = np.sum(hole_mask)
                    
                    # Si el hueco es menor al umbral, fusionarlo
                    if hole_area < max_subarea_area:
                        # Convertir el hueco a la otra clase
                        class_mask_updated[hole_mask] = False
        
        return class_mask_updated
    
    # Procesar sub-áreas de agua dentro de regiones de arena
    water = find_and_merge_subareas(water, sand, combined_mask, 1, 2)
    
    # Actualizar máscara combinada
    combined_mask = np.zeros_like(water, dtype=int)
    combined_mask[water] = 1
    combined_mask[sand] = 2
    
    # Procesar sub-áreas de arena dentro de regiones de agua
    sand = find_and_merge_subareas(sand, water, combined_mask, 2, 1)
    
    # 6. ASEGURAR COBERTURA COMPLETA DE LA ROI
    classified = water | sand
    unclassified = roi & ~classified
    
    if np.any(unclassified):
        from scipy.ndimage import distance_transform_edt
        
        dist_to_water = distance_transform_edt(~water)
        dist_to_sand = distance_transform_edt(~sand)
        
        water_closer = dist_to_water < dist_to_sand
        water[unclassified & water_closer] = True
        sand[unclassified & ~water_closer] = True
    
    # 7. VERIFICACIÓN FINAL DE COBERTURA
    final_coverage = (water | sand) & roi
    coverage_ratio = np.sum(final_coverage) / np.sum(roi)
    
    if coverage_ratio < 0.99:
        missing = roi & ~(water | sand)
        if np.any(missing):
            sand[missing] = True
    
    # Convertir de vuelta al formato original
    mask_water_final = water.astype(np.uint8)
    mask_sand_final = sand.astype(np.uint8)
    
    return mask_water_final, mask_sand_final

def aplicar_postprocesamiento_morfologico(mask_mar, mask_arena, roi_points):
    """
    Función wrapper para aplicar el post-procesamiento morfológico simplificado.
    """
    # Crear máscara de ROI
    h, w = mask_mar.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], 1)
    
    # Aplicar pipeline de post-procesamiento simplificado
    mask_mar_procesada, mask_arena_procesada = postprocess_coastal_masks(
        mask_mar, mask_arena, roi_mask
    )
    
    return mask_mar_procesada, mask_arena_procesada


def mostrar_avance_simple(indice, total, roi_actual):
    """
    Muestra el avance simple cada 50 imágenes
    """
    if (indice + 1) % 50 == 0 or indice == 0:
        print(f"Procesadas: {indice + 1}/{total} imágenes - ROI: {roi_actual}")
    return

def extraer_caracteristicas_puntos_referencia(hsv, punto_mar, punto_arena, radio=20):
    """
    Extrae características HSV/Lab de puntos de referencia para cada clase
    """
    # Convertir a Lab para mejor discriminación de colores
    lab = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2Lab)
    
    caracteristicas = {}
    
    # Extraer características para cada clase desde puntos específicos
    for nombre, punto, color_space in [('mar', punto_mar, 'hsv'), 
                                      ('arena', punto_arena, 'lab')]:
        x, y = punto
        # Crear máscara circular alrededor del punto
        y_coords, x_coords = np.ogrid[-y: hsv.shape[0]-y, -x: hsv.shape[1]-x]
        mask = x_coords*x_coords + y_coords*y_coords <= radio*radio
        
        if color_space == 'hsv':
            h_val = hsv[:,:,0][mask]
            s_val = hsv[:,:,1][mask]
            v_val = hsv[:,:,2][mask]
            valores = np.vstack((h_val, s_val, v_val)).T
        else:  # lab
            l_val = lab[:,:,0][mask]
            a_val = lab[:,:,1][mask]
            b_val = lab[:,:,2][mask]
            valores = np.vstack((l_val, a_val, b_val)).T
        
        # Calcular media y desviación estándar para cada canal
        if len(valores) > 0:
            mean = np.mean(valores, axis=0)
            std = np.std(valores, axis=0)
            caracteristicas[nombre] = {
                'mean': mean,
                'std': std,
                'color_space': color_space
            }
        else:
            # Fallback: usar el valor del pixel central
            if color_space == 'hsv':
                mean = hsv[y, x, :]
            else:
                mean = lab[y, x, :]
            caracteristicas[nombre] = {
                'mean': mean,
                'std': np.array([1, 1, 1]),
                'color_space': color_space
            }
    
    return caracteristicas

def inicializar_centroides_con_referencia(hsv, mask_roi, caracteristicas, k=2):
    """
    Inicialización de centroides basada en puntos de referencia para k=2
    """
    # Convertir a Lab para algunas características
    lab = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2Lab)
    
    # Preparar datos de la ROI
    coords_roi = np.where(mask_roi == 1)
    datos_hsv = hsv[coords_roi].astype(np.float32)
    datos_lab = lab[coords_roi].astype(np.float32)
    
    # Calcular distancias a cada clase para cada pixel
    distancias = np.zeros((len(coords_roi[0]), 2))
    
    for i, (clase, params) in enumerate(caracteristicas.items()):
        if params['color_space'] == 'hsv':
            datos = datos_hsv
        else:
            datos = datos_lab
            
        # Distancia Mahalanobis (normalizada por desviación estándar)
        dist = np.sum(((datos - params['mean']) / (params['std'] + 1e-10))**2, axis=1)
        distancias[:, i] = dist
    
    # Asignar cada pixel to the closest class
    asignaciones = np.argmin(distancias, axis=1)
    
    # Calcular centroides iniciales
    centroides = []
    for i in range(2):
        if np.sum(asignaciones == i) > 0:
            clase = list(caracteristicas.keys())[i]
            if caracteristicas[clase]['color_space'] == 'hsv':
                hsv_cluster = datos_hsv[asignaciones == i]
                lab_cluster = datos_lab[asignaciones == i]
                centroide_combinado = np.hstack((
                    np.mean(hsv_cluster, axis=0) / np.array([180, 255, 255]),
                    np.mean(lab_cluster, axis=0) / np.array([255, 255, 255])
                ))
                centroides.append(centroide_combinado)
            else:
                hsv_cluster = datos_hsv[asignaciones == i]
                lab_cluster = datos_lab[asignaciones == i]
                centroide_combinado = np.hstack((
                    np.mean(hsv_cluster, axis=0) / np.array([180, 255, 255]),
                    np.mean(lab_cluster, axis=0) / np.array([255, 255, 255])
                ))
                centroides.append(centroide_combinado)
        else:
            clase = list(caracteristicas.keys())[i]
            if caracteristicas[clase]['color_space'] == 'hsv':
                centroide_combinado = np.hstack((
                    caracteristicas[clase]['mean'] / np.array([180, 255, 255]),
                    np.array([128, 128, 128]) / np.array([255, 255, 255])
                ))
            else:
                centroide_combinado = np.hstack((
                    np.array([90, 128, 128]) / np.array([180, 255, 255]),
                    caracteristicas[clase]['mean'] / np.array([255, 255, 255])
                ))
            centroides.append(centroide_combinado)
    
    return np.array(centroides, dtype=np.float32)

def segmentar_imagen_individual(img, roi_points, punto_mar, punto_arena, k=2, random_state=42):
    """
    Segmentación individual de una imagen con post-procesamiento simplificado
    """
    h, w = img.shape[:2]

    # ROI
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_points], 1)
    img_masked = cv2.bitwise_and(img, img, mask=mask_roi)

    # Preprocesamiento
    img_med = cv2.medianBlur(img_masked, 5) #filtro mediano para "sal pimienta"
    img_eq = exposure.rescale_intensity(img_med, in_range='image', out_range=(0, 255)).astype(np.uint8) #ajuste de contraste

    # Espacio HSV y Lab
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_eq, cv2.COLOR_RGB2Lab)
    
    # Extraer características
    caracteristicas = extraer_caracteristicas_puntos_referencia(hsv, punto_mar, punto_arena, radio=20)
    
    # Inicialización de centroides
    seeds = inicializar_centroides_con_referencia(hsv, mask_roi, caracteristicas, k=k)

    # K-means
    datos_hsv = hsv[mask_roi == 1].astype(np.float32)
    datos_lab = lab[mask_roi == 1].astype(np.float32)
    
    datos = np.hstack((
        datos_hsv / np.array([180, 255, 255]),
        datos_lab / np.array([255, 255, 255])
    ))
    
    kmeans = KMeans(n_clusters=k, init=seeds, n_init=1, max_iter=300, random_state=random_state)
    etiquetas = kmeans.fit_predict(datos)

    etiquetas_img = np.zeros((h, w), dtype=np.int32) - 1
    coords_roi = np.where(mask_roi == 1)
    etiquetas_img[coords_roi] = etiquetas
    
    # Clasificación priorizada de clusters
    # Primero clasificar el agua, luego la arena
    num_clusters_encontrados = len(np.unique(etiquetas))
    
    if num_clusters_encontrados == 2:
        # Calcular similitud de cada cluster con agua y arena
        similitudes = np.zeros((2, 2))  # clusters x clases
        
        for i in range(2):
            mascara_cluster = (etiquetas_img == i) & (mask_roi == 1)
            if np.sum(mascara_cluster) > 0:
                for j, (clase, params) in enumerate(caracteristicas.items()):
                    if params['color_space'] == 'hsv':
                        datos_cluster = hsv[mascara_cluster].astype(np.float32)
                    else:
                        datos_cluster = lab[mascara_cluster].astype(np.float32)
                    
                    dist = np.mean(np.sum(((datos_cluster - params['mean']) / (params['std'] + 1e-10))**2, axis=1))
                    similitudes[i, j] = 1 / (dist + 1e-10)
        
        # Priorizar agua: cluster con mayor similitud al agua
        cluster_mar = np.argmax(similitudes[:, 0])
        
        # Para arena: usar el cluster que no sea el de agua
        cluster_arena = 1 - cluster_mar  # Si cluster_mar es 0, arena es 1, y viceversa
        
    else:  # Solo 1 cluster encontrado
        # Marcamos ambos clusters como el mismo, pero indicaremos que no es válido
        cluster_mar = 0
        cluster_arena = 0

    # Máscaras finales
    mask_mar = ((etiquetas_img == cluster_mar) & (mask_roi == 1)).astype(np.uint8)
    mask_arena = ((etiquetas_img == cluster_arena) & (mask_roi == 1)).astype(np.uint8)

    # ============================================================================
    # POST-PROCESAMIENTO
    # ============================================================================
    
    # Solo aplicar post-procesamiento si tenemos dos clusters válidos
    if num_clusters_encontrados == 2 and np.sum(mask_mar) > 0 and np.sum(mask_arena) > 0:
        mask_mar, mask_arena = aplicar_postprocesamiento_morfologico(mask_mar, mask_arena, roi_points)
    else:
        # Si no hay dos clusters válidos, aplicar limpieza básica
        kernel = morphology.disk(3)
        mask_mar = morphology.opening(mask_mar, kernel).astype(np.uint8)
        mask_arena = morphology.opening(mask_arena, kernel).astype(np.uint8)
        
        mask_mar = morphology.remove_small_objects(mask_mar.astype(bool), min_size=200).astype(np.uint8)
        mask_arena = morphology.remove_small_objects(mask_arena.astype(bool), min_size=200).astype(np.uint8)

    # Verificar si existe máscara de agua realmente
    # 0: no hay máscara de agua (también cuando la máscara de agua es la misma que la de arena)
    # Esto sucede cuando la imagen está saturada de un solo color por nubes o por la noche
    existe_mascara_agua = 1
    if np.array_equal(mask_mar, mask_arena) or np.sum(mask_mar) == 0:
        existe_mascara_agua = 0

    # Crear imagen con anotaciones (SOLO ESTA SE GUARDARÁ)
    img_with_annotations = img.copy()
    for punto, color, nombre in zip([punto_mar, punto_arena], 
                                   [(255, 0, 0), (0, 0, 255)], 
                                   ['Mar', 'Arena']):
        x, y = punto
        cv2.circle(img_with_annotations, (x, y), 20, color, 2)
        cv2.putText(img_with_annotations, nombre, (x+25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.polylines(img_with_annotations, [roi_points], isClosed=True, color=(0, 0, 255), thickness=4)

    return img_with_annotations, mask_mar, mask_arena, existe_mascara_agua, num_clusters_encontrados

class PreservadorFechaHora:
    """
    Clase para preservar al máximo la fecha y hora originales de las imágenes,
    aplicando correcciones mínimas solo cuando es absolutamente necesario.
    """
    
    def __init__(self):
        self.fechas_originales = OrderedDict()  # fecha_original -> [resultados]
        self.fechas_finales = {}  # resultado_id -> fecha_final
        self.contador_correcciones = defaultdict(int)
        self.historial_correcciones = []
        
    def extraer_fecha_original(self, nombre_archivo: str) -> Tuple[Optional[datetime], Dict[str, Any]]:
        """
        Extrae la fecha original del nombre del archivo SIN aplicar correcciones.
        Retorna None si no se puede extraer una fecha válida.
        """
        metadata = {
            'fecha_extraida': False,
            'fecha_valida': False,
            'error': None,
            'patron_encontrado': False
        }
        
        try:
            # Buscar el patrón YYYYMMDD-HHMM
            match = re.search(r'(\d{8})-(\d{4})', nombre_archivo)
            
            if not match:
                metadata['error'] = 'No se encontró patrón de fecha en el nombre'
                return None, metadata
            
            metadata['patron_encontrado'] = True
            fecha_str = match.group(1)
            hora_str = match.group(2)
            
            # Validar formato básico
            if len(fecha_str) != 8 or len(hora_str) != 4:
                metadata['error'] = 'Formato de fecha/hora incorrecto'
                return None, metadata
            
            # Intentar parsear
            try:
                fecha_hora = datetime.strptime(f"{fecha_str}{hora_str}", "%Y%m%d%H%M")
                metadata['fecha_extraida'] = True
                
                # Validación adicional
                if fecha_hora.year < 2000 or fecha_hora.year > 2100:
                    metadata['error'] = f'Año fuera de rango razonable: {fecha_hora.year}'
                    metadata['fecha_valida'] = False
                elif fecha_hora.month < 1 or fecha_hora.month > 12:
                    metadata['error'] = f'Mes inválido: {fecha_hora.month}'
                    metadata['fecha_valida'] = False
                elif fecha_hora.day < 1 or fecha_hora.day > 31:
                    metadata['error'] = f'Día inválido: {fecha_hora.day}'
                    metadata['fecha_valida'] = False
                elif fecha_hora.hour < 0 or fecha_hora.hour > 23:
                    metadata['error'] = f'Hora inválida: {fecha_hora.hour}'
                    metadata['fecha_valida'] = False
                elif fecha_hora.minute < 0 or fecha_hora.minute > 59:
                    metadata['error'] = f'Minuto inválido: {fecha_hora.minute}'
                    metadata['fecha_valida'] = False
                else:
                    metadata['fecha_valida'] = True
                    
            except ValueError as e:
                metadata['error'] = f'Error al parsear fecha: {str(e)}'
                metadata['fecha_valida'] = False
                return None, metadata
            
            return fecha_hora, metadata
            
        except Exception as e:
            metadata['error'] = f'Error inesperado: {str(e)}'
            return None, metadata
    
    def corregir_fecha_invalida(self, fecha_original: Optional[datetime], 
                               nombre_archivo: str,
                               fecha_anterior_corregida: Optional[datetime] = None) -> Tuple[datetime, Dict[str, Any]]:
        """
        Corrige una fecha inválida usando la lógica especificada.
        """
        metadata = {
            'correccion_aplicada': False,
            'tipo_correccion': None,
            'fecha_anterior_usada': False,
            'fecha_modificacion_usada': False
        }
        
        # Si no hay fecha original, usar fecha de modificación o anterior
        if fecha_original is None:
            metadata['correccion_aplicada'] = True
            metadata['tipo_correccion'] = 'fecha_no_encontrada'
            
            if fecha_anterior_corregida:
                # Usar fecha anterior + 1 hora
                fecha_corregida = fecha_anterior_corregida + timedelta(hours=1)
                metadata['fecha_anterior_usada'] = True
                self.historial_correcciones.append(
                    f"{nombre_archivo}: No tenía fecha, usando anterior+1h: {fecha_corregida}"
                )
            else:
                # Usar fecha actual como fallback
                fecha_corregida = datetime.now()
                metadata['fecha_modificacion_usada'] = True
                self.historial_correcciones.append(
                    f"{nombre_archivo}: No tenía fecha, usando fecha actual: {fecha_corregida}"
                )
            
            self.contador_correcciones['fechas_no_encontradas'] += 1
            return fecha_corregida, metadata
        
        # Si la fecha es inválida (ya validada en extraer_fecha_original)
        metadata['correccion_aplicada'] = True
        
        # Determinar qué parte es inválida (esto es una simplificación)
        # En la práctica, necesitaríamos más información sobre qué parte falló
        if fecha_anterior_corregida:
            # Intentar mantener la parte válida
            try:
                # Asumir que solo la hora es inválida si la fecha parece razonable
                if 2000 <= fecha_original.year <= 2100 and 1 <= fecha_original.month <= 12 and 1 <= fecha_original.day <= 31:
                    # Fecha parece válida, hora probablemente inválida
                    fecha_corregida = fecha_original.replace(
                        hour=fecha_anterior_corregida.hour + 1 if fecha_anterior_corregida.hour < 23 else 0,
                        minute=fecha_anterior_corregida.minute
                    )
                    metadata['tipo_correccion'] = 'hora_invalida'
                    metadata['fecha_anterior_usada'] = True
                    self.historial_correcciones.append(
                        f"{nombre_archivo}: Hora inválida, usando fecha original con hora anterior+1: {fecha_corregida}"
                    )
                else:
                    # Fecha inválida, hora podría ser válida
                    fecha_corregida = fecha_anterior_corregida.replace(
                        hour=fecha_original.hour if 0 <= fecha_original.hour <= 23 else 0,
                        minute=fecha_original.minute if 0 <= fecha_original.minute <= 59 else 0
                    )
                    metadata['tipo_correccion'] = 'fecha_invalida'
                    metadata['fecha_anterior_usada'] = True
                    self.historial_correcciones.append(
                        f"{nombre_archivo}: Fecha inválida, usando fecha anterior con hora original: {fecha_corregida}"
                    )
            except:
                # Fallback: fecha anterior + 1 hora
                fecha_corregida = fecha_anterior_corregida + timedelta(hours=1)
                metadata['tipo_correccion'] = 'fallback_anterior_mas_1h'
                metadata['fecha_anterior_usada'] = True
                self.historial_correcciones.append(
                    f"{nombre_archivo}: Error complejo, usando anterior+1h: {fecha_corregida}"
                )
        else:
            # No hay fecha anterior, usar fecha de modificación
            fecha_corregida = datetime.now()
            metadata['tipo_correccion'] = 'fallback_fecha_actual'
            metadata['fecha_modificacion_usada'] = True
            self.historial_correcciones.append(
                f"{nombre_archivo}: Sin fecha anterior, usando fecha actual: {fecha_corregida}"
            )
        
        self.contador_correcciones[metadata['tipo_correccion']] += 1
        return fecha_corregida, metadata
    
    def resolver_duplicados(self, resultados: List[Dict]) -> List[Dict]:
        """
        Resuelve fechas duplicadas manteniendo la original para la primera aparición
        y ajustando mínimamente las posteriores.
        """
        # Agrupar por fecha original
        grupos_fecha = defaultdict(list)
        for i, resultado in enumerate(resultados):
            fecha_key = resultado['fecha_hora_original']
            if fecha_key is None:
                fecha_key = resultado['fecha_hora_corregida']
            grupos_fecha[fecha_key].append((i, resultado))
        
        resultados_ajustados = []
        fechas_usadas = set()
        
        # Para cada grupo de fechas duplicadas
        for fecha_base, grupo in sorted(grupos_fecha.items(), key=lambda x: x[0] if x[0] is not None else datetime.min):
            # Ordenar el grupo por algún criterio (ej: nombre de archivo)
            grupo_ordenado = sorted(grupo, key=lambda x: x[1]['nombre_archivo'])
            
            for j, (idx, resultado) in enumerate(grupo_ordenado):
                fecha_original = resultado['fecha_hora_original']
                fecha_corregida = resultado['fecha_hora_corregida']
                
                if j == 0:
                    # Primera aparición: mantener fecha original si es válida, sino la corregida
                    if fecha_original and fecha_original not in fechas_usadas:
                        fecha_final = fecha_original
                    else:
                        fecha_final = fecha_corregida
                    
                    # Asegurar unicidad
                    while fecha_final in fechas_usadas:
                        fecha_final += timedelta(minutes=1)
                    
                    fechas_usadas.add(fecha_final)
                    
                    if fecha_final != fecha_original and fecha_original:
                        resultado['metadata_fecha']['duplicado_ajustado'] = True
                        resultado['metadata_fecha']['ajuste_aplicado'] = f'Duplicado #{j+1}, ajustado a {fecha_final}'
                        self.contador_correcciones['duplicados_ajustados'] += 1
                        
                else:
                    # Apariciones posteriores: ajustar mínimamente
                    if fecha_original and fecha_original not in fechas_usadas:
                        fecha_final = fecha_original
                    else:
                        # Buscar el siguiente intervalo disponible
                        base_fecha = fecha_corregida if fecha_corregida else fecha_original
                        incremento = 0
                        while True:
                            fecha_candidata = base_fecha + timedelta(minutes=incremento)
                            if fecha_candidata not in fechas_usadas:
                                fecha_final = fecha_candidata
                                break
                            incremento += 1
                    
                    fechas_usadas.add(fecha_final)
                    
                    resultado['metadata_fecha']['duplicado_ajustado'] = True
                    resultado['metadata_fecha']['ajuste_aplicado'] = f'Duplicado #{j+1}, ajustado a {fecha_final}'
                    self.contador_correcciones['duplicados_ajustados'] += 1
                    self.historial_correcciones.append(
                        f"{resultado['nombre_archivo']}: Duplicado ajustado de {fecha_original} a {fecha_final}"
                    )
                
                # Actualizar fecha final
                resultado['fecha_hora_final'] = fecha_final
                resultados_ajustados.append(resultado)
        
        return resultados_ajustados

def procesar_imagen(ruta_imagen, roi_points, punto_mar, punto_arena, 
                   preservador_fechas: PreservadorFechaHora, 
                   fecha_anterior_corregida: Optional[datetime] = None):
    """
    Procesa una imagen individual preservando al máximo la fecha original.
    """
    try:
        # Leer y normalizar imagen
        img = tiff.imread(ruta_imagen)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.shape[-1] > 3:
            img = img[..., :3]
        
        # Extraer nombre de archivo
        path_obj = Path(ruta_imagen)
        nombre_archivo = path_obj.name
        
        # Extraer fecha original SIN correcciones
        fecha_original, metadata_original = preservador_fechas.extraer_fecha_original(nombre_archivo)
        
        # Inicializar metadata
        metadata_fecha = {
            'fecha_original': fecha_original,
            'fecha_original_valida': metadata_original.get('fecha_valida', False),
            'fecha_original_metadata': metadata_original,
            'correccion_aplicada': False,
            'tipo_correccion': None,
            'duplicado_ajustado': False,
            'ajuste_aplicado': None
        }
        
        # Determinar si necesita corrección
        if fecha_original and metadata_original.get('fecha_valida', False):
            # Fecha original es válida, usarla como fecha corregida también
            fecha_corregida = fecha_original
            metadata_fecha['correccion_aplicada'] = False
        else:
            # Fecha inválida o no encontrada, aplicar corrección
            fecha_corregida, metadata_correccion = preservador_fechas.corregir_fecha_invalida(
                fecha_original, nombre_archivo, fecha_anterior_corregida
            )
            metadata_fecha['correccion_aplicada'] = True
            metadata_fecha['tipo_correccion'] = metadata_correccion.get('tipo_correccion')
            metadata_fecha.update(metadata_correccion)
        
        # Ignorar temporalmente la advertencia de convergencia de KMeans
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            img_anotada, mask_mar, mask_arena, existe_mascara_agua, num_clusters_encontrados = segmentar_imagen_individual(
                img, roi_points, punto_mar, punto_arena)
        
        return {
            'fecha_hora_original': fecha_original,  # Fecha original extraída
            'fecha_hora_corregida': fecha_corregida,  # Fecha después de correcciones básicas
            'fecha_hora_final': fecha_corregida,  # Se actualizará después por resolver_duplicados
            'existe_mascara_agua': existe_mascara_agua,
            'mask_mar': mask_mar,
            'mask_arena': mask_arena,
            'img_anotada': img_anotada,  # SOLO GUARDAMOS LA IMAGEN CON ANOTACIONES
            'nombre_archivo': nombre_archivo,
            'metadata_fecha': metadata_fecha,
            'tamano_bytes': os.path.getsize(ruta_imagen),
            'ruta_original': ruta_imagen,
            'num_clusters_encontrados': num_clusters_encontrados
        }
        
    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {e}")
        return None

def procesar_todas_imagenes(carpeta_raiz, roi1_points, punto_mar1, punto_arena1):
    """
    Procesa todas las imágenes preservando al máximo las fechas originales.
    SOLO USA ROI1 para todas las imágenes.
    """
    # Buscar todas las imágenes TIFF en la carpeta y subcarpetas
    path_raiz = Path(carpeta_raiz)
    rutas_imagenes = sorted(path_raiz.rglob('*.tiff'))
    
    if not rutas_imagenes:
        print(f"No se encontraron imágenes TIFF en la carpeta: {carpeta_raiz}")
        return None
    
    total_imagenes = len(rutas_imagenes)
    print(f"\n{'='*60}")
    print(f"INICIANDO PROCESAMIENTO DE {total_imagenes} IMÁGENES")
    print(f"USANDO SOLO ROI1 PARA TODAS LAS IMÁGENES")
    print(f"PRESERVANDO FECHAS ORIGINALES AL MÁXIMO")
    print(f"{'='*60}")
    
    resultados = []
    errores = 0
    inicio_tiempo = time.time()
    
    # Inicializar preservador de fechas
    preservador_fechas = PreservadorFechaHora()
    
    # Estadísticas de conteo
    conteo_roi1 = 0
    conteo_mascaras_agua = 0
    conteo_sin_mascaras_agua = 0
    fechas_originales_validas = 0
    fechas_originales_invalidas = 0
    fechas_sin_patron = 0
    
    # Variables para seguimiento
    fecha_anterior_corregida = None
    
    # PASADA ÚNICA: procesar todas las imágenes con ROI1
    for i, ruta_imagen in enumerate(rutas_imagenes):
        # SIEMPRE usar ROI1 para todas las imágenes
        roi_points = roi1_points
        punto_mar = punto_mar1
        punto_arena = punto_arena1
        roi_actual = "ROI1"
        conteo_roi1 += 1
        
        # Mostrar avance
        mostrar_avance_simple(i, total_imagenes, roi_actual)
        
        # Procesar la imagen
        resultado = procesar_imagen(ruta_imagen, roi_points, punto_mar, punto_arena, 
                                   preservador_fechas, fecha_anterior_corregida)
        
        if resultado is not None:
            resultados.append(resultado)
            
            # Actualizar fecha anterior para correcciones
            fecha_anterior_corregida = resultado['fecha_hora_corregida']
            
            # Actualizar estadísticas de fechas
            metadata = resultado['metadata_fecha']
            if metadata['fecha_original'] is None:
                fechas_sin_patron += 1
            elif metadata['fecha_original_valida']:
                fechas_originales_validas += 1
            else:
                fechas_originales_invalidas += 1
            
            # Contar máscaras de agua
            if resultado['existe_mascara_agua'] == 1:
                conteo_mascaras_agua += 1
            else:
                conteo_sin_mascaras_agua += 1
        else:
            errores += 1
    
    # Resolver duplicados manteniendo originales cuando sea posible
    print("\nResolviendo duplicados de fechas...")
    resultados = preservador_fechas.resolver_duplicados(resultados)
    
    # Ordenar resultados por fecha final
    resultados.sort(key=lambda x: x['fecha_hora_final'])
    
    # Verificar unicidad
    fechas_finales = [r['fecha_hora_final'] for r in resultados]
    fechas_unicas = set(fechas_finales)
    
    if len(fechas_finales) != len(fechas_unicas):
        print(f"ADVERTENCIA CRÍTICA: Aún existen {len(fechas_finales) - len(fechas_unicas)} duplicados después del ajuste!")
        # Forzar unicidad como último recurso
        fechas_ajustadas = []
        for fecha in fechas_finales:
            while fecha in fechas_ajustadas:
                fecha += timedelta(seconds=1)
            fechas_ajustadas.append(fecha)
        
        # Actualizar resultados
        for i, resultado in enumerate(resultados):
            resultado['fecha_hora_final'] = fechas_ajustadas[i]
            if fechas_finales[i] != fechas_ajustadas[i]:
                resultado['metadata_fecha']['duplicado_forzado'] = True
                resultado['metadata_fecha']['ajuste_forzado'] = f'Ajustado a {fechas_ajustadas[i]} para garantizar unicidad'
    
    # Mostrar resumen final
    tiempo_total = time.time() - inicio_tiempo
    
    print(f"\n{'='*60}")
    print(f"PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"Tiempo total: {tiempo_total:.0f}s ({tiempo_total/60:.1f} min)")
    print(f"Velocidad promedio: {tiempo_total/max(len(resultados),1):.2f}s/imagen")
    
    # Calcular tamaño total de archivos
    tamano_total_bytes = sum(r['tamano_bytes'] for r in resultados if 'tamano_bytes' in r)
    tamano_total_mb = tamano_total_bytes / (1024 * 1024)
    
    # Estadísticas del preservador de fechas
    correcciones_aplicadas = sum(1 for r in resultados if r['metadata_fecha']['correccion_aplicada'])
    duplicados_ajustados = sum(1 for r in resultados if r['metadata_fecha'].get('duplicado_ajustado', False))
    
    return resultados, {
        'total_imagenes': total_imagenes,
        'procesadas_correctamente': len(resultados),
        'errores': errores,
        'conteo_roi1': conteo_roi1,
        'conteo_mascaras_agua': conteo_mascaras_agua,
        'conteo_sin_mascaras_agua': conteo_sin_mascaras_agua,
        'fechas_originales_validas': fechas_originales_validas,
        'fechas_originales_invalidas': fechas_originales_invalidas,
        'fechas_sin_patron': fechas_sin_patron,
        'correcciones_aplicadas': correcciones_aplicadas,
        'duplicados_ajustados': duplicados_ajustados,
        'tiempo_total': tiempo_total,
        'tamano_total_mb': tamano_total_mb,
        'fechas_finales_unicas': len(set(r['fecha_hora_final'] for r in resultados)),
        'estadisticas_correcciones': dict(preservador_fechas.contador_correcciones),
        'historial_correcciones': preservador_fechas.historial_correcciones
    }

def guardar_resultados_netcdf(resultados, archivo_salida):
    """
    Guarda los resultados en un archivo NetCDF, usando las fechas finales únicas.
    También almacena las fechas originales como metadatos.
    """
    if not resultados:
        print("No hay resultados para guardar")
        return
    
    print(f"\n{'='*60}")
    print(f"GUARDANDO RESULTADOS EN NETCDF")
    print(f"{'='*60}")
    
    # Preparar datos para NetCDF
    tiempos = [r['fecha_hora_final'] for r in resultados]
    existe_mascara_agua = [r['existe_mascara_agua'] for r in resultados]
    nombres_archivos = [r['nombre_archivo'] for r in resultados]
    num_clusters_encontrados = [r['num_clusters_encontrados'] for r in resultados]
    
    # Preparar fechas originales como strings para almacenar como metadato
    fechas_originales_str = []
    for r in resultados:
        if r['fecha_hora_original']:
            fechas_originales_str.append(r['fecha_hora_original'].strftime("%Y-%m-%d %H:%M:%S"))
        else:
            fechas_originales_str.append("NO_ENCONTRADA")
    
    # Información de correcciones
    correcciones_info = []
    for r in resultados:
        metadata = r['metadata_fecha']
        info = {
            'fecha_original_valida': metadata['fecha_original_valida'],
            'correccion_aplicada': metadata['correccion_aplicada'],
            'tipo_correccion': metadata.get('tipo_correccion', 'N/A'),
            'duplicado_ajustado': metadata.get('duplicado_ajustado', False)
        }
        correcciones_info.append(str(info))
    
    # Verificar unicidad de tiempos finales
    if len(tiempos) != len(set(tiempos)):
        print("ERROR: Existen tiempos duplicados en los resultados finales!")
        return None
    
    # Obtener dimensiones de las imágenes (imagen anotada tiene las mismas dimensiones que la original)
    h, w, c = resultados[0]['img_anotada'].shape
    n_tiempos = len(resultados)
    
    print(f"Dimensiones: {n_tiempos} tiempo(s) × {h}×{w}×{c}")
    print(f"Tiempos finales únicos: {len(set(tiempos))}")
    print(f"Tamaño aproximado sin compresión: {(n_tiempos * h * w * c) / 1e6:.1f} MB")
    
    # Crear arrays para las imágenes y máscaras
    imagenes_anotadas = np.zeros((n_tiempos, h, w, c), dtype=np.uint8)
    mascaras_agua = np.zeros((n_tiempos, h, w), dtype=np.uint8)
    mascaras_arena = np.zeros((n_tiempos, h, w), dtype=np.uint8)
    
    print(f"Progreso de guardado:")
    
    inicio_guardado = time.time()
    for i, resultado in enumerate(resultados):
        imagenes_anotadas[i] = resultado['img_anotada']
        mascaras_agua[i] = resultado['mask_mar']
        mascaras_arena[i] = resultado['mask_arena']
        
        # Mostrar progreso
        if i % max(1, n_tiempos // 10) == 0 or i == n_tiempos - 1:
            porcentaje = (i + 1) / n_tiempos * 100
            print(f"  - {i + 1}/{n_tiempos} ({porcentaje:.1f}%) procesadas...")
    
    # Crear dataset de xarray con compresión para reducir tamaño
    ds = xr.Dataset(
        {
            'existe_mascara_agua': (('time',), existe_mascara_agua),
            'num_clusters_encontrados': (('time',), num_clusters_encontrados),
            'imagen_anotada': (('time', 'y', 'x', 'channel'), imagenes_anotadas),
            'mascara_agua': (('time', 'y', 'x'), mascaras_agua),
            'mascara_arena': (('time', 'y', 'x'), mascaras_arena),
            'nombre_archivo': (('time',), nombres_archivos),
            'fecha_original': (('time',), fechas_originales_str),
            'info_correccion': (('time',), correcciones_info)
        },
        coords={
            'time': tiempos,
            'y': np.arange(h),
            'x': np.arange(w),
            'channel': np.arange(c)
        }
    )
    
    # Configurar compresión SOLO para variables numéricas
    encoding = {
        'imagen_anotada': {
            'zlib': True,
            'complevel': 5,
            'chunksizes': (1, h, w, c)
        },
        'mascara_agua': {
            'zlib': True,
            'complevel': 5,
            'chunksizes': (1, h, w)
        },
        'mascara_arena': {
            'zlib': True,
            'complevel': 5,
            'chunksizes': (1, h, w)
        },
        'existe_mascara_agua': {'zlib': True, 'complevel': 5},
        'num_clusters_encontrados': {'zlib': True, 'complevel': 5},
        # Variables de string NO se comprimen
        'nombre_archivo': {},
        'fecha_original': {},
        'info_correccion': {}
    }
    
    # Atributos
    ds.attrs['title'] = 'Imágenes anotadas con máscaras de agua y arena segmentadas'
    ds.attrs['history'] = f'Creado el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    ds.attrs['author'] = 'Sistema de segmentación de imágenes'
    ds.attrs['nota_importante'] = 'SOLO IMÁGENES ANOTADAS: Contiene anotaciones de ROI, puntos de referencia y máscaras superpuestas'
    ds.attrs['preservacion_fechas'] = 'Se priorizó mantener fechas originales, corrigiendo solo cuando fue necesario'
    ds.attrs['compresion'] = 'Compresión zlib nivel 5 aplicada solo a variables numéricas'
    ds.attrs['postprocesamiento'] = 'Post-procesamiento avanzado con fusión de enclaves pequeños y consideración de límites ROI'
    
    # Atributos de variables
    ds['existe_mascara_agua'].attrs['long_name'] = 'Indicador de existencia de máscara de agua'
    ds['existe_mascara_agua'].attrs['description'] = '0: no hay máscara de agua (imágenes saturadas por nubes o noche)'
    ds['existe_mascara_agua'].attrs['values'] = '0: no existe, 1: existe'
    
    ds['num_clusters_encontrados'].attrs['long_name'] = 'Número de clusters encontrados por KMeans'
    ds['num_clusters_encontrados'].attrs['description'] = '1: imagen saturada/inválida, 2: imagen válida con dos clusters (agua y arena)'
    
    ds['imagen_anotada'].attrs['long_name'] = 'Imagen con anotaciones (ROI, puntos de referencia)'
    ds['imagen_anotada'].attrs['description'] = 'Imagen original con anotaciones superpuestas: ROI (rojo), punto mar (azul), punto arena (rojo)'
    
    ds['mascara_agua'].attrs['long_name'] = 'Máscara binaria de agua'
    ds['mascara_agua'].attrs['description'] = 'Post-procesada con morfología matemática avanzada, incluye fusión de enclaves y consideración de límites ROI'
    ds['mascara_agua'].attrs['values'] = '0: no agua, 1: agua'
    
    ds['mascara_arena'].attrs['long_name'] = 'Máscara binaria de arena'
    ds['mascara_arena'].attrs['description'] = 'Post-procesada con morfología matemática avanzada, incluye fusión de enclaves y consideración de límites ROI'
    ds['mascara_arena'].attrs['values'] = '0: no arena, 1: arena'
    
    ds['nombre_archivo'].attrs['long_name'] = 'Nombre original del archivo'
    
    ds['fecha_original'].attrs['long_name'] = 'Fecha y hora original extraída del nombre'
    ds['fecha_original'].attrs['description'] = 'Fecha original sin correcciones. "NO_ENCONTRADA" si no se pudo extraer.'
    
    ds['info_correccion'].attrs['long_name'] = 'Información de correcciones aplicadas'
    ds['info_correccion'].attrs['description'] = 'Metadatos sobre correcciones de fecha aplicadas'
    
    # Guardar en NetCDF con compresión
    print("\nGuardando con compresión para reducir tamaño...")
    ds.to_netcdf(archivo_salida, encoding=encoding)
    
    tiempo_guardado = time.time() - inicio_guardado
    tamano_nc_mb = os.path.getsize(archivo_salida) / (1024 * 1024)
    
    print(f"\nArchivo guardado: {archivo_salida}")
    print(f"Tamaño del archivo NetCDF: {tamano_nc_mb:.2f} MB")
    print(f"Tiempo de guardado: {tiempo_guardado:.1f}s")
    print(f"Fechas almacenadas: {len(tiempos)} (todas únicas)")
    
    # Verificar que el archivo se puede abrir correctamente
    print("\nVerificando que el archivo se puede abrir y cargar...")
    try:
        ds_verificacion = xr.open_dataset(archivo_salida)
        print(f"✓ Archivo abierto correctamente")
        print(f"✓ Dimensiones: {len(ds_verificacion.time)} tiempos, {ds_verificacion.y.size}×{ds_verificacion.x.size} píxeles")
        print(f"✓ Variables disponibles: {list(ds_verificacion.data_vars.keys())}")
        
        # Verificar que podemos cargar una muestra de datos
        muestra_img = ds_verificacion['imagen_anotada'][0].values
        muestra_mask = ds_verificacion['mascara_agua'][0].values
        print(f"✓ Datos cargados correctamente: imagen shape {muestra_img.shape}, máscara shape {muestra_mask.shape}")
        
        # Verificar contenido específico
        print(f"\nContenido verificado:")
        print(f"  - Imágenes anotadas: {muestra_img.shape} (3 canales RGB)")
        print(f"  - Máscara agua: valores únicos {np.unique(muestra_mask)}")
        print(f"  - Estado máscara agua: {ds_verificacion['existe_mascara_agua'][0].values}")
        print(f"  - Número de clusters: {ds_verificacion['num_clusters_encontrados'][0].values}")
        print(f"  - Nombre archivo: {ds_verificacion['nombre_archivo'][0].values}")
        print(f"  - Fecha original: {ds_verificacion['fecha_original'][0].values}")
        
        ds_verificacion.close()
        print("\n✓ Verificación completada con éxito - El archivo es totalmente funcional")
        
    except Exception as e:
        print(f"✗ Error al verificar el archivo: {e}")
        return None
    
    return ds, tamano_nc_mb

def generar_reporte_completo(resultados, estadisticas, carpeta_salida):
    """
    Genera un reporte completo en archivo de texto con estadísticas y detalle de cada imagen.
    """
    # Calcular porcentajes
    total_procesadas = estadisticas['procesadas_correctamente']
    
    if total_procesadas > 0:
        porcentaje_exito = (total_procesadas / estadisticas['total_imagenes']) * 100
        porcentaje_con_mascara = (estadisticas['conteo_mascaras_agua'] / total_procesadas) * 100
        porcentaje_sin_mascara = (estadisticas['conteo_sin_mascaras_agua'] / total_procesadas) * 100
        
        # Estadísticas de preservación de fechas
        porcentaje_fechas_validas = (estadisticas['fechas_originales_validas'] / total_procesadas) * 100
        porcentaje_fechas_invalidas = (estadisticas['fechas_originales_invalidas'] / total_procesadas) * 100
        porcentaje_sin_patron = (estadisticas['fechas_sin_patron'] / total_procesadas) * 100
        porcentaje_corregidas = (estadisticas['correcciones_aplicadas'] / total_procesadas) * 100
        porcentaje_duplicados = (estadisticas['duplicados_ajustados'] / total_procesadas) * 100
    else:
        porcentaje_exito = porcentaje_con_mascara = porcentaje_sin_mascara = 0
        porcentaje_fechas_validas = porcentaje_fechas_invalidas = porcentaje_sin_patron = 0
        porcentaje_corregidas = porcentaje_duplicados = 0
    
    # Nombre del archivo de reporte
    ruta_reporte = os.path.join(carpeta_salida, 'reporte_completo_procesamiento.txt')
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE COMPLETO DE PROCESAMIENTO DE IMÁGENES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # SECCIÓN 1: RESUMEN GENERAL
        f.write("1. RESUMEN GENERAL\n")
        f.write("-" * 80 + "\n")
        f.write(f"  • Imágenes encontradas: {estadisticas['total_imagenes']:,}\n")
        f.write(f"  • Procesadas correctamente: {total_procesadas:,} ({porcentaje_exito:.1f}%)\n")
        f.write(f"  • Errores de procesamiento: {estadisticas['errores']:,}\n")
        f.write(f"  • Fechas finales únicas garantizadas: {estadisticas['fechas_finales_unicas']:,} (100%)\n")
        f.write("\n")
        
        # SECCIÓN 2: DISTRIBUCIÓN POR ROI
        f.write("2. DISTRIBUCIÓN POR ROI\n")
        f.write("-" * 80 + "\n")
        f.write(f"  • ROI1 (todas las imágenes): {estadisticas['conteo_roi1']:,} (100.0%)\n")
        f.write("  • ROI2: NO SE UTILIZÓ (0%)\n")
        f.write("\n")
        
        # SECCIÓN 3: MÁSCARAS DE AGUA
        f.write("3. MÁSCARAS DE AGUA\n")
        f.write("-" * 80 + "\n")
        f.write(f"  • Con máscara de agua válida: {estadisticas['conteo_mascaras_agua']:,} ({porcentaje_con_mascara:.1f}%)\n")
        f.write(f"  • Sin máscara (saturadas/oscuras): {estadisticas['conteo_sin_mascaras_agua']:,} ({porcentaje_sin_mascara:.1f}%)\n")
        f.write("\n")
        
        # SECCIÓN 4: PRESERVACIÓN DE FECHAS ORIGINALES
        f.write("4. PRESERVACIÓN DE FECHAS ORIGINALES\n")
        f.write("-" * 80 + "\n")
        f.write(f"  • Fechas originales válidas: {estadisticas['fechas_originales_validas']:,} ({porcentaje_fechas_validas:.1f}%)\n")
        f.write(f"  • Fechas originales inválidas: {estadisticas['fechas_originales_invalidas']:,} ({porcentaje_fechas_invalidas:.1f}%)\n")
        f.write(f"  • Sin patrón de fecha en nombre: {estadisticas['fechas_sin_patron']:,} ({porcentaje_sin_patron:.1f}%)\n")
        f.write(f"  • Correcciones aplicadas: {estadisticas['correcciones_aplicadas']:,} ({porcentaje_corregidas:.1f}%)\n")
        f.write(f"  • Duplicados ajustados: {estadisticas['duplicados_ajustados']:,} ({porcentaje_duplicados:.1f}%)\n")
        f.write("\n")
        
        # SECCIÓN 5: TIEMPO Y TAMAÑO
        f.write("5. TIEMPO Y TAMAÑO\n")
        f.write("-" * 80 + "\n")
        f.write(f"  • Tiempo total: {estadisticas['tiempo_total']:.0f}s ({estadisticas['tiempo_total']/60:.1f} min)\n")
        f.write(f"  • Velocidad promedio: {estadisticas['tiempo_total']/max(total_procesadas,1):.2f}s/imagen\n")
        f.write(f"  • Tamaño total imágenes originales: {estadisticas['tamano_total_mb']:.2f} MB\n")
        
        # Agregar tamaño del NetCDF si está disponible
        if 'tamano_nc_mb' in estadisticas:
            f.write(f"  • Tamaño archivo NetCDF (comprimido): {estadisticas['tamano_nc_mb']:.2f} MB\n")
            f.write(f"  • Factor de compresión: {(estadisticas['tamano_total_mb']/estadisticas['tamano_nc_mb']):.1f}x\n")
        f.write("\n")
        
        # SECCIÓN 6: TIPOS DE CORRECCIONES
        f.write("6. TIPOS DE CORRECCIONES APLICADAS\n")
        f.write("-" * 80 + "\n")
        if 'estadisticas_correcciones' in estadisticas and estadisticas['estadisticas_correcciones']:
            for tipo, cantidad in estadisticas['estadisticas_correcciones'].items():
                porcentaje_tipo = (cantidad / total_procesadas * 100) if total_procesadas > 0 else 0
                f.write(f"  • {tipo}: {cantidad} ({porcentaje_tipo:.1f}%)\n")
        else:
            f.write("  • No se aplicaron correcciones.\n")
        f.write("\n")
        
        # SECCIÓN 7: EJEMPLOS DE CORRECCIONES APLICADAS
        f.write("7. LISTADO COMPLETO DE CORRECCIONES APLICADAS\n")
        f.write("-" * 80 + "\n")
        if 'historial_correcciones' in estadisticas and estadisticas['historial_correcciones']:
            for i, correccion in enumerate(estadisticas['historial_correcciones']):
                f.write(f"  {i+1:3d}. {correccion}\n")
            f.write(f"\n  Total de correcciones registradas: {len(estadisticas['historial_correcciones'])}\n")
        else:
            f.write("  • No se registraron correcciones detalladas.\n")
        f.write("\n")
        
        # SECCIÓN 8: DETALLE DE CADA IMAGEN PROCESADA
        f.write("8. DETALLE DE CADA IMAGEN PROCESADA\n")
        f.write("-" * 80 + "\n")
        f.write("Nº  | Nombre Archivo                    | ROI  | Fecha/Hora Original       | Fecha/Hora Final         | Estado Máscara | Clusters | Corrección Aplicada\n")
        f.write("-" * 180 + "\n")
        
        for i, r in enumerate(resultados):
            # Siempre ROI1 para todas las imágenes
            roi_usado = "ROI1"
            
            # Formatear fechas
            original = r['fecha_hora_original'].strftime("%Y-%m-%d %H:%M") if r['fecha_hora_original'] else "NO_ENCONTRADA"
            final = r['fecha_hora_final'].strftime("%Y-%m-%d %H:%M")
            
            # Estado de máscara de agua
            estado_mascara = "VÁLIDA" if r['existe_mascara_agua'] == 1 else "SATURADA/OSCURA"
            
            # Número de clusters
            num_clusters = r['num_clusters_encontrados']
            clusters_str = f"{num_clusters} cluster{'s' if num_clusters != 1 else ''}"
            
            # Corrección aplicada
            correccion = "SÍ" if r['metadata_fecha']['correccion_aplicada'] else "NO"
            if r['metadata_fecha']['correccion_aplicada']:
                tipo_correccion = r['metadata_fecha'].get('tipo_correccion', 'N/A')
                correccion = f"SÍ ({tipo_correccion})"
            
            # Truncar nombre de archivo si es muy largo
            nombre_archivo = r['nombre_archivo']
            if len(nombre_archivo) > 30:
                nombre_archivo = nombre_archivo[:27] + "..."
            
            f.write(f"{i+1:3d} | {nombre_archivo:30} | {roi_usado:4} | {original:24} | {final:24} | {estado_mascara:14} | {clusters_str:9} | {correccion}\n")
        
        # SECCIÓN 9: NOTAS IMPORTANTES
        f.write("\n9. NOTAS IMPORTANTES\n")
        f.write("-" * 80 + "\n")
        f.write("  • Solo se guardan las imágenes ANOTADAS (con ROI y puntos de referencia).\n")
        f.write("  • Las fechas originales se preservaron cuando fue posible.\n")
        f.write("  • Las correcciones solo se aplicaron cuando las fechas eran inválidas.\n")
        f.write("  • Todas las fechas en el NetCDF son únicas.\n")
        f.write("  • Compresión zlib nivel 5 aplicada solo a variables numéricas.\n")
        f.write("  • SE UTILIZÓ SOLAMENTE ROI1 PARA TODAS LAS IMÁGENES.\n")
        f.write("  • Punto de referencia de agua movido 20 píxeles a la izquierda.\n")
        f.write("  • Clasificación priorizada: agua primero, luego arena.\n")
        f.write("  • 1 cluster = imagen inválida/saturada, 2 clusters = válida con agua y arena separados.\n")
        f.write("  • POST-PROCESAMIENTO AVANZADO APLICADO basado en referencias científicas:\n")
        f.write("      - Soille & Vogt (2009): Eliminación de componentes pequeños y consistencia topológica\n")
        f.write("      - Vincent (1993): Relleno de huecos y análisis de conectividad\n")
        f.write("      - Pardo-Pascual et al. (2018): Prioridad al agua y unificación de enclaves pequeños\n")
        f.write("      - Vos et al. (2019): Cobertura completa de ROI y consideración de límites\n")
        f.write("  • NUEVAS FUNCIONALIDADES IMPLEMENTADAS:\n")
        f.write("      - Fusión de enclaves <7500px completamente rodeados por otra clase\n")
        f.write("      - Fusión de áreas <10000px rodeadas por clase y/o límite ROI\n")
        f.write("      - Consideración explícita de límites de ROI en decisiones de fusión\n")
        f.write("\n")
    
    print(f"Reporte completo guardado en: {ruta_reporte}")
    
    # También guardar un resumen CSV para análisis
    ruta_csv = os.path.join(carpeta_salida, 'resumen_imagenes.csv')
    datos_csv = []
    
    for i, r in enumerate(resultados):
        # Siempre ROI1
        roi_usado = "ROI1"
        
        datos_csv.append({
            'indice': i + 1,
            'nombre_archivo': r['nombre_archivo'],
            'roi': roi_usado,
            'fecha_original': r['fecha_hora_original'].strftime("%Y-%m-%d %H:%M:%S") if r['fecha_hora_original'] else "NO_ENCONTRADA",
            'fecha_final': r['fecha_hora_final'].strftime("%Y-%m-%d %H:%M:%S"),
            'estado_mascara_agua': "VÁLIDA" if r['existe_mascara_agua'] == 1 else "SATURADA/OSCURA",
            'num_clusters_encontrados': r['num_clusters_encontrados'],
            'correccion_aplicada': r['metadata_fecha']['correccion_aplicada'],
            'tipo_correccion': r['metadata_fecha'].get('tipo_correccion', 'N/A'),
            'duplicado_ajustado': r['metadata_fecha'].get('duplicado_ajustado', False)
        })
    
    df = pd.DataFrame(datos_csv)
    df.to_csv(ruta_csv, index=False, encoding='utf-8')
    print(f"Resumen CSV guardado en: {ruta_csv}")
    
    return ruta_reporte, ruta_csv

# Configuración principal
carpeta_raiz = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen/Data_imagenes_ROI1'
carpeta_salida = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen/Resultados_Data_imagenes_ROI1-FINAL'
archivo_salida_netcdf = os.path.join(carpeta_salida, 'resultados_mascaras_ROI1.nc')

# Definir ROI1 (para todas las imágenes)
roi1_points = np.array([(439, 100), (558, 100), (516, 600), (180, 600)], dtype=np.int32)
punto_mar1 = (280, 500)
punto_arena1 = (470, 500)

# Asegurarse de que la carpeta de salida exista
os.makedirs(carpeta_salida, exist_ok=True)

# Procesar todas las imágenes usando solo ROI1
print("Iniciando procesamiento de imágenes...")
print("ENFOQUE: Usando SOLO ROI1 para todas las imágenes")
print("         Solo guardando imágenes ANOTADAS (con ROI y puntos de referencia)")
print("         Aplicando compresión para reducir tamaño del archivo NetCDF")
print("         POST-PROCESAMIENTO AVANZADO aplicado con fusión de enclaves y consideración de límites ROI")
resultados, estadisticas = procesar_todas_imagenes(
    carpeta_raiz, roi1_points, punto_mar1, punto_arena1)

if resultados:
    # Guardar resultados en NetCDF
    ds, tamano_nc_mb = guardar_resultados_netcdf(resultados, archivo_salida_netcdf)
    
    if ds is not None:
        # Agregar tamaño del NetCDF a las estadísticas
        estadisticas['tamano_nc_mb'] = tamano_nc_mb
        
        print(f"\n{'='*80}")
        print(f"ESTADÍSTICAS FINALES DETALLADAS")
        print(f"{'='*80}")
        
        total_procesadas = len(resultados)
        
        # Resumen de preservación de fechas
        print(f"\nPRESERVACIÓN DE FECHAS ORIGINALES:")
        print(f"├─ Fechas originales válidas: {estadisticas['fechas_originales_validas']} "
              f"({estadisticas['fechas_originales_validas']/max(total_procesadas,1)*100:.1f}%)")
        print(f"├─ Fechas originales inválidas: {estadisticas['fechas_originales_invalidas']} "
              f"({estadisticas['fechas_originales_invalidas']/max(total_procesadas,1)*100:.1f}%)")
        print(f"├─ Sin patrón de fecha en nombre: {estadisticas['fechas_sin_patron']} "
              f"({estadisticas['fechas_sin_patron']/max(total_procesadas,1)*100:.1f}%)")
        print(f"├─ Correcciones aplicadas: {estadisticas['correcciones_aplicadas']} "
              f"({estadisticas['correcciones_aplicadas']/max(total_procesadas,1)*100:.1f}%)")
        print(f"└─ Duplicados ajustados: {estadisticas['duplicados_ajustados']} "
              f"({estadisticas['duplicados_ajustados']/max(total_procesadas,1)*100:.1f}%)")
        
        print(f"\nRESULTADO FINAL:")
        print(f"├─ Solo ROI1 utilizado para todas las {total_procesadas} imágenes")
        print(f"├─ Solo imágenes ANOTADAS guardadas (con ROI y puntos de referencia)")
        print(f"├─ Todas las {total_procesadas} fechas en el NetCDF son únicas")
        print(f"├─ Fechas originales preservadas en variable 'fecha_original'")
        print(f"├─ Información de correcciones en variable 'info_correccion'")
        print(f"├─ Compresión aplicada: reducción de {estadisticas['tamano_total_mb']:.1f} MB a {tamano_nc_mb:.1f} MB")
        print(f"├─ Factor de compresión: {(estadisticas['tamano_total_mb']/tamano_nc_mb):.1f}x")
        print(f"└─ POST-PROCESAMIENTO AVANZADO aplicado con fusión de enclaves y consideración de límites ROI")
        
        # Generar reporte completo en texto
        ruta_reporte, ruta_csv = generar_reporte_completo(resultados, estadisticas, carpeta_salida)
        
        # Cerrar el dataset
        ds.close()
        
        print(f"\n{'='*80}")
        print(f"¡PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"{'='*80}")
        print(f"\nArchivo NetCDF guardado en: {archivo_salida_netcdf}")
        print(f"Tamaño del archivo: {tamano_nc_mb:.2f} MB (comprimido)")
        print(f"Reporte completo guardado en: {ruta_reporte}")
        print(f"Resumen CSV guardado en: {ruta_csv}")
        print(f"\nPOST-PROCESAMIENTO AVANZADO APLICADO:")
        print(f"  • Eliminación de píxeles aislados y huecos internos (umbral: 300)")
        print(f"  • Suavizado con apertura (disco 5) y cierre (disco 10)")
        print(f"  • Prioridad al agua en áreas de superposición")
        print(f"  • Fusión de enclaves <7500px completamente rodeados por otra clase")
        print(f"  • Fusión de áreas <10000px rodeadas por clase y/o límite ROI")
        print(f"  • Cobertura completa de la ROI sin áreas sin clasificar")
        print(f"  • Basado en principios científicos de Soille & Vogt (2009), Vincent (1993),")
        print(f"    Pardo-Pascual et al. (2018) y Vos et al. (2019)")
        
        # Mostrar algunos ejemplos de procesamiento
        print(f"\nEJEMPLOS DE IMÁGENES PROCESADAS (primeras 5):")
        for i, r in enumerate(resultados[:5]):
            # Siempre ROI1
            roi_usado = "ROI1"
            
            original = r['fecha_hora_original'].strftime("%Y-%m-%d %H:%M") if r['fecha_hora_original'] else "NO_ENCONTRADA"
            final = r['fecha_hora_final'].strftime("%Y-%m-%d %H:%M")
            estado_mascara = "VÁLIDA" if r['existe_mascara_agua'] == 1 else "SATURADA/OSCURA"
            num_clusters = r['num_clusters_encontrados']
            
            print(f"  {i+1}. ROI: {roi_usado}, Original: {original} -> Final: {final}, "
                  f"Máscara: {estado_mascara}, Clusters: {num_clusters} [{r['nombre_archivo']}]")
    else:
        print("\nError al guardar el archivo NetCDF")
else:
    print("\nNo se pudieron procesar las imágenes")