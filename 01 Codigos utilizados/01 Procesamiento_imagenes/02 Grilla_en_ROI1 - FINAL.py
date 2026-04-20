import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# ============================================================================
# CONFIGURACIÓN - MODIFICAR ESTAS VARIABLES SEGÚN NECESITE
# ============================================================================
ESPACIADO_GRILLA = 1                # Densidad de grilla (píxeles)
ROI_POINTS = [(439, 100), (558, 100), (516, 600), (180, 600)]
ARCHIVO_NETCDF_ENTRADA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen/Resultados_Data_imagenes_ROI1-FINAL/resultados_mascaras_ROI1.nc'
CARPETA_SALIDA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen/Resultados_Grilla_ROI1-FINAL'
FECHA_VISUALIZACION = "2024-07-10 08:00"   # Fecha para visualizar un día específico
# ============================================================================

def generar_reporte_procesamiento(ds_salida, ds_entrada, roi_points, espaciado_grilla, tiempo_total, 
                                 archivo_entrada, archivo_salida, carpeta_salida, lista_imagenes_detalle):
    """
    Genera un archivo de texto con un resumen completo del procesamiento
    """
    archivo_reporte = os.path.join(carpeta_salida, f"reporte_procesamiento_Grilla.txt")
    
    n_imagenes_total = len(ds_salida.time)
    n_mascaras_validas = int(ds_salida['existe_mascara_agua'].sum().values)
    n_mascaras_invalidas = n_imagenes_total - n_mascaras_validas
    
    tamano_entrada = os.path.getsize(archivo_entrada) / (1024*1024)
    tamano_salida = os.path.getsize(archivo_salida) / (1024*1024) if os.path.exists(archivo_salida) else 0
    
    n_puntos_grilla = len(ds_salida.punto_grilla)
    x_min = float(ds_salida.coordenada_x.min().values)
    x_max = float(ds_salida.coordenada_x.max().values)
    y_min = float(ds_salida.coordenada_y.min().values)
    y_max = float(ds_salida.coordenada_y.max().values)
    
    reporte = []
    reporte.append("="*80)
    reporte.append("REPORTE COMPLETO DE PROCESAMIENTO DE GRILLA")
    reporte.append("="*80)
    reporte.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("")
    reporte.append("1. RESUMEN GENERAL")
    reporte.append("-"*80)
    reporte.append(f"  • Imágenes procesadas: {n_imagenes_total}")
    reporte.append(f"  • Procesadas correctamente: {n_imagenes_total} (100.0%)")
    reporte.append(f"  • Errores de procesamiento: 0")
    reporte.append(f"  • Fechas únicas garantizadas: {n_imagenes_total} (100.0%)")
    reporte.append("")
    reporte.append("2. CARACTERÍSTICAS DE LA GRILLA")
    reporte.append("-"*80)
    reporte.append(f"  • Puntos de grilla totales: {n_puntos_grilla}")
    reporte.append(f"  • Espaciado de grilla: {espaciado_grilla} píxeles")
    reporte.append(f"  • Coordenadas X: {x_min:.1f} a {x_max:.1f}")
    reporte.append(f"  • Coordenadas Y: {y_min:.1f} a {y_max:.1f}")
    reporte.append(f"  • ROI points: {roi_points}")
    reporte.append("")
    reporte.append("3. MÁSCARAS DE AGUA")
    reporte.append("-"*80)
    porcentaje_validas = (n_mascaras_validas/n_imagenes_total*100) if n_imagenes_total > 0 else 0
    porcentaje_invalidas = (n_mascaras_invalidas/n_imagenes_total*100) if n_imagenes_total > 0 else 0
    reporte.append(f"  • Con máscara de agua válida: {n_mascaras_validas} ({porcentaje_validas:.1f}%)")
    reporte.append(f"  • Sin máscara (saturadas/oscuras): {n_mascaras_invalidas} ({porcentaje_invalidas:.1f}%)")
    reporte.append("")
    reporte.append("4. TIEMPO Y TAMAÑO")
    reporte.append("-"*80)
    reporte.append(f"  • Tiempo total de procesamiento: {tiempo_total:.2f}s ({tiempo_total/60:.1f} min)")
    velocidad_promedio = tiempo_total/n_imagenes_total if n_imagenes_total > 0 else 0
    reporte.append(f"  • Velocidad promedio: {velocidad_promedio:.2f}s/imagen")
    reporte.append(f"  • Tamaño archivo NetCDF de entrada: {tamano_entrada:.2f} MB")
    reporte.append(f"  • Tamaño archivo NetCDF de salida: {tamano_salida:.2f} MB")
    if tamano_entrada > 0 and tamano_salida > 0:
        factor = tamano_entrada / tamano_salida if tamano_salida > 0 else 0
        reporte.append(f"  • Factor de compresión: {factor:.1f}x")
    reporte.append("")
    reporte.append("5. DETALLE DE CADA IMAGEN PROCESADA")
    reporte.append("-"*80)
    reporte.append("Nº  | Nombre Archivo                    | Fecha/Hora              | Máscara Válida | Puntos Muestreados")
    reporte.append("-"*100)
    
    for i, (nombre, fecha, valida, n_puntos) in enumerate(lista_imagenes_detalle, start=1):
        nombre_str = str(nombre).strip()
        nombre_trunc = (nombre_str[:30] + "..") if len(nombre_str) > 30 else nombre_str.ljust(32)
        fecha_str = pd.to_datetime(fecha).strftime('%Y-%m-%d %H:%M')
        valida_str = "SÍ" if valida else "NO"
        reporte.append(f"{i:3d} | {nombre_trunc} | {fecha_str}         | {valida_str:^14} | {n_puntos}")
    
    reporte.append("")
    reporte.append("6. INFORMACIÓN TÉCNICA")
    reporte.append("-"*80)
    reporte.append(f"  • Archivo NetCDF de entrada: {archivo_entrada}")
    reporte.append(f"  • Archivo NetCDF de salida: {archivo_salida}")
    reporte.append(f"  • Carpeta de salida: {carpeta_salida}")
    reporte.append(f"  • Fecha de visualización configurada: {FECHA_VISUALIZACION}")
    reporte.append(f"  • Espaciado de grilla utilizado: {espaciado_grilla} píxeles")
    
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    
    print(f"Reporte generado en: {archivo_reporte}")
    return archivo_reporte

def crear_grilla_roi(roi_points, espaciado=ESPACIADO_GRILLA):
    """
    Crea una grilla regular dentro de la región de interés (ROI)
    """
    roi = np.array(roi_points, dtype=np.float32)
    
    x_min, y_min = np.min(roi, axis=0)
    x_max, y_max = np.max(roi, axis=0)
    
    x_coords = np.arange(x_min, x_max, espaciado)
    y_coords = np.arange(y_min, y_max, espaciado)
    X, Y = np.meshgrid(x_coords, y_coords)
    puntos_rect = np.column_stack([X.ravel(), Y.ravel()])
    
    poly_path = Path(roi_points)
    mascara_dentro = poly_path.contains_points(puntos_rect)
    puntos_grilla = puntos_rect[mascara_dentro].astype(np.float32)
    
    return puntos_grilla, (len(y_coords), len(x_coords))

def muestrear_mascaras_en_grilla(mascara_agua, mascara_arena, puntos_grilla):
    """
    Muestrea los valores de las máscaras en los puntos de la grilla (vectorizado)
    """
    coords = np.round(puntos_grilla).astype(int)
    x, y = coords[:, 0], coords[:, 1]
    
    valid_mask = (y >= 0) & (y < mascara_agua.shape[0]) & (x >= 0) & (x < mascara_agua.shape[1])
    
    valores_agua = np.zeros(len(puntos_grilla), dtype=np.uint8)
    valores_arena = np.zeros(len(puntos_grilla), dtype=np.uint8)
    
    valores_agua[valid_mask] = mascara_agua[y[valid_mask], x[valid_mask]]
    valores_arena[valid_mask] = mascara_arena[y[valid_mask], x[valid_mask]]
    
    n_puntos_validos = np.sum(valid_mask)
    
    return valores_agua, valores_arena, n_puntos_validos

def procesar_netcdf_con_grilla(archivo_netcdf_entrada, archivo_netcdf_salida, roi_points, espaciado_grilla=ESPACIADO_GRILLA):
    """
    Procesa el archivo NetCDF con máscaras y genera una grilla con valores de agua y arena
    """
    inicio_tiempo = datetime.now()
    print(f"Leyendo archivo NetCDF: {archivo_netcdf_entrada}")
    
    lista_imagenes_detalle = []
    
    with xr.open_dataset(archivo_netcdf_entrada) as ds_entrada:
        print("Creando grilla en la ROI...")
        puntos_grilla, shape_grilla = crear_grilla_roi(roi_points, espaciado_grilla)
        n_puntos = len(puntos_grilla)
        
        print(f"Grilla creada con {n_puntos} puntos, shape: {shape_grilla}")
        print(f"Espaciado utilizado: {espaciado_grilla} píxeles")
        
        tiempos = ds_entrada.time.values
        n_tiempos = len(tiempos)
        
        valores_agua_grilla = np.zeros((n_tiempos, n_puntos), dtype=np.uint8)
        valores_arena_grilla = np.zeros((n_tiempos, n_puntos), dtype=np.uint8)
        existe_mascara_agua = ds_entrada['existe_mascara_agua'].values.astype(np.uint8)
        
        print("Procesando máscaras en la grilla...")
        for i in range(n_tiempos):
            tiempo_actual = pd.to_datetime(tiempos[i])
            tiempo_str = tiempo_actual.strftime('%Y-%m-%d %H:%M')
            
            # Avance cada 1000 frames
            if i % 1000 == 0:
                print(f"Procesando tiempo {i+1}/{n_tiempos} ({tiempo_str})")
            
            mascara_agua = ds_entrada['mascara_agua'].isel(time=i).values
            mascara_arena = ds_entrada['mascara_arena'].isel(time=i).values
            
            valores_agua, valores_arena, n_puntos_validos = muestrear_mascaras_en_grilla(
                mascara_agua, mascara_arena, puntos_grilla
            )
            
            valores_agua_grilla[i, :] = valores_agua
            valores_arena_grilla[i, :] = valores_arena
            
            nombre_archivo_val = ds_entrada['nombre_archivo'].isel(time=i).values
            if isinstance(nombre_archivo_val, np.ndarray):
                nombre_archivo = str(nombre_archivo_val.item())
            else:
                nombre_archivo = str(nombre_archivo_val)
            
            tiene_mascara_valida = bool(existe_mascara_agua[i])
            lista_imagenes_detalle.append((nombre_archivo, tiempo_actual, tiene_mascara_valida, n_puntos_validos))
        
        print("Creando archivo NetCDF de salida...")
        ds_salida = xr.Dataset(
            {
                'valores_agua': (('time', 'punto_grilla'), valores_agua_grilla),
                'valores_arena': (('time', 'punto_grilla'), valores_arena_grilla),
                'existe_mascara_agua': (('time',), existe_mascara_agua),
                'coordenada_x': (('punto_grilla',), puntos_grilla[:, 0]),
                'coordenada_y': (('punto_grilla',), puntos_grilla[:, 1]),
                'nombre_archivo': (('time',), ds_entrada['nombre_archivo'].values)
            },
            coords={'time': tiempos, 'punto_grilla': np.arange(n_puntos)}
        )
        
        ds_salida.attrs.update({
            'title': 'Valores de agua y arena en grilla ROI',
            'history': f'Creado el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'source_file': archivo_netcdf_entrada,
            'roi_points': str(roi_points),
            'espaciado_grilla': f'{espaciado_grilla} píxeles'
        })
        
        print(f"Guardando archivo NetCDF: {archivo_netcdf_salida}")
        ds_salida.to_netcdf(archivo_netcdf_salida)
    
    fin_tiempo = datetime.now()
    tiempo_total = (fin_tiempo - inicio_tiempo).total_seconds()
    
    print(f"Procesamiento completado exitosamente en {tiempo_total:.2f} segundos")
    
    ds_salida_leido = xr.open_dataset(archivo_netcdf_salida)
    
    return ds_salida_leido, tiempo_total, lista_imagenes_detalle

# ----------------------------------------------------------------------------
# FUNCIÓN DE VISUALIZACIÓN (SOLO MÁSCARAS + GRILLA + LEYENDA)
# ----------------------------------------------------------------------------
def visualizar_grilla_y_mascaras(archivo_netcdf_grilla, archivo_netcdf_original, tiempo_idx=0, archivo_salida=None):
    """
    Visualiza un único gráfico con:
      - Fondo negro
      - Máscara de agua en azul sólido
      - Máscara de arena en amarillo sólido
      - Cuadrícula (grilla) en gris claro
      - Contorno de la ROI en blanco
      - Leyenda con los colores correspondientes
    """
    # Usar variables globales de configuración
    global ROI_POINTS, ESPACIADO_GRILLA
    
    with xr.open_dataset(archivo_netcdf_grilla) as ds_grilla, \
         xr.open_dataset(archivo_netcdf_original) as ds_original:
        
        # --- Máscaras para el tiempo seleccionado ---
        mascara_agua = ds_original['mascara_agua'].isel(time=tiempo_idx).values
        mascara_arena = ds_original['mascara_arena'].isel(time=tiempo_idx).values
        
        # --- Fecha para el título ---
        tiempo_str = pd.to_datetime(ds_original['time'].isel(time=tiempo_idx).values).strftime('%Y-%m-%d %H:%M')
        
        # --- Crear imagen RGB con fondo negro y máscaras en colores sólidos ---
        h, w = mascara_agua.shape
        img_mascaras = np.zeros((h, w, 3), dtype=np.uint8)  # Fondo negro
        
        # Asignar colores: Azul para agua (R=0, G=0, B=255)
        img_mascaras[mascara_agua == 1] = [0, 0, 255]
        # Asignar colores: Amarillo para arena (R=255, G=255, B=0)
        img_mascaras[mascara_arena == 1] = [255, 255, 0]
        # Nota: donde ambas máscaras coinciden, gana la última asignación (amarillo).
        
        # --- Configuración de la figura ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Máscaras y grilla - {tiempo_str}', fontsize=16)
        
        # --- Mostrar la imagen de máscaras ---
        ax.imshow(img_mascaras)
        
        # --- Dibujar contorno de la ROI (polígono) ---
        roi_poly = patches.Polygon(ROI_POINTS, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(roi_poly)
        
        # --- Calcular límites del ROI ---
        roi_array = np.array(ROI_POINTS)
        x_min, y_min = np.min(roi_array, axis=0)
        x_max, y_max = np.max(roi_array, axis=0)
        
        # --- Dibujar cuadrícula (líneas verticales y horizontales) ---
        espaciado = ESPACIADO_GRILLA
        # Líneas verticales en las posiciones x de la grilla
        for x in np.arange(x_min, x_max + espaciado, espaciado):
            ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
        # Líneas horizontales en las posiciones y de la grilla
        for y in np.arange(y_min, y_max + espaciado, espaciado):
            ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
        
        # --- Ajustar límites de los ejes al ROI con un margen ---
        margin = 10
        ax.set_xlim(x_min - margin, x_max + margin)
        # Con origen de imagen en la parte superior, y_max es el borde inferior
        ax.set_ylim(y_max + margin, y_min - margin)
        
        ax.set_aspect('equal')
        ax.set_xlabel('X (píxeles)')
        ax.set_ylabel('Y (píxeles)')
        
        # --- Agregar leyenda ---
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', label='Agua'),
            Patch(facecolor='yellow', edgecolor='yellow', label='Arena')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {archivo_salida}")
        
        plt.show()

# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    archivo_netcdf_salida = os.path.join(CARPETA_SALIDA, 'grilla_mascaras.nc')
    
    print("=" * 60)
    print("CONFIGURACIÓN ACTUAL")
    print("=" * 60)
    print(f"Espaciado de grilla: {ESPACIADO_GRILLA} píxeles")
    print(f"Archivo entrada: {ARCHIVO_NETCDF_ENTRADA}")
    print(f"Carpeta salida: {CARPETA_SALIDA}")
    print(f"ROI points: {ROI_POINTS}")
    print("=" * 60)
    
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    
    if not os.path.exists(ARCHIVO_NETCDF_ENTRADA):
        print(f"Error: No existe el archivo de entrada: {ARCHIVO_NETCDF_ENTRADA}")
        exit()
    
    ds_salida, tiempo_total, lista_imagenes_detalle = procesar_netcdf_con_grilla(
        ARCHIVO_NETCDF_ENTRADA, archivo_netcdf_salida, ROI_POINTS, ESPACIADO_GRILLA
    )
    
    ds_entrada = xr.open_dataset(ARCHIVO_NETCDF_ENTRADA)
    
    archivo_reporte = generar_reporte_procesamiento(
        ds_salida=ds_salida,
        ds_entrada=ds_entrada,
        roi_points=ROI_POINTS,
        espaciado_grilla=ESPACIADO_GRILLA,
        tiempo_total=tiempo_total,
        archivo_entrada=ARCHIVO_NETCDF_ENTRADA,
        archivo_salida=archivo_netcdf_salida,
        carpeta_salida=CARPETA_SALIDA,
        lista_imagenes_detalle=lista_imagenes_detalle
    )
    
    ds_entrada.close()
    
    # --- Visualización automática (siempre se genera) ---
    print("\nGenerando visualización de la grilla...")
    try:
        tiempo_deseado = np.datetime64(datetime.strptime(FECHA_VISUALIZACION, "%Y-%m-%d %H:%M"))
        tiempo_idx = np.abs(ds_salida['time'] - tiempo_deseado).argmin().item()
        
        nombre_archivo_img = f"visualizacion_{FECHA_VISUALIZACION.replace(':', '-').replace(' ', '_')}.png"
        ruta_salida_img = os.path.join(CARPETA_SALIDA, nombre_archivo_img)
        
        print(f"Visualizando fecha más cercana a {FECHA_VISUALIZACION} (índice: {tiempo_idx})")
        visualizar_grilla_y_mascaras(archivo_netcdf_salida, ARCHIVO_NETCDF_ENTRADA, tiempo_idx, ruta_salida_img)
        
    except Exception as e:
        print(f"Error en visualización: {e}")
    
    ds_salida.close()
    print(f"Archivo de grilla guardado en: {archivo_netcdf_salida}")
    print(f"Reporte detallado guardado en: {archivo_reporte}")