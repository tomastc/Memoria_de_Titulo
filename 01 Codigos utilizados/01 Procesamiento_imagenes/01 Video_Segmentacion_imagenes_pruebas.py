import cv2
import numpy as np
import xarray as xr
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN GLOBAL DE FPS - MODIFICA AQUÍ
# ============================================================================
FPS_VIDEO = 5  # Cambia este valor para ajustar los FPS del video
# ============================================================================

def crear_video_desde_netcdf(archivo_netcdf, archivo_salida_video, intervalo=0.1):
    """
    Crea un video a partir de los resultados en el archivo NetCDF.
    
    Parámetros:
    -----------
    archivo_netcdf : str
        Ruta al archivo NetCDF generado por la segmentación
    archivo_salida_video : str
        Ruta donde se guardará el video de salida
    intervalo : float
        Intervalo entre frames en segundos (usado para calcular tiempo real)
    """
    
    print(f"\n{'='*80}")
    print(f"CREANDO VIDEO DESDE ARCHIVO NETCDF")
    print(f"{'='*80}")
    
    inicio_video = time.time()
    
    try:
        # Abrir el dataset NetCDF
        print(f"Abriendo archivo NetCDF: {archivo_netcdf}")
        ds = xr.open_dataset(archivo_netcdf)
        
        # Obtener dimensiones
        n_tiempos = ds.sizes['time']
        h, w = ds.sizes['y'], ds.sizes['x']
        
        print(f"Total de frames en NetCDF: {n_tiempos}")
        print(f"Dimensiones de cada imagen: {h}×{w} píxeles")
        
        # Calcular tamaño de la ventana (4 columnas: anotada, agua, arena, superpuesta)
        window_width = w * 4
        window_height = h
        
        # Configurar VideoWriter usando FPS global
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(archivo_salida_video), fourcc, FPS_VIDEO, (window_width, window_height))
        
        if not video_writer.isOpened():
            print(f"Error al crear video: {archivo_salida_video}")
            ds.close()
            return False
        
        print(f"\nConfiguración del video:")
        print(f"  • Dimensiones: {window_width}×{window_height}")
        print(f"  • FPS: {FPS_VIDEO} (configuración global)")
        print(f"  • Duración aproximada: {n_tiempos/FPS_VIDEO:.1f} segundos")
        print(f"  • Archivo de salida: {archivo_salida_video}")
        
        print(f"\nIniciando procesamiento de frames...")
        print(f"{'-'*40}")
        
        # Estadísticas para mostrar al final
        total_con_mascara_agua = 0
        total_sin_mascara_agua = 0
        frames_procesados = 0
        
        for i in range(n_tiempos):
            try:
                # Obtener datos para este tiempo
                img_anotada = ds['imagen_anotada'][i].values
                mask_agua = ds['mascara_agua'][i].values.astype(np.uint8)
                mask_arena = ds['mascara_arena'][i].values.astype(np.uint8)
                existe_mascara_agua = int(ds['existe_mascara_agua'][i].values)
                nombre_archivo = str(ds['nombre_archivo'][i].values)
                tiempo = ds['time'][i].values
                
                # Verificar dimensiones de las máscaras
                if mask_agua.ndim != 2:
                    print(f"  Advertencia frame {i+1}: máscara agua tiene {mask_agua.ndim} dimensiones, redimensionando...")
                    if mask_agua.ndim == 3:
                        mask_agua = mask_agua[:, :, 0]  # Tomar solo el primer canal si es 3D
                
                if mask_arena.ndim != 2:
                    print(f"  Advertencia frame {i+1}: máscara arena tiene {mask_arena.ndim} dimensiones, redimensionando...")
                    if mask_arena.ndim == 3:
                        mask_arena = mask_arena[:, :, 0]  # Tomar solo el primer canal si es 3D
                
                # Actualizar estadísticas
                if existe_mascara_agua == 1:
                    total_con_mascara_agua += 1
                else:
                    total_sin_mascara_agua += 1
                
                # Convertir imagen anotada a BGR para OpenCV
                if img_anotada.shape[2] == 4:  # Si tiene canal alpha (RGBA)
                    img_anotada = img_anotada[:, :, :3]  # Quitar canal alpha
                
                img_anotada_bgr = cv2.cvtColor(img_anotada, cv2.COLOR_RGB2BGR)
                
                # 1. Crear visualización de máscara de agua (en azul) - versión CORREGIDA
                mask_agua_vis = np.zeros((h, w, 3), dtype=np.uint8)
                # Asegurar que mask_agua sea 2D para indexación booleana
                agua_mask_2d = mask_agua == 1
                mask_agua_vis[agua_mask_2d] = [255, 0, 0]  # BGR: Azul
                
                # 2. Crear visualización de máscara de arena (en verde) - versión CORREGIDA
                mask_arena_vis = np.zeros((h, w, 3), dtype=np.uint8)
                # Asegurar que mask_arena sea 2D para indexación booleana
                arena_mask_2d = mask_arena == 1
                mask_arena_vis[arena_mask_2d] = [0, 255, 0]  # BGR: Verde
                
                # 3. Crear imagen superpuesta - versión CORREGIDA
                img_superpuesta = img_anotada_bgr.copy()
                
                # Superponer agua en azul (semi-transparente)
                if np.any(agua_mask_2d):
                    # Crear una imagen azul del mismo tamaño
                    azul_layer = np.zeros_like(img_superpuesta)
                    azul_layer[agua_mask_2d] = [255, 0, 0]  # Azul en BGR
                    
                    # Aplicar superposición
                    alpha = 0.6  # Transparencia del color azul
                    img_superpuesta[agua_mask_2d] = cv2.addWeighted(
                        img_superpuesta[agua_mask_2d], 0.4,
                        azul_layer[agua_mask_2d], alpha, 0
                    )
                
                # Superponer arena en verde (semi-transparente)
                if np.any(arena_mask_2d):
                    # Crear una imagen verde del mismo tamaño
                    verde_layer = np.zeros_like(img_superpuesta)
                    verde_layer[arena_mask_2d] = [0, 255, 0]  # Verde en BGR
                    
                    # Aplicar superposición
                    alpha = 0.6  # Transparencia del color verde
                    img_superpuesta[arena_mask_2d] = cv2.addWeighted(
                        img_superpuesta[arena_mask_2d], 0.4,
                        verde_layer[arena_mask_2d], alpha, 0
                    )
                
                # Crear canvas vacío para los 4 paneles
                canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                
                # Colocar las 4 imágenes en el canvas
                canvas[0:h, 0:w] = img_anotada_bgr          # Columna 1: Imagen anotada
                canvas[0:h, w:2*w] = mask_agua_vis          # Columna 2: Máscara agua
                canvas[0:h, 2*w:3*w] = mask_arena_vis       # Columna 3: Máscara arena
                canvas[0:h, 3*w:4*w] = img_superpuesta      # Columna 4: Superposición
                
                # Añadir texto informativo
                # Formatear tiempo
                if isinstance(tiempo, np.datetime64):
                    tiempo_str = pd.to_datetime(tiempo).strftime("%Y-%m-%d %H:%M")
                else:
                    tiempo_str = str(tiempo)
                
                # Estado de máscara de agua
                estado_agua = "VÁLIDA" if existe_mascara_agua == 1 else "SATURADA/OSCURA"
                color_estado = (0, 255, 0) if existe_mascara_agua == 1 else (0, 0, 255)
                
                # Información en la primera columna
                cv2.putText(canvas, f'Frame: {i+1}/{n_tiempos}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(canvas, f'Fecha: {tiempo_str}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Truncar nombre de archivo si es muy largo
                nombre_corto = nombre_archivo
                if len(nombre_archivo) > 30:
                    nombre_corto = nombre_archivo[:27] + "..."
                
                cv2.putText(canvas, f'Archivo: {nombre_corto}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(canvas, f'Estado agua: {estado_agua}', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)
                
                # Información en la segunda columna (máscara agua)
                area_agua = np.sum(mask_agua == 1)
                porcentaje_agua = (area_agua / (h * w)) * 100 if (h * w) > 0 else 0
                cv2.putText(canvas, f'Agua: {area_agua:,} px', (w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, f'({porcentaje_agua:.1f}%)', (w + 10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Información en la tercera columna (máscara arena)
                area_arena = np.sum(mask_arena == 1)
                porcentaje_arena = (area_arena / (h * w)) * 100 if (h * w) > 0 else 0
                cv2.putText(canvas, f'Arena: {area_arena:,} px', (2*w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, f'({porcentaje_arena:.1f}%)', (2*w + 10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Información en la cuarta columna (superposición)
                area_total = area_agua + area_arena
                porcentaje_total = (area_total / (h * w)) * 100 if (h * w) > 0 else 0
                cv2.putText(canvas, f'Total: {area_total:,} px', (3*w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, f'({porcentaje_total:.1f}%)', (3*w + 10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Títulos de las columnas
                titulos = ["IMAGEN ANOTADA", "MÁSCARA AGUA", "MÁSCARA ARENA", "SUPERPOSICIÓN"]
                for j, titulo in enumerate(titulos):
                    x_pos = j * w + 10
                    y_pos = h - 20
                    cv2.putText(canvas, titulo, (x_pos, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Añadir separadores entre columnas
                for j in range(1, 4):
                    x_pos = j * w
                    cv2.line(canvas, (x_pos, 0), (x_pos, h), (255, 255, 255), 2)
                
                # Escribir frame en el video
                video_writer.write(canvas)
                frames_procesados += 1
                
                # Mostrar progreso cada 10 frames o al final
                if (i + 1) % 10 == 0 or i == n_tiempos - 1:
                    porcentaje = ((i + 1) / n_tiempos) * 100
                    print(f"  Procesado frame {i+1}/{n_tiempos} ({porcentaje:.1f}%)")
                    
            except Exception as e:
                print(f"  Error en frame {i+1}: {e}")
                continue
        
        # Liberar recursos
        video_writer.release()
        ds.close()
        
        # Calcular tiempo de procesamiento
        tiempo_video = time.time() - inicio_video
        
        # Mostrar resumen final
        print(f"\n{'='*80}")
        print(f"VIDEO CREADO EXITOSAMENTE")
        print(f"{'='*80}")
        
        if os.path.exists(archivo_salida_video):
            tamano_video = os.path.getsize(archivo_salida_video) / (1024 * 1024)
            
            print(f"\nRESUMEN FINAL:")
            print(f"  • Archivo de video: {archivo_salida_video}")
            print(f"  • Tamaño del video: {tamano_video:.2f} MB")
            print(f"  • Frames procesados: {frames_procesados}/{n_tiempos}")
            print(f"  • Dimensiones: {window_width}×{window_height}")
            print(f"  • FPS: {FPS_VIDEO} (configuración global)")
            print(f"  • Duración: {n_tiempos/FPS_VIDEO:.1f} segundos")
            print(f"  • Tiempo de procesamiento: {tiempo_video:.1f}s")
            
            print(f"\nESTADÍSTICAS DEL CONTENIDO:")
            print(f"  • Frames con máscara de agua válida: {total_con_mascara_agua} ({total_con_mascara_agua/n_tiempos*100:.1f}%)")
            print(f"  • Frames sin máscara (saturados/oscuros): {total_sin_mascara_agua} ({total_sin_mascara_agua/n_tiempos*100:.1f}%)")
            
            print(f"\nCONFIGURACIÓN ACTUAL:")
            print(f"  • FPS del video: {FPS_VIDEO}")
            print(f"  • Para cambiar los FPS, modifica la variable 'FPS_VIDEO' al inicio del script.")
            
            print(f"\n Video listo para visualizar.")
            print(f"  Puedes abrirlo con cualquier reproductor multimedia.")
            
            return True
        else:
            print(f"\n Error: El video no se creó correctamente.")
            return False
        
    except Exception as e:
        print(f"\n Error al procesar el archivo NetCDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def crear_imagen_muestra(archivo_netcdf, archivo_video):
    """
    Crea una imagen de muestra con el primer frame del video para referencia.
    Opcional: puedes comentar esta función si no la necesitas.
    """
    try:
        # Obtener la carpeta del video
        carpeta_video = os.path.dirname(archivo_video)
        nombre_base = os.path.splitext(os.path.basename(archivo_video))[0]
        archivo_muestra = os.path.join(carpeta_video, f"{nombre_base}_muestra.png")
        
        # Abrir el dataset para obtener el primer frame
        ds = xr.open_dataset(archivo_netcdf)
        
        # Obtener primer frame
        img_anotada = ds['imagen_anotada'][0].values
        mask_agua = ds['mascara_agua'][0].values.astype(np.uint8)
        mask_arena = ds['mascara_arena'][0].values.astype(np.uint8)
        existe_mascara_agua = int(ds['existe_mascara_agua'][0].values)
        tiempo = ds['time'][0].values
        
        # Verificar dimensiones de las máscaras
        if mask_agua.ndim != 2:
            if mask_agua.ndim == 3:
                mask_agua = mask_agua[:, :, 0]
        
        if mask_arena.ndim != 2:
            if mask_arena.ndim == 3:
                mask_arena = mask_arena[:, :, 0]
        
        ds.close()
        
        # Convertir a BGR
        if img_anotada.shape[2] == 4:
            img_anotada = img_anotada[:, :, :3]
        
        img_anotada_bgr = cv2.cvtColor(img_anotada, cv2.COLOR_RGB2BGR)
        
        # Crear visualizaciones
        h, w = img_anotada.shape[:2]
        
        # Máscara agua (azul)
        mask_agua_vis = np.zeros((h, w, 3), dtype=np.uint8)
        agua_mask_2d = mask_agua == 1
        mask_agua_vis[agua_mask_2d] = [255, 0, 0]
        
        # Máscara arena (verde)
        mask_arena_vis = np.zeros((h, w, 3), dtype=np.uint8)
        arena_mask_2d = mask_arena == 1
        mask_arena_vis[arena_mask_2d] = [0, 255, 0]
        
        # Superposición
        img_superpuesta = img_anotada_bgr.copy()
        
        if np.any(agua_mask_2d):
            azul_layer = np.zeros_like(img_superpuesta)
            azul_layer[agua_mask_2d] = [255, 0, 0]
            img_superpuesta[agua_mask_2d] = cv2.addWeighted(
                img_superpuesta[agua_mask_2d], 0.4,
                azul_layer[agua_mask_2d], 0.6, 0
            )
        
        if np.any(arena_mask_2d):
            verde_layer = np.zeros_like(img_superpuesta)
            verde_layer[arena_mask_2d] = [0, 255, 0]
            img_superpuesta[arena_mask_2d] = cv2.addWeighted(
                img_superpuesta[arena_mask_2d], 0.4,
                verde_layer[arena_mask_2d], 0.6, 0
            )
        
        # Crear imagen con los 4 paneles
        imagen_muestra = np.zeros((h, w * 4, 3), dtype=np.uint8)
        imagen_muestra[0:h, 0:w] = img_anotada_bgr
        imagen_muestra[0:h, w:2*w] = mask_agua_vis
        imagen_muestra[0:h, 2*w:3*w] = mask_arena_vis
        imagen_muestra[0:h, 3*w:4*w] = img_superpuesta
        
        # Añadir títulos
        titulos = ["IMAGEN ANOTADA", "MÁSCARA AGUA", "MÁSCARA ARENA", "SUPERPOSICIÓN"]
        for j, titulo in enumerate(titulos):
            x_pos = j * w + 10
            y_pos = 30
            cv2.putText(imagen_muestra, titulo, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Guardar imagen
        cv2.imwrite(archivo_muestra, imagen_muestra)
        print(f" Imagen de muestra guardada en: {archivo_muestra}")
        
        return archivo_muestra
        
    except Exception as e:
        print(f" No se pudo crear la imagen de muestra: {e}")
        return None

# Configuración principal
if __name__ == "__main__":
    # Configurar rutas
    carpeta_resultados = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen/Resultados_Data_imagenes_pruebas2'
    archivo_netcdf = os.path.join(carpeta_resultados, 'resultados_mascaras_ROI1.nc')
    archivo_video = os.path.join(carpeta_resultados, 'video_segmentacion_ROI1_4dias.mp4')
    
    # Verificar que el archivo NetCDF existe
    if not os.path.exists(archivo_netcdf):
        print(f" Error: No se encontró el archivo NetCDF: {archivo_netcdf}")
        print("  Asegúrate de haber ejecutado primero el script de segmentación.")
        exit(1)
    
    # Crear video
    print(f"{'='*80}")
    print(f"GENERADOR DE VIDEO DE SEGMENTACIÓN")
    print(f"{'='*80}")
    print(f"Archivo NetCDF fuente: {archivo_netcdf}")
    print(f"Archivo video destino: {archivo_video}")
    print(f"FPS CONFIGURADO: {FPS_VIDEO} (para cambiar, modifica la variable 'FPS_VIDEO' al inicio del script)")
    print(f"{'='*80}")
    
    # Crear el video
    exito = crear_video_desde_netcdf(
        archivo_netcdf=archivo_netcdf,
        archivo_salida_video=archivo_video,
        intervalo=0.1
    )
    
    if exito:
        # Opcional: crear imagen de muestra (comenta si no la necesitas)
        crear_imagen_muestra(archivo_netcdf, archivo_video)
        
        print(f"\n{'='*80}")
        print(f"¡PROCESO COMPLETADO!")
        print(f"{'='*80}")
        
    else:
        print(f"\n Error al crear el video.")