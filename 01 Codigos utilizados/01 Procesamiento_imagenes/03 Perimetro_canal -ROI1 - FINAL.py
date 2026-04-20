import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

def generar_reporte_perimetros(ds_perimetros, ds_grilla, tiempo_total, archivo_grilla_entrada, 
                              archivo_perimetro_salida, carpeta_salida, dias_a_mostrar):
    """
    Genera un archivo de texto con un resumen completo del procesamiento de perímetros
    """
    # Crear nombre de archivo
    archivo_reporte = os.path.join(carpeta_salida, "reporte_procesamiento_Perimetros.txt")
    
    # Calcular estadísticas básicas
    n_imagenes_total = len(ds_perimetros.time)
    n_mascaras_validas = int(ds_perimetros['existe_mascara_agua'].sum().values)
    n_mascaras_invalidas = n_imagenes_total - n_mascaras_validas
    
    # Calcular estadísticas de perímetros
    total_puntos_perimetro = int(np.sum(ds_perimetros['perimetro_arena_agua'].values == 1))
    total_puntos_arena = int(np.sum(ds_grilla['valores_arena'].values == 1))
    
    # Calcular ratio perímetro/arena
    ratio_perimetro_arena = total_puntos_perimetro / total_puntos_arena if total_puntos_arena > 0 else 0
    
    # Estadísticas por tiempo
    tiempos_procesados = []
    for i in range(n_imagenes_total):
        tiempo = pd.to_datetime(ds_perimetros['time'].values[i])
        tiene_mascara = bool(ds_perimetros['existe_mascara_agua'].values[i])
        puntos_perimetro_tiempo = int(np.sum(ds_perimetros['perimetro_arena_agua'].values[i, :] == 1))
        puntos_arena_tiempo = int(np.sum(ds_grilla['valores_arena'].values[i, :] == 1))
        
        tiempos_procesados.append({
            'indice': i + 1,
            'fecha': tiempo.strftime('%Y-%m-%d %H:%M'),
            'tiene_mascara': tiene_mascara,
            'puntos_perimetro': puntos_perimetro_tiempo,
            'puntos_arena': puntos_arena_tiempo,
            'ratio': puntos_perimetro_tiempo / puntos_arena_tiempo if puntos_arena_tiempo > 0 else 0
        })
    
    # Calcular tamaño de archivos
    tamano_grilla = os.path.getsize(archivo_grilla_entrada) / (1024*1024) if os.path.exists(archivo_grilla_entrada) else 0
    tamano_perimetros = os.path.getsize(archivo_perimetro_salida) / (1024*1024) if os.path.exists(archivo_perimetro_salida) else 0
    
    # Crear contenido del reporte
    reporte = []
    reporte.append("="*80)
    reporte.append("REPORTE COMPLETO DE PROCESAMIENTO DE PERÍMETROS")
    reporte.append("="*80)
    reporte.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("")
    
    reporte.append("1. RESUMEN GENERAL")
    reporte.append("-"*80)
    reporte.append(f"  • Imágenes procesadas: {n_imagenes_total}")
    reporte.append(f"  • Procesadas correctamente: {n_mascaras_validas} ({n_mascaras_validas/n_imagenes_total*100:.1f}%)")
    reporte.append(f"  • Sin máscara de agua válida: {n_mascaras_invalidas} ({n_mascaras_invalidas/n_imagenes_total*100:.1f}%)")
    reporte.append("")
    
    reporte.append("2. CARACTERÍSTICAS DE LOS PERÍMETROS")
    reporte.append("-"*80)
    reporte.append(f"  • Puntos de perímetro totales identificados: {total_puntos_perimetro}")
    reporte.append(f"  • Puntos de arena totales: {total_puntos_arena}")
    reporte.append(f"  • Radio perímetro/arena: {ratio_perimetro_arena:.4f}")
    reporte.append(f"  • Grosor del perímetro: 1 punto (borde real)")
    reporte.append(f"  • Exclusión de bordes de ROI: ACTIVADA")
    reporte.append(f"  • Metodología: Dilatación morfológica - máscara original")
    reporte.append("")
    
    reporte.append("3. TIEMPO Y TAMAÑO")
    reporte.append("-"*80)
    reporte.append(f"  • Tiempo total de procesamiento: {tiempo_total:.2f}s ({tiempo_total/60:.1f} min)")
    velocidad_promedio = tiempo_total/n_imagenes_total if n_imagenes_total > 0 else 0
    reporte.append(f"  • Velocidad promedio: {velocidad_promedio:.2f}s/imagen")
    reporte.append(f"  • Tamaño archivo grilla de entrada: {tamano_grilla:.2f} MB")
    reporte.append(f"  • Tamaño archivo perímetros de salida: {tamano_perimetros:.2f} MB")
    if tamano_grilla > 0 and tamano_perimetros > 0:
        factor = tamano_grilla / tamano_perimetros if tamano_perimetros > 0 else 0
        reporte.append(f"  • Factor de compresión: {factor:.1f}x")
    reporte.append("")
    
    reporte.append("4. DÍAS VISUALIZADOS")
    reporte.append("-"*80)
    for i, fecha_str in enumerate(dias_a_mostrar, 1):
        reporte.append(f"  {i}. {fecha_str}")
    reporte.append("")
    
    reporte.append("5. DETALLE DE CADA IMAGEN PROCESADA")
    reporte.append("-"*80)
    reporte.append("Nº  | Fecha/Hora              | Máscara Válida | Puntos Arena | Puntos Perímetro | Radio P/A")
    reporte.append("-"*105)
    
    for img in tiempos_procesados:
        mascara_str = "SÍ" if img['tiene_mascara'] else "NO"
        reporte.append(f"{img['indice']:3d} | {img['fecha']}         | {mascara_str:^13} | {img['puntos_arena']:^11d} | {img['puntos_perimetro']:^16d} | {img['ratio']:.4f}")
    
    reporte.append("")
    reporte.append("6. ESTADÍSTICAS RESUMEN")
    reporte.append("-"*80)
    
    # Calcular promedios
    puntos_perimetro_prom = np.mean([img['puntos_perimetro'] for img in tiempos_procesados if img['tiene_mascara']])
    puntos_arena_prom = np.mean([img['puntos_arena'] for img in tiempos_procesados if img['tiene_mascara']])
    ratio_prom = np.mean([img['ratio'] for img in tiempos_procesados if img['tiene_mascara'] and img['puntos_arena'] > 0])
    
    reporte.append(f"  • Puntos de perímetro promedio: {puntos_perimetro_prom:.0f}")
    reporte.append(f"  • Puntos de arena promedio: {puntos_arena_prom:.0f}")
    reporte.append(f"  • Radio perímetro/arena promedio: {ratio_prom:.4f}")
    reporte.append("")
    
    reporte.append("7. INFORMACIÓN TÉCNICA")
    reporte.append("-"*80)
    reporte.append(f"  • Archivo grilla de entrada: {archivo_grilla_entrada}")
    reporte.append(f"  • Archivo perímetros de salida: {archivo_perimetro_salida}")
    reporte.append(f"  • Carpeta de salida: {carpeta_salida}")
    reporte.append(f"  • Metodología de cálculo: Dilatación morfológica (scipy.ndimage)")
    reporte.append(f"  • Conectividad utilizada: 4 (vecinos ortogonales)")
    reporte.append(f"  • Exclusión de bordes ROI: Sí (1 nodo de grosor)")
    
    # Escribir archivo
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    
    print(f"Reporte de perímetros generado en: {archivo_reporte}")
    return archivo_reporte

def calcular_perimetro_arena_agua(archivo_grilla_entrada, archivo_perimetro_salida):
    """
    Calcula el perímetro REAL de arena (borde de 1 punto) excluyendo bordes de ROI
    """
    print(f"Leyendo archivo de grilla: {archivo_grilla_entrada}")
    
    with xr.open_dataset(archivo_grilla_entrada) as ds_grilla:
        # Obtener datos básicos
        tiempos = ds_grilla.time.values
        n_tiempos = len(tiempos)
        n_puntos = len(ds_grilla.punto_grilla)
        
        # Coordenadas de la grilla
        coord_x = ds_grilla.coordenada_x.values
        coord_y = ds_grilla.coordenada_y.values
        
        # Encontrar límites de la ROI para excluir bordes
        x_min, x_max = np.min(coord_x), np.max(coord_x)
        y_min, y_max = np.min(coord_y), np.max(coord_y)
        
        # Identificar puntos en el borde de la ROI (1 nodo de grosor)
        en_borde_roi = ((coord_x == x_min) | (coord_x == x_max) | 
                        (coord_y == y_min) | (coord_y == y_max))
        
        print(f"Puntos en borde de ROI: {np.sum(en_borde_roi)}/{n_puntos}")
        
        # Preparar array para perímetros
        perimetro_arena_agua = np.zeros((n_tiempos, n_puntos), dtype=np.uint8)
        
        # Procesar cada tiempo que tiene máscara de agua
        print("Calculando perímetros REALES de arena (borde 1 punto, excluyendo bordes ROI)...")
        
        for i in range(n_tiempos):
            if i % 500 == 0:
                tiempo_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M')
                print(f"Procesando tiempo {i+1}/{n_tiempos} ({tiempo_str})")
            
            # Solo procesar si existe máscara de agua
            if ds_grilla.existe_mascara_agua.values[i] == 1:
                valores_arena = ds_grilla.valores_arena.values[i, :]
                
                # RECONSTRUIR LA GRILLA 2D PARA CALCULAR PERÍMETROS
                # Asumimos que la grilla es regular
                unique_x = np.unique(coord_x)
                unique_y = np.unique(coord_y)
                nx = len(unique_x)
                ny = len(unique_y)
                
                # Crear máscara 2D de arena
                mascara_arena_2d = np.zeros((nx, ny), dtype=bool)
                # Mapear coordenadas a índices
                x_to_idx = {x: idx for idx, x in enumerate(unique_x)}
                y_to_idx = {y: idx for idx, y in enumerate(unique_y)}
                
                for j in range(n_puntos):
                    if valores_arena[j] == 1:
                        idx_x = x_to_idx[coord_x[j]]
                        idx_y = y_to_idx[coord_y[j]]
                        mascara_arena_2d[idx_x, idx_y] = True
                
                # CALCULAR EL PERÍMETRO REAL (borde de 1 punto)
                # Usar operación morfológica para encontrar bordes
                estructura = ndimage.generate_binary_structure(2, 1)  # Conectividad 4
                mascara_dilatada = ndimage.binary_dilation(mascara_arena_2d, structure=estructura)
                perimetro_2d = mascara_dilatada & ~mascara_arena_2d
                
                # Convertir perímetro 2D de vuelta a 1D
                for j in range(n_puntos):
                    idx_x = x_to_idx[coord_x[j]]
                    idx_y = y_to_idx[coord_y[j]]
                    if perimetro_2d[idx_x, idx_y] and not en_borde_roi[j]:
                        perimetro_arena_agua[i, j] = 1
        
        # Crear dataset de salida OPTIMIZADO
        print("Creando archivo de perímetros...")
        
        # Filtrar solo los puntos que son perímetro en al menos un tiempo
        puntos_con_perimetro = np.any(perimetro_arena_agua == 1, axis=0)
        indices_perimetro = np.where(puntos_con_perimetro)[0]
        
        print(f"Puntos de perímetro identificados: {len(indices_perimetro)}/{n_puntos}")
        
        # Estadísticas adicionales
        total_puntos_arena = np.sum(ds_grilla.valores_arena.values == 1)
        total_puntos_perimetro = np.sum(perimetro_arena_agua == 1)
        print(f"Puntos de arena totales: {total_puntos_arena}")
        print(f"Puntos de perímetro totales: {total_puntos_perimetro}")
        print(f"Radio perímetro/arena: {total_puntos_perimetro/total_puntos_arena:.4f}")
        
        # Crear dataset solo con puntos de perímetro
        ds_salida = xr.Dataset(
            {
                'perimetro_arena_agua': (('time', 'punto_perimetro'), 
                                        perimetro_arena_agua[:, indices_perimetro]),
                'coordenada_x': (('punto_perimetro',), coord_x[indices_perimetro]),
                'coordenada_y': (('punto_perimetro',), coord_y[indices_perimetro]),
                'existe_mascara_agua': (('time',), ds_grilla.existe_mascara_agua.values.astype(np.uint8)),
                'nombre_archivo': (('time',), ds_grilla.nombre_archivo.values)
            },
            coords={
                'time': tiempos,
                'punto_perimetro': np.arange(len(indices_perimetro))
            }
        )
        
        # Atributos científicos
        ds_salida.attrs.update({
            'title': 'PERÍMETRO REAL arena (borde 1 punto, excluyendo bordes ROI)',
            'history': f'Creado el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'source_file': archivo_grilla_entrada,
            'description': 'Solo puntos del perímetro (borde) de arena con grosor de 1 punto, excluyendo bordes de ROI',
            'methodology': 'Dilatación morfológica - máscara original (borde 1 punto)',
            'total_puntos_grilla': n_puntos,
            'puntos_perimetro': len(indices_perimetro),
            'exclusion_bordes_roi': '1 nodo',
            'grosor_perimetro': '1 punto'
        })
        
        # Guardar
        print(f"Guardando archivo de perímetros: {archivo_perimetro_salida}")
        ds_salida.to_netcdf(archivo_perimetro_salida)
    
    print("Cálculo de perímetros REALES completado")
    return ds_salida

# ----------------------------------------------------------------------------
# FUNCIÓN DE VISUALIZACIÓN MODIFICADA (colores, sin stats, título personalizado)
# ----------------------------------------------------------------------------
def visualizar_perimetros_3_dias(archivo_perimetros, archivo_grilla_original, dias_a_mostrar, archivo_salida=None):
    """
    Visualiza 3 días mostrando el perímetro REAL de arena.
    Diseño científico y profesional.
    """
    with xr.open_dataset(archivo_perimetros) as ds_perimetros, \
         xr.open_dataset(archivo_grilla_original) as ds_grilla:
        
        # Configurar la figura con diseño profesional
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Elementos de la leyenda (colores actualizados: azul para agua, amarillo para arena)
        legend_elements = [
            Patch(facecolor='blue', label='Agua', alpha=0.7),
            Patch(facecolor='yellow', label='Arena', alpha=0.7),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, label='Perímetro Arena (borde 1pt)')
        ]
        
        # Procesar cada día
        for idx, (ax, fecha_str) in enumerate(zip(axes, dias_a_mostrar)):
            try:
                # Buscar tiempo más cercano
                tiempo_deseado = np.datetime64(datetime.strptime(fecha_str, "%Y-%m-%d %H:%M"))
                tiempo_idx = np.abs(ds_perimetros['time'] - tiempo_deseado).argmin().item()
                tiempo_real = ds_perimetros['time'].isel(time=tiempo_idx).values
                tiempo_str = pd.to_datetime(tiempo_real).strftime('%Y-%m-%d %H:%M')
                
                print(f"Buscando {fecha_str} -> Encontrado: {tiempo_str} (índice: {tiempo_idx})")
                
                # Verificar si existe máscara de agua
                if ds_perimetros['existe_mascara_agua'].isel(time=tiempo_idx).values == 0:
                    ax.text(0.5, 0.5, f'Sin máscara de agua\n{tiempo_str}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11)
                    ax.set_title(f'{tiempo_str}', fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Obtener datos
                perimetro = ds_perimetros['perimetro_arena_agua'].isel(time=tiempo_idx).values
                x_perimetro = ds_perimetros['coordenada_x'].values
                y_perimetro = ds_perimetros['coordenada_y'].values
                
                valores_agua = ds_grilla['valores_agua'].isel(time=tiempo_idx).values
                valores_arena = ds_grilla['valores_arena'].isel(time=tiempo_idx).values
                coord_x_grilla = ds_grilla['coordenada_x'].values
                coord_y_grilla = ds_grilla['coordenada_y'].values
                
                # Crear visualización del fondo (agua y arena) con colores azul y amarillo
                agua_mask = valores_agua == 1
                arena_mask = valores_arena == 1
                
                if np.any(agua_mask):
                    ax.scatter(coord_x_grilla[agua_mask], coord_y_grilla[agua_mask],
                               color='blue', s=10, alpha=0.6, marker='s')
                if np.any(arena_mask):
                    ax.scatter(coord_x_grilla[arena_mask], coord_y_grilla[arena_mask],
                               color='yellow', s=10, alpha=0.6, marker='s')

                # Superponer perímetro REAL
                puntos_perimetro_activos = perimetro == 1
                if np.any(puntos_perimetro_activos):
                    ax.scatter(x_perimetro[puntos_perimetro_activos], 
                             y_perimetro[puntos_perimetro_activos], 
                             c='red', s=12, alpha=0.9, marker='o',
                             edgecolors='darkred', linewidths=0.5)
                
                # Configurar gráfico
                ax.set_xlim(np.min(coord_x_grilla) - 5, np.max(coord_x_grilla) + 5)
                ax.set_ylim(np.max(coord_y_grilla) + 5, np.min(coord_y_grilla) - 5)
                
                # Título limpio
                ax.set_title(f'{tiempo_str}', fontsize=12, pad=10)
                ax.set_xlabel('Coordenada X', fontsize=10)
                ax.set_ylabel('Coordenada Y', fontsize=10)
                
                # Grid sutil
                ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
                
                # Leyenda solo en el primer gráfico
                if idx == 0:
                    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                             framealpha=0.9, fancybox=True)
                
                # --- ELIMINADO: Recuadro de estadísticas (puntos de perímetro y arena) ---
                
            except Exception as e:
                print(f"Error procesando {fecha_str}: {e}")
                ax.text(0.5, 0.5, f'Error\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{fecha_str}', fontsize=12)
        
        # Título general modificado
        plt.suptitle('Perímetros entre Agua y Arena', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Visualización guardada: {archivo_salida}")
        
        plt.show()

# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL (sin cambios)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Iniciar medición de tiempo total
    inicio_tiempo_total = datetime.now()
    
    # Configuración de rutas
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    archivo_grilla_entrada = os.path.join(carpeta_base, 'Resultados_Grilla_ROI1-FINAL', 'grilla_mascaras.nc')
    carpeta_salida = os.path.join(carpeta_base, 'Resultados_Perimetros_ROI1-FINAL')
    
    # Archivos de salida
    archivo_perimetro_salida = os.path.join(carpeta_salida, 'Data_perimetros_arena_agua.nc')
    archivo_visualizacion = os.path.join(carpeta_salida, 'Resultados_perimetros.png')
    
    # Días a visualizar
    dias_a_mostrar = [
        "2023-05-08 14:00",
        "2023-09-07 12:00",  
        "2024-07-10 08:00"
    ]
    
    # Crear carpeta de salida
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Verificar archivos
    if not os.path.exists(archivo_grilla_entrada):
        print(f"Error: No existe el archivo de grilla: {archivo_grilla_entrada}")
        print("Ejecuta primero el código de generación de grilla.")
        exit()
    
    # Procesar perímetros
    print("=== CÁLCULO DE PERÍMETROS REALES ARENA ===")
    print("Metodología: Borde de 1 punto (dilatación - máscara original) - bordes ROI")
    
    inicio_procesamiento = datetime.now()
    ds_perimetros = calcular_perimetro_arena_agua(
        archivo_grilla_entrada, 
        archivo_perimetro_salida
    )
    fin_procesamiento = datetime.now()
    tiempo_procesamiento = (fin_procesamiento - inicio_procesamiento).total_seconds()
    
    # Leer el dataset de grilla original para el reporte
    ds_grilla = xr.open_dataset(archivo_grilla_entrada)
    
    # Generar reporte
    print("\n=== GENERANDO REPORTE DE PROCESAMIENTO ===")
    archivo_reporte = generar_reporte_perimetros(
        ds_perimetros=ds_perimetros,
        ds_grilla=ds_grilla,
        tiempo_total=tiempo_procesamiento,
        archivo_grilla_entrada=archivo_grilla_entrada,
        archivo_perimetro_salida=archivo_perimetro_salida,
        carpeta_salida=carpeta_salida,
        dias_a_mostrar=dias_a_mostrar
    )
    
    # Cerrar datasets
    ds_grilla.close()
    
    # Visualizar resultados
    print("\n=== VISUALIZACIÓN DE PERÍMETROS REALES ===")
    print(f"Mostrando perímetros REALES para {len(dias_a_mostrar)} días:")
    for fecha in dias_a_mostrar:
        print(f"  - {fecha}")
    
    visualizar_perimetros_3_dias(
        archivo_perimetro_salida,
        archivo_grilla_entrada,
        dias_a_mostrar,
        archivo_visualizacion
    )
    
    # Calcular tiempo total
    fin_tiempo_total = datetime.now()
    tiempo_total = (fin_tiempo_total - inicio_tiempo_total).total_seconds()
    
    print(f"\n● Proceso completado en {tiempo_total:.2f} segundos:")
    print(f"  - Archivo de perímetros REALES: {archivo_perimetro_salida}")
    print(f"  - Visualización: {archivo_visualizacion}")
    print(f"  - Reporte detallado: {archivo_reporte}")