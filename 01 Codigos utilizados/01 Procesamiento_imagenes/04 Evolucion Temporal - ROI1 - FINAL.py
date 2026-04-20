import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

def generar_graficas_superpuestas(archivo_perimetros, archivo_grilla_original, fecha_inicio, dias, carpeta_salida):
    """
    Genera gráficas superpuestas de perímetros en una ventana de tiempo de 'dias' días.
    - Más de 1 día: un color por día, una leyenda por día.
    - 1 día: un color por instante, leyenda por instante.
    """
    print(f"Generando gráficas superpuestas para ventana: {dias} días")
    
    with xr.open_dataset(archivo_perimetros) as ds_perimetros, \
         xr.open_dataset(archivo_grilla_original) as ds_grilla:
        
        # Convertir fecha de inicio
        fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d %H:%M")
        fecha_inicio_np = np.datetime64(fecha_inicio_dt)
        
        # Calcular fecha fin según número de días
        fecha_fin_np = fecha_inicio_np + np.timedelta64(dias, 'D')
        nombre_ventana = f"{dias}_dias"
        
        # Filtrar datos en la ventana de tiempo
        mask_tiempo = (ds_perimetros.time >= fecha_inicio_np) & (ds_perimetros.time <= fecha_fin_np)
        tiempos_ventana = ds_perimetros.time.where(mask_tiempo, drop=True)
        
        # Filtrar solo tiempos con máscara de agua
        existe_agua_ventana = ds_perimetros.existe_mascara_agua.where(mask_tiempo, drop=True)
        tiempos_validos = tiempos_ventana.where(existe_agua_ventana == 1, drop=True)
        
        print(f"Tiempos encontrados en ventana: {len(tiempos_validos)}")
        
        if len(tiempos_validos) == 0:
            print("No hay datos válidos en la ventana de tiempo especificada")
            return None
        
        # Convertir tiempos a fechas (sin hora) para agrupar por día
        fechas_validos = [pd.to_datetime(t.values).strftime('%Y-%m-%d') for t in tiempos_validos]
        fechas_unicas = sorted(set(fechas_validos))
        n_dias = len(fechas_unicas)
        
        # Configurar figura
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Coordenadas de referencia para límites
        coord_x_grilla = ds_grilla['coordenada_x'].values
        coord_y_grilla = ds_grilla['coordenada_y'].values
        
        # ------------------- CASO: MÚLTIPLES DÍAS -------------------
        if n_dias > 1:
            # Un color por día
            colores_dia = plt.cm.viridis(np.linspace(0, 1, n_dias))
            color_map = {fecha: colores_dia[i] for i, fecha in enumerate(fechas_unicas)}
            
            # Agrupar y dibujar todos los puntos de cada día con su color
            for fecha in fechas_unicas:
                puntos_x = []
                puntos_y = []
                for i, tiempo in enumerate(tiempos_validos):
                    if fechas_validos[i] == fecha:
                        tiempo_idx = np.where(ds_perimetros.time == tiempo)[0][0]
                        perimetro = ds_perimetros['perimetro_arena_agua'].isel(time=tiempo_idx).values
                        x_perimetro = ds_perimetros['coordenada_x'].values
                        y_perimetro = ds_perimetros['coordenada_y'].values
                        mascara = perimetro == 1
                        if np.any(mascara):
                            puntos_x.extend(x_perimetro[mascara])
                            puntos_y.extend(y_perimetro[mascara])
                if puntos_x:
                    ax.scatter(puntos_x, puntos_y, c=color_map[fecha], s=8, alpha=0.7, label=fecha)
            
            # Leyenda: una entrada por día (máx 15)
            handles = []
            labels = []
            for fecha in fechas_unicas[:15]:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color_map[fecha], markersize=6))
                labels.append(fecha)
            if n_dias > 15:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='gray', markersize=6))
                labels.append(f'+ {n_dias-15} días')
        
        # ------------------- CASO: UN SOLO DÍA -------------------
        else:
            n_tiempos = len(tiempos_validos)
            colores_tiempo = plt.cm.viridis(np.linspace(0, 1, n_tiempos))
            leyenda_perimetros = []
            
            for i, tiempo in enumerate(tiempos_validos):
                tiempo_idx = np.where(ds_perimetros.time == tiempo)[0][0]
                tiempo_str = pd.to_datetime(tiempo.values).strftime('%Y-%m-%d %H:%M')
                perimetro = ds_perimetros['perimetro_arena_agua'].isel(time=tiempo_idx).values
                x_perimetro = ds_perimetros['coordenada_x'].values
                y_perimetro = ds_perimetros['coordenada_y'].values
                mascara = perimetro == 1
                if np.any(mascara):
                    x_activos = x_perimetro[mascara]
                    y_activos = y_perimetro[mascara]
                    scatter = ax.scatter(x_activos, y_activos,
                                        c=[colores_tiempo[i]], s=8, alpha=0.7,
                                        label=tiempo_str)
                    leyenda_perimetros.append((scatter, tiempo_str))
            
            # Leyenda: una entrada por instante (máx 30)
            handles = []
            labels = []
            max_leyenda = min(30, len(leyenda_perimetros))
            for i in range(max_leyenda):
                scatter, tiempo_str = leyenda_perimetros[i]
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=scatter.get_facecolor()[0],
                                          markersize=6))
                labels.append(tiempo_str)
            if len(leyenda_perimetros) > 30:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='gray', markersize=6))
                labels.append(f'+ {len(leyenda_perimetros)-30} más')
        
        # Configurar límites, grid, etc.
        ax.set_xlim(np.min(coord_x_grilla) - 10, np.max(coord_x_grilla) + 10)
        ax.set_ylim(np.max(coord_y_grilla) + 10, np.min(coord_y_grilla) - 10)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_xlabel('Coordenada X (píxeles)', fontsize=11)
        ax.set_ylabel('Coordenada Y (píxeles)', fontsize=11)
        
        # Título con formato requerido
        titulo = f"Evolución temporal perímetro entre Agua y Arena\nVentana tiempo: {dias} día(s)   Fecha inicio: {fecha_inicio}"
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
        
        # Leyenda (si existe)
        if handles:
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1),
                      fontsize=9, framealpha=0.95, fancybox=True)
        
        # Cuadro de estadísticas
        stats_text = f'Perímetros: {len(tiempos_validos)}\nDías: {n_dias}\nVentana: {dias} días'
        ax.text(0.02, 0.03, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Ajuste por leyenda externa
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        # Guardar
        nombre_archivo = f"perimetros_superpuestos_{nombre_ventana}_{fecha_inicio.replace(':', '-').replace(' ', '_')}.png"
        ruta_completa = os.path.join(carpeta_salida, nombre_archivo)
        plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
        print(f"Gráfico superpuesto guardado: {ruta_completa}")
        
        plt.show()
        return len(tiempos_validos), ruta_completa


def generar_video_evolucion(archivo_perimetros, archivo_grilla_original, fecha_inicio, dias, carpeta_salida, fps=1):
    """
    Genera un video MP4 con evolución de perímetros.
    - Cada día tiene un color fijo.
    - El perímetro recién agregado se muestra en rojo.
    - Título con ventana de tiempo y fecha de inicio.
    """
    print(f"Generando video de evolución para ventana: {dias} días")
    
    with xr.open_dataset(archivo_perimetros) as ds_perimetros, \
         xr.open_dataset(archivo_grilla_original) as ds_grilla:
        
        # Fechas
        fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d %H:%M")
        fecha_inicio_np = np.datetime64(fecha_inicio_dt)
        fecha_fin_np = fecha_inicio_np + np.timedelta64(dias, 'D')
        nombre_ventana = f"{dias}_dias"
        
        # Filtrar ventana
        mask_tiempo = (ds_perimetros.time >= fecha_inicio_np) & (ds_perimetros.time <= fecha_fin_np)
        tiempos_ventana = ds_perimetros.time.where(mask_tiempo, drop=True)
        existe_agua_ventana = ds_perimetros.existe_mascara_agua.where(mask_tiempo, drop=True)
        tiempos_validos = tiempos_ventana.where(existe_agua_ventana == 1, drop=True)
        
        print(f"Tiempos para video: {len(tiempos_validos)}")
        if len(tiempos_validos) == 0:
            print("No hay datos válidos para generar el video")
            return None
        
        # Agrupar por día y asignar colores fijos por día
        fechas_validos = [pd.to_datetime(t.values).strftime('%Y-%m-%d') for t in tiempos_validos]
        fechas_unicas = sorted(set(fechas_validos))
        n_dias = len(fechas_unicas)
        
        # Colores por día (viridis)
        colores_dia = plt.cm.viridis(np.linspace(0, 1, n_dias))
        color_por_fecha = {fecha: colores_dia[i] for i, fecha in enumerate(fechas_unicas)}
        
        # Precargar datos de perímetros para agilizar animación
        datos_perimetros = []
        for tiempo in tiempos_validos:
            tiempo_idx = np.where(ds_perimetros.time == tiempo)[0][0]
            perimetro = ds_perimetros['perimetro_arena_agua'].isel(time=tiempo_idx).values
            x_perimetro = ds_perimetros['coordenada_x'].values
            y_perimetro = ds_perimetros['coordenada_y'].values
            mascara = perimetro == 1
            if np.any(mascara):
                x_activos = x_perimetro[mascara]
                y_activos = y_perimetro[mascara]
            else:
                x_activos = np.array([])
                y_activos = np.array([])
            datos_perimetros.append((x_activos, y_activos))
        
        # Configurar figura
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        coord_x_grilla = ds_grilla['coordenada_x'].values
        coord_y_grilla = ds_grilla['coordenada_y'].values
        ax.set_xlim(np.min(coord_x_grilla) - 10, np.max(coord_x_grilla) + 10)
        ax.set_ylim(np.max(coord_y_grilla) + 10, np.min(coord_y_grilla) - 10)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_xlabel('Coordenada X (píxeles)', fontsize=10)
        ax.set_ylabel('Coordenada Y (píxeles)', fontsize=10)
        
        # Título base (fijo)
        base_title = f"Evolución temporal perímetro entre Agua y Arena\nVentana tiempo: {dias} día(s)   Fecha inicio: {fecha_inicio}"
        ax.set_title(base_title, fontsize=12, fontweight='bold', pad=15)
        
        scatter_plots = []
        
        def animate(frame):
            # Limpiar puntos del frame anterior
            for sc in scatter_plots:
                sc.remove()
            scatter_plots.clear()
            
            # Dibujar todos los perímetros hasta el frame actual
            for i in range(frame + 1):
                x_activos, y_activos = datos_perimetros[i]
                if len(x_activos) > 0:
                    # Color: rojo para el frame actual, color del día para los anteriores
                    if i == frame:
                        color = 'red'
                    else:
                        color = color_por_fecha[fechas_validos[i]]
                    
                    sc = ax.scatter(x_activos, y_activos, color=color, s=6, alpha=0.7)
                    scatter_plots.append(sc)
            
            # Actualizar título con tiempo actual
            tiempo_actual = pd.to_datetime(tiempos_validos[frame].values).strftime('%Y-%m-%d %H:%M')
            ax.set_title(f"{base_title}\n{tiempo_actual} ({frame+1}/{len(tiempos_validos)})",
                        fontsize=12, fontweight='bold', pad=15)
            return scatter_plots
        
        anim = animation.FuncAnimation(fig, animate, frames=len(tiempos_validos),
                                       interval=1000//fps, blit=False, repeat=True)
        
        # Guardar video
        nombre_video = f"evolucion_perimetros_{nombre_ventana}_{fecha_inicio.replace(':', '-').replace(' ', '_')}.mp4"
        ruta_video = os.path.join(carpeta_salida, nombre_video)
        
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Sistema Perímetros'), bitrate=1800)
            anim.save(ruta_video, writer=writer, dpi=150)
            print(f"Video MP4 guardado: {ruta_video}")
        except Exception as e:
            print(f"Error guardando MP4: {e}")
            try:
                ruta_gif = ruta_video.replace('.mp4', '.gif')
                anim.save(ruta_gif, writer='pillow', fps=fps)
                print(f"Video GIF guardado: {ruta_gif}")
                ruta_video = ruta_gif
            except Exception as e2:
                print(f"No se pudo guardar el video: {e2}")
                return None
        
        plt.close()
        return len(tiempos_validos), ruta_video


def procesar_ventana_seleccionada(archivo_perimetros, archivo_grilla_original, fecha_inicio, dias, carpeta_salida):
    """
    Procesa una ventana de tiempo de 'dias' días: genera imagen superpuesta y video.
    """
    print("\n" + "="*60)
    print("GENERADOR DE ANÁLISIS TEMPORAL DE PERÍMETROS")
    print("="*60)
    print(f"Fecha de inicio configurada: {fecha_inicio}")
    print(f"Días a analizar: {dias}")
    print(f"Carpeta salida: {carpeta_salida}")
    
    resultados = {}
    
    try:
        # Gráfica superpuesta
        print(f"\n{'─'*40}")
        print("GENERANDO GRÁFICA SUPERPUESTA")
        print(f"{'─'*40}")
        n_perimetros, ruta_grafica = generar_graficas_superpuestas(
            archivo_perimetros, archivo_grilla_original,
            fecha_inicio, dias, carpeta_salida
        )
        if n_perimetros is not None:
            resultados['grafica'] = {'perimetros': n_perimetros, 'ruta': ruta_grafica}
        
        # Video de evolución
        print(f"\n{'─'*40}")
        print("GENERANDO VIDEO DE EVOLUCIÓN")
        print(f"{'─'*40}")
        n_frames, ruta_video = generar_video_evolucion(
            archivo_perimetros, archivo_grilla_original,
            fecha_inicio, dias, carpeta_salida, fps=1
        )
        if n_frames is not None:
            resultados['video'] = {'frames': n_frames, 'ruta': ruta_video}
        
    except Exception as e:
        print(f"Error procesando ventana: {e}")
        return None
    
    return resultados


if __name__ == "__main__":
    # ------------------------------------------------------------
    # CONFIGURACIÓN - MODIFICAR AQUÍ FECHA INICIO Y NÚMERO DE DÍAS
    # ------------------------------------------------------------
    FECHA_INICIO = "2024-07-10 00:00"   # Formato: "YYYY-MM-DD HH:MM"
    NUMERO_DIAS = 5                    # Número de días a analizar
    
    # Rutas fijas (ajustar según tu sistema)
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    archivo_perimetros = os.path.join(carpeta_base, 'Resultados_Perimetros_ROI1-FINAL', 'Data_perimetros_arena_agua.nc')
    archivo_grilla_original = os.path.join(carpeta_base, 'Resultados_Grilla_ROI1-FINAL', 'grilla_mascaras.nc')
    carpeta_salida = os.path.join(carpeta_base, 'Resultados_Evolucion_ROI1-FINAL')
    
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Verificar archivos
    if not os.path.exists(archivo_perimetros):
        print(f"Error: No existe archivo de perímetros: {archivo_perimetros}")
        print("Ejecute primero el código de cálculo de perímetros.")
        exit()
    if not os.path.exists(archivo_grilla_original):
        print(f"Error: No existe archivo de grilla: {archivo_grilla_original}")
        exit()
    
    # Procesar
    resultados = procesar_ventana_seleccionada(
        archivo_perimetros,
        archivo_grilla_original,
        FECHA_INICIO,
        NUMERO_DIAS,
        carpeta_salida
    )
    
    if resultados:
        print(f"\n{'═'*50}")
        print("RESULTADOS DEL ANÁLISIS")
        print(f"{'═'*50}")
        if 'grafica' in resultados:
            print(f"✓ Gráfica superpuesta: {resultados['grafica']['perimetros']} perímetros")
            print(f"   Ruta: {resultados['grafica']['ruta']}")
        if 'video' in resultados:
            print(f"✓ Video de evolución: {resultados['video']['frames']} frames")
            print(f"   Ruta: {resultados['video']['ruta']}")
    
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"Todos los resultados han sido guardados en:")
    print(f" {carpeta_salida}")