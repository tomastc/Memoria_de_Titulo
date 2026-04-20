import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN MODIFICABLE =====================
# Parámetros para el gráfico de rango de fechas (Requerimiento 4)
FECHA_INICIO = '2024-07-10'       # Fecha de inicio (YYYY-MM-DD)
DIAS_A_GRAFICAR = 7                # Cantidad de días a incluir
# =====================================================================

def add_closed_vertical_lines(ax, fechas, estados, alpha=0.3, linewidth=0.8):
    """
    Agrega líneas verticales rojas en los instantes donde el estado es cerrado.
    """
    for i, (fecha, estado) in enumerate(zip(fechas, estados)):
        if estado == 0:
            ax.axvline(x=fecha, color='red', alpha=alpha, linewidth=linewidth, zorder=1)

def generar_graficos_mensuales_anchos(archivo_anchos_nc, carpeta_resultados):
    """
    Genera gráficos mensuales de la evolución temporal de los anchos de desembocadura
    a partir del archivo NetCDF de salida ya calculado.
    Retorna el número de gráficos individuales generados.
    """
    print("="*80)
    print("GENERANDO GRÁFICOS MENSUALES DE ANCHOS DE DESEMBOCADURA")
    print("="*80)
    print(f"Archivo de entrada: {archivo_anchos_nc}")
    
    # Abrir el archivo NetCDF con los anchos ya calculados
    try:
        ds_anchos = xr.open_dataset(archivo_anchos_nc)
    except Exception as e:
        print(f"ERROR: No se pudo abrir el archivo NetCDF: {e}")
        return 0
    
    # Verificar que existan las variables necesarias
    variables_requeridas = ['time', 'estado_apertura', 'ancho_desembocadura']
    for var in variables_requeridas:
        if var not in ds_anchos:
            print(f"ERROR: Variable requerida '{var}' no encontrada en el archivo")
            ds_anchos.close()
            return 0
    
    # Extraer datos
    tiempos = ds_anchos.time.values
    estados = ds_anchos.estado_apertura.values
    anchos = ds_anchos.ancho_desembocadura.values
    
    # Convertir a DataFrame para facilitar el procesamiento por mes
    df = pd.DataFrame({
        'fecha': pd.to_datetime(tiempos),
        'estado': estados,
        'ancho': anchos
    })
    
    # Asegurarse de que las fechas estén ordenadas
    df = df.sort_values('fecha')
    
    # Extraer año y mes
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['mes_nombre'] = df['fecha'].dt.strftime('%B')
    df['año_mes'] = df['fecha'].dt.to_period('M')
    
    # Obtener lista única de meses
    periodos_unicos = sorted(df['año_mes'].unique())
    
    print(f"\nTotal de instantes procesados: {len(df)}")
    print(f"Períodos encontrados: {len(periodos_unicos)} meses")
    
    # Crear subcarpeta para gráficos mensuales (Requerimiento 1: nombre exacto)
    carpeta_graficos_mensuales = os.path.join(carpeta_resultados, "Graficas mensuales")
    os.makedirs(carpeta_graficos_mensuales, exist_ok=True)
    
    # Generar un gráfico por cada mes
    graficos_generados = 0
    
    for periodo in periodos_unicos:
        # Filtrar datos del mes
        df_mes = df[df['año_mes'] == periodo]
        
        if len(df_mes) == 0:
            continue
        
        # Extraer información del mes
        año = df_mes['año'].iloc[0]
        mes_num = df_mes['mes'].iloc[0]
        mes_nombre = df_mes['mes_nombre'].iloc[0]
        
        # Calcular estadísticas del mes
        total_instantes = len(df_mes)
        instantes_abiertos = np.sum(df_mes['estado'] == 1)
        instantes_cerrados = np.sum(df_mes['estado'] == 0)
        
        # Anchos válidos (estado=1 y ancho>0)
        df_anchos_validos = df_mes[(df_mes['estado'] == 1) & (df_mes['ancho'] > 0)]
        anchos_validos_count = len(df_anchos_validos)
        
        if anchos_validos_count > 0:
            ancho_promedio = np.mean(df_anchos_validos['ancho'])
            ancho_maximo = np.max(df_anchos_validos['ancho'])
            ancho_minimo = np.min(df_anchos_validos['ancho'])
            ancho_std = np.std(df_anchos_validos['ancho'])
        else:
            ancho_promedio = 0.0
            ancho_maximo = 0.0
            ancho_minimo = 0.0
            ancho_std = 0.0
        
        # Separar datos para graficar
        fechas = df_mes['fecha']
        anchos_mes = df_mes['ancho']
        estados_mes = df_mes['estado']
        
        # Crear máscaras para diferentes categorías
        mascara_abierto_con_ancho = (estados_mes == 1) & (anchos_mes > 0)
        mascara_abierto_sin_ancho = (estados_mes == 1) & (anchos_mes == 0)
        mascara_cerrado = (estados_mes == 0)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Graficar puntos según categoría
        if np.any(mascara_abierto_con_ancho):
            n = np.sum(mascara_abierto_con_ancho)
            porcentaje = n / total_instantes * 100
            ax.scatter(fechas[mascara_abierto_con_ancho], 
                      anchos_mes[mascara_abierto_con_ancho],
                      c='green', s=40, alpha=0.7, marker='o', 
                      label=f'Abierto con ancho > 0: {n} ({porcentaje:.1f}%)')
        
        if np.any(mascara_abierto_sin_ancho):
            n = np.sum(mascara_abierto_sin_ancho)
            porcentaje = n / total_instantes * 100
            ax.scatter(fechas[mascara_abierto_sin_ancho], 
                      anchos_mes[mascara_abierto_sin_ancho],
                      c='orange', s=30, alpha=0.5, marker='^',
                      label=f'Abierto con ancho = 0: {n} ({porcentaje:.1f}%)')
        
        if np.any(mascara_cerrado):
            n = np.sum(mascara_cerrado)
            porcentaje = n / total_instantes * 100
            ax.scatter(fechas[mascara_cerrado], 
                      anchos_mes[mascara_cerrado],
                      c='red', s=20, alpha=0.3, marker='x',
                      label=f'Cerrado: {n} ({porcentaje:.1f}%)')
        
        # Agregar líneas verticales rojas para estados cerrados (Requerimiento 2)
        add_closed_vertical_lines(ax, fechas, estados_mes)
        
        # Configurar ejes y formato
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
        
        # Formatear eje x para mostrar todos los días del mes (Requerimiento 3)
        dias_del_mes = pd.date_range(start=fechas.min().replace(day=1), 
                                     end=fechas.max().replace(day=pd.Timestamp(fechas.min().year, fechas.min().month, 1).days_in_month),
                                     freq='D')
        ax.set_xticks(dias_del_mes)
        ax.xaxis.set_major_formatter(DateFormatter('%d'))
        
        # Rotar etiquetas de fecha
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='right')
        
        # Agregar cuadrícula
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Configurar límites del eje y
        if anchos_validos_count > 0:
            y_max = max(ancho_maximo * 1.1, 1)  # Mínimo 1 metro para visualización
            ax.set_ylim(bottom=0, top=y_max)
        
        # Título del gráfico
        titulo = (f'EVOLUCIÓN TEMPORAL DEL ANCHO DE DESEMBOCADURA - {mes_nombre} {año}\n'
                 f'Total instantes: {total_instantes}')
        
        ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
        
        # Texto con estadísticas (incluyendo porcentajes)
        if anchos_validos_count > 0:
            stats_text = (f'Estadísticas (solo anchos > 0):\n'
                         f'• Ancho promedio: {ancho_promedio:.2f} m\n'
                         f'• Desviación estándar: {ancho_std:.2f} m\n'
                         f'• Ancho máximo: {ancho_maximo:.2f} m\n'
                         f'• Ancho mínimo: {ancho_minimo:.2f} m\n'
                         f'• Datos con ancho>0: {anchos_validos_count}/{instantes_abiertos} ({anchos_validos_count/instantes_abiertos*100:.1f}% de abiertos)')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            stats_text = (f'No hay anchos válidos (> 0) este mes\n'
                         f'Instantes abiertos: {instantes_abiertos}\n'
                         f'Instantes cerrados: {instantes_cerrados}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Leyenda
        ax.legend(loc='upper right', fontsize=10)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        nombre_archivo = f'anchos_mensual_{año:04d}_{mes_num:02d}_{mes_nombre}.png'
        ruta_completa = os.path.join(carpeta_graficos_mensuales, nombre_archivo)
        
        plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        graficos_generados += 1
        
        print(f"  Gráfico generado: {nombre_archivo}")
        print(f"    • Período: {mes_nombre} {año}")
        print(f"    • Instantes: {total_instantes} (Abiertos: {instantes_abiertos}, Cerrados: {instantes_cerrados})")
        if anchos_validos_count > 0:
            print(f"    • Ancho promedio: {ancho_promedio:.2f} m, Máximo: {ancho_maximo:.2f} m")
        print()
    
    # También generar un gráfico resumen con todos los meses en subplots (modificado: 3 columnas, 6 filas)
    if len(periodos_unicos) > 1:
        generar_grafico_resumen_mensual(df, periodos_unicos, carpeta_graficos_mensuales)
    
    # Cerrar dataset
    ds_anchos.close()
    
    print(f"\nResumen de gráficos mensuales generados:")
    print(f"  • Total de gráficos mensuales: {graficos_generados}")
    print(f"  • Carpeta de gráficos: {carpeta_graficos_mensuales}")
    print(f"  • Archivo NetCDF fuente: {archivo_anchos_nc}")
    
    return graficos_generados

def generar_grafico_resumen_mensual(df, periodos_unicos, carpeta_graficos_mensuales):
    """
    Genera gráficos resumen con todos los meses en subplots de 3 columnas y hasta 6 filas por figura.
    Si hay más de 18 meses, se generan múltiples figuras.
    """
    print("Generando gráfico(s) resumen con todos los meses...")
    
    n_meses = len(periodos_unicos)
    meses_por_figura = 18  # 3 columnas x 6 filas
    n_figuras = int(np.ceil(n_meses / meses_por_figura))
    
    for fig_num in range(n_figuras):
        start_idx = fig_num * meses_por_figura
        end_idx = min(start_idx + meses_por_figura, n_meses)
        periodos_figura = periodos_unicos[start_idx:end_idx]
        
        n_meses_fig = len(periodos_figura)
        n_cols = 3
        n_rows = int(np.ceil(n_meses_fig / n_cols))
        
        # Crear figura
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        # Aplanar ejes a una lista unidimensional de objetos Axes
        if n_rows * n_cols == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
        
        # Para cada mes en esta figura
        for idx, (ax, periodo) in enumerate(zip(axes_list, periodos_figura)):
            if idx >= n_meses_fig:
                # Esto no debería ocurrir porque zip se detiene al menor, pero por seguridad
                break
            
            # Filtrar datos del mes
            df_mes = df[df['año_mes'] == periodo]
            
            # Extraer información del mes
            año = df_mes['año'].iloc[0]
            mes_num = df_mes['mes'].iloc[0]
            mes_nombre_corto = df_mes['fecha'].iloc[0].strftime('%b')
            
            # Separar datos
            fechas = df_mes['fecha']
            anchos_mes = df_mes['ancho']
            estados_mes = df_mes['estado']
            
            # Crear máscaras
            mascara_abierto_con_ancho = (estados_mes == 1) & (anchos_mes > 0)
            mascara_abierto_sin_ancho = (estados_mes == 1) & (anchos_mes == 0)
            mascara_cerrado = (estados_mes == 0)
            
            # Graficar puntos según categoría
            if np.any(mascara_abierto_con_ancho):
                ax.scatter(fechas[mascara_abierto_con_ancho], 
                          anchos_mes[mascara_abierto_con_ancho],
                          c='green', s=15, alpha=0.7, marker='o')
            
            if np.any(mascara_abierto_sin_ancho):
                ax.scatter(fechas[mascara_abierto_sin_ancho], 
                          anchos_mes[mascara_abierto_sin_ancho],
                          c='orange', s=10, alpha=0.5, marker='^')
            
            if np.any(mascara_cerrado):
                ax.scatter(fechas[mascara_cerrado], 
                          anchos_mes[mascara_cerrado],
                          c='red', s=5, alpha=0.3, marker='x')
            
            # Líneas verticales para cerrados
            add_closed_vertical_lines(ax, fechas, estados_mes, alpha=0.2, linewidth=0.5)
            
            # Configurar subgráfico
            ax.set_title(f'{mes_nombre_corto} {año}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Mostrar todos los días del mes en el eje X (aunque no haya datos)
            dias_del_mes = pd.date_range(start=fechas.min().replace(day=1), 
                                         end=fechas.min().replace(day=pd.Timestamp(fechas.min().year, fechas.min().month, 1).days_in_month),
                                         freq='D')
            ax.set_xticks(dias_del_mes)
            ax.xaxis.set_major_formatter(DateFormatter('%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
            
            # Configurar límites del eje y
            if np.any(mascara_abierto_con_ancho):
                ancho_max = np.max(anchos_mes[mascara_abierto_con_ancho])
                y_max = max(ancho_max * 1.1, 1)
                ax.set_ylim(bottom=0, top=y_max)
            else:
                ax.set_ylim(bottom=0, top=10)  # Valor por defecto
            
            # Estadísticas en el gráfico con porcentajes
            total = len(df_mes)
            abiertos = np.sum(estados_mes == 1)
            cerrados = np.sum(estados_mes == 0)
            if np.any(mascara_abierto_con_ancho):
                n_validos = np.sum(mascara_abierto_con_ancho)
                pct_validos = n_validos / total * 100
                ancho_prom = np.mean(anchos_mes[mascara_abierto_con_ancho])
                stats_text = f'Ab>0: {n_validos} ({pct_validos:.1f}%)\nμ={ancho_prom:.1f}m'
            else:
                stats_text = f'Ab: {abiertos}/{total}\nCerr: {cerrados}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Ocultar ejes sobrantes (los que no se usaron)
        for ax in axes_list[n_meses_fig:]:
            ax.axis('off')
        
        # Ajustar layout general
        if n_figuras > 1:
            titulo_fig = f'RESUMEN MENSUAL DE ANCHOS (Figura {fig_num+1} de {n_figuras})'
        else:
            titulo_fig = 'RESUMEN MENSUAL DE ANCHOS'
        plt.suptitle(titulo_fig, fontsize=16, fontweight='bold', y=0.98)
        
        # Agregar leyenda general
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Abierto con ancho > 0'),
            Patch(facecolor='orange', alpha=0.5, label='Abierto con ancho = 0'),
            Patch(facecolor='red', alpha=0.3, label='Cerrado')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Guardar gráfico resumen
        if n_figuras > 1:
            nombre_resumen = f"resumen_mensual_anchos_parte{fig_num+1}.png"
        else:
            nombre_resumen = "resumen_mensual_anchos.png"
        ruta_resumen = os.path.join(carpeta_graficos_mensuales, nombre_resumen)
        plt.savefig(ruta_resumen, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Gráfico resumen generado: {nombre_resumen}")
        print(f"    • Meses incluidos: {n_meses_fig}")

# ===================== FUNCIÓN RESTAURADA: gráfico único para rango de fechas (sin tendencia) =====================
def generar_grafico_rango_fechas(df, carpeta_resultados, fecha_inicio, dias):
    """
    Genera un gráfico para un rango de fechas específico (Requerimiento 4).
    Retorna True si se generó el gráfico.
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO PARA RANGO DE FECHAS PERSONALIZADO")
    print("="*80)
    
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = fecha_inicio_dt + timedelta(days=dias)
    
    # Filtrar datos en el rango
    df_rango = df[(df['fecha'] >= fecha_inicio_dt) & (df['fecha'] < fecha_fin_dt)].copy()
    
    if len(df_rango) == 0:
        print(f"ADVERTENCIA: No hay datos en el rango {fecha_inicio} a {fecha_fin_dt.date()}")
        return False
    
    print(f"Rango seleccionado: {fecha_inicio} a {fecha_fin_dt.date()}")
    print(f"Instantes encontrados: {len(df_rango)}")
    
    # Calcular estadísticas
    total = len(df_rango)
    abiertos = np.sum(df_rango['estado'] == 1)
    cerrados = np.sum(df_rango['estado'] == 0)
    
    df_anchos_validos = df_rango[(df_rango['estado'] == 1) & (df_rango['ancho'] > 0)]
    anchos_validos_count = len(df_anchos_validos)
    
    if anchos_validos_count > 0:
        ancho_prom = np.mean(df_anchos_validos['ancho'])
        ancho_std = np.std(df_anchos_validos['ancho'])
        ancho_max = np.max(df_anchos_validos['ancho'])
        ancho_min = np.min(df_anchos_validos['ancho'])
    else:
        ancho_prom = ancho_std = ancho_max = ancho_min = 0.0
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Separar por categorías
    mascara_abierto_pos = (df_rango['estado'] == 1) & (df_rango['ancho'] > 0)
    mascara_abierto_cero = (df_rango['estado'] == 1) & (df_rango['ancho'] == 0)
    mascara_cerrado = (df_rango['estado'] == 0)
    
    # Graficar con porcentajes en leyenda
    if np.any(mascara_abierto_pos):
        n = np.sum(mascara_abierto_pos)
        pct = n / total * 100
        ax.scatter(df_rango.loc[mascara_abierto_pos, 'fecha'], 
                  df_rango.loc[mascara_abierto_pos, 'ancho'],
                  c='green', s=40, alpha=0.7, marker='o',
                  label=f'Abierto con ancho > 0: {n} ({pct:.1f}%)')
    
    if np.any(mascara_abierto_cero):
        n = np.sum(mascara_abierto_cero)
        pct = n / total * 100
        ax.scatter(df_rango.loc[mascara_abierto_cero, 'fecha'], 
                  df_rango.loc[mascara_abierto_cero, 'ancho'],
                  c='orange', s=30, alpha=0.5, marker='^',
                  label=f'Abierto con ancho = 0: {n} ({pct:.1f}%)')
    
    if np.any(mascara_cerrado):
        n = np.sum(mascara_cerrado)
        pct = n / total * 100
        ax.scatter(df_rango.loc[mascara_cerrado, 'fecha'], 
                  df_rango.loc[mascara_cerrado, 'ancho'],
                  c='red', s=20, alpha=0.3, marker='x',
                  label=f'Cerrado: {n} ({pct:.1f}%)')
    
    # Líneas verticales para cerrados
    add_closed_vertical_lines(ax, df_rango['fecha'], df_rango['estado'])
    
    # Configuración del eje X con formato detallado
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Fecha y hora', fontsize=12)
    ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
    ax.set_title(f'EVOLUCIÓN DEL ANCHO - {fecha_inicio} a {fecha_fin_dt.date()}\n'
                f'Total instantes: {total} | Abiertos: {abiertos} | Cerrados: {cerrados}',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if anchos_validos_count > 0:
        y_max = max(ancho_max * 1.1, 1)
        ax.set_ylim(bottom=0, top=y_max)
        
        stats_text = (f'Estadísticas (anchos > 0):\n'
                     f'• Promedio: {ancho_prom:.2f} m\n'
                     f'• Desv. estándar: {ancho_std:.2f} m\n'
                     f'• Máximo: {ancho_max:.2f} m\n'
                     f'• Mínimo: {ancho_min:.2f} m\n'
                     f'• Datos con ancho>0: {anchos_validos_count}/{abiertos} ({anchos_validos_count/abiertos*100:.1f}% de abiertos)')
    else:
        stats_text = f'No hay anchos válidos (> 0) en el período\nAbiertos: {abiertos}\nCerrados: {cerrados}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar
    nombre_archivo = f"Evolucion_ancho_{dias}_dias.png"
    ruta_archivo = os.path.join(carpeta_resultados, nombre_archivo)
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Gráfico guardado en: {ruta_archivo}")
    return True

# ===================== FUNCIÓN MODIFICADA: boxplots diarios por mes (un gráfico por mes con boxplots por día) =====================
def generar_boxplots_diarios_mensuales(df, carpeta_resultados):
    """
    Genera, para cada mes con datos, un gráfico que contiene un boxplot por cada día del mes.
    Cada boxplot muestra la distribución de los anchos de ese día (solo instantes válidos).
    Además, genera un resumen con todos los meses en subplots (cada subplot contiene los boxplots diarios de ese mes),
    con distribución de 3 columnas y 6 filas.
    Retorna una tupla (número de gráficos individuales, número de resúmenes).
    """
    print("\n" + "="*80)
    print("GENERANDO BOXPLOTS DIARIOS POR MES (UN GRÁFICO POR MES CON BOXPLOTS POR DÍA)")
    print("="*80)
    
    # Crear carpeta para los boxplots
    carpeta_boxplots = os.path.join(carpeta_resultados, "boxplots_diarios_mensuales")
    os.makedirs(carpeta_boxplots, exist_ok=True)
    
    # Filtrar solo instantes válidos (estado=1 y ancho>0)
    df_validos = df[(df['estado'] == 1) & (df['ancho'] > 0)].copy()
    if len(df_validos) == 0:
        print("No hay datos válidos para generar boxplots.")
        return (0, 0)
    
    # Crear columnas de año-mes y día
    df_validos['año_mes'] = df_validos['fecha'].dt.to_period('M')
    df_validos['dia'] = df_validos['fecha'].dt.day
    
    # Obtener lista de meses únicos
    meses_unicos = sorted(df_validos['año_mes'].unique())
    print(f"Meses con datos válidos: {len(meses_unicos)}")
    
    n_individuales = 0
    # Generar un gráfico por cada mes con boxplots por día
    for mes in meses_unicos:
        df_mes = df_validos[df_validos['año_mes'] == mes]
        año = int(str(mes).split('-')[0])
        mes_num = int(str(mes).split('-')[1])
        mes_nombre = pd.Timestamp(año, mes_num, 1).strftime('%B')
        
        # Determinar el último día del mes
        ultimo_dia = pd.Timestamp(año, mes_num, 1).days_in_month
        
        # Preparar datos por día
        data_por_dia = []
        dias_con_datos = []
        for dia in range(1, ultimo_dia + 1):
            anchos_dia = df_mes[df_mes['dia'] == dia]['ancho'].values
            if len(anchos_dia) > 0:
                data_por_dia.append(anchos_dia)
                dias_con_datos.append(dia)
        
        if len(data_por_dia) == 0:
            continue  # No hay datos en ningún día de este mes (no debería ocurrir)
        
        # Crear figura con tamaño fijo (15,8) como los gráficos mensuales
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Boxplot por día
        bp = ax.boxplot(data_por_dia, positions=dias_con_datos, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='darkblue'),
                        whiskerprops=dict(color='darkblue'),
                        capprops=dict(color='darkblue'),
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
        
        # Configurar ejes
        ax.set_xlabel('Día del mes', fontsize=12)
        ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
        ax.set_title(f'Distribución diaria de anchos - {mes_nombre} {año}\n'
                     f'(solo instantes abiertos con ancho > 0)', fontsize=14, fontweight='bold')
        
        # Establecer ticks del eje X en todos los días del mes
        ax.set_xticks(range(1, ultimo_dia + 1))
        ax.set_xticklabels([str(d) for d in range(1, ultimo_dia + 1)], rotation=0, ha='right')
        ax.set_xlim(0, ultimo_dia + 1)
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Estadísticas del mes
        stats_text = (f'Total datos: {len(df_mes)}\n'
                     f'Días con datos: {len(dias_con_datos)}\n'
                     f'Promedio mensual: {np.mean(df_mes["ancho"]):.2f} m\n'
                     f'Desv. estándar: {np.std(df_mes["ancho"]):.2f} m')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar
        nombre_archivo = f'boxplot_diario_{año:04d}_{mes_num:02d}_{mes_nombre}.png'
        ruta_completa = os.path.join(carpeta_boxplots, nombre_archivo)
        plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
        plt.close(fig)
        n_individuales += 1
        
        print(f"  Boxplot diario generado: {nombre_archivo}")
    
    # Ahora generar un resumen con todos los meses en subplots, mostrando los boxplots diarios de cada mes
    print("\nGenerando resumen con boxplots diarios de todos los meses (3 columnas, 6 filas)...")
    
    n_meses = len(meses_unicos)
    meses_por_figura = 18  # 3 columnas x 6 filas
    n_figuras = int(np.ceil(n_meses / meses_por_figura))
    n_resumenes = 0
    
    for fig_num in range(n_figuras):
        start_idx = fig_num * meses_por_figura
        end_idx = min(start_idx + meses_por_figura, n_meses)
        meses_figura = meses_unicos[start_idx:end_idx]
        n_meses_fig = len(meses_figura)
        n_cols = 3
        n_rows = int(np.ceil(n_meses_fig / n_cols))
        
        # Tamaño de figura similar al resumen mensual
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        # Aplanar ejes a una lista unidimensional
        if n_rows * n_cols == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
        
        for idx, (ax, mes) in enumerate(zip(axes_list, meses_figura)):
            if idx >= n_meses_fig:
                break
            
            # Datos del mes
            df_mes = df_validos[df_validos['año_mes'] == mes]
            año = int(str(mes).split('-')[0])
            mes_num = int(str(mes).split('-')[1])
            mes_nombre_corto = pd.Timestamp(año, mes_num, 1).strftime('%b')
            
            # Determinar el último día del mes
            ultimo_dia = pd.Timestamp(año, mes_num, 1).days_in_month
            
            # Preparar datos por día para este mes
            data_por_dia = []
            dias_con_datos = []
            for dia in range(1, ultimo_dia + 1):
                anchos_dia = df_mes[df_mes['dia'] == dia]['ancho'].values
                if len(anchos_dia) > 0:
                    data_por_dia.append(anchos_dia)
                    dias_con_datos.append(dia)
            
            if len(data_por_dia) == 0:
                # No hay datos en este mes (no debería ocurrir)
                ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center')
                ax.set_title(f'{mes_nombre_corto} {año}', fontsize=11, fontweight='bold')
                continue
            
            # Boxplot por día en el subplot
            ax.boxplot(data_por_dia, positions=dias_con_datos, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='darkblue', linewidth=0.8),
                       whiskerprops=dict(color='darkblue', linewidth=0.8),
                       capprops=dict(color='darkblue', linewidth=0.8),
                       medianprops=dict(color='red', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.5))
            
            ax.set_title(f'{mes_nombre_corto} {año}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Ancho (m)', fontsize=8)
            
            # Configurar eje X: mostrar todos los días del mes
            ax.set_xticks(range(1, ultimo_dia + 1))
            ax.set_xticklabels([str(d) for d in range(1, ultimo_dia + 1)], rotation=0, fontsize=5)
            ax.set_xlim(0, ultimo_dia + 1)
            
            ax.grid(True, alpha=0.2, linestyle='--', axis='y')
            
            # Texto con estadísticas rápidas (opcional, se puede omitir para no saturar)
            stats_text = f'N={len(df_mes)}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=6,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Ocultar sobrantes
        for ax in axes_list[n_meses_fig:]:
            ax.axis('off')
        
        if n_figuras > 1:
            titulo = f'RESUMEN DE BOXPLOTS DIARIOS (Figura {fig_num+1} de {n_figuras})'
        else:
            titulo = 'RESUMEN DE BOXPLOTS DIARIOS'
        plt.suptitle(titulo, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if n_figuras > 1:
            nombre_resumen = f'resumen_boxplots_diarios_parte{fig_num+1}.png'
        else:
            nombre_resumen = 'resumen_boxplots_diarios.png'
        ruta_resumen = os.path.join(carpeta_boxplots, nombre_resumen)
        plt.savefig(ruta_resumen, dpi=300, bbox_inches='tight')
        plt.close(fig)
        n_resumenes += 1
        
        print(f"  Resumen guardado: {nombre_resumen}")
    
    print(f"\nTodos los boxplots diarios guardados en: {carpeta_boxplots}")
    return (n_individuales, n_resumenes)
# =====================================================================

def generar_grafico_promedio_mensual_global(df, carpeta_resultados):
    """
    Genera un boxplot del ancho para cada mes (todos los años) (Requerimiento 6).
    Retorna True si se generó el gráfico.
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO DE PROMEDIO MENSUAL GLOBAL (BOXPLOT)")
    print("="*80)
    
    # Solo instantes abiertos con ancho > 0
    df_validos = df[(df['estado'] == 1) & (df['ancho'] > 0)].copy()
    if len(df_validos) == 0:
        print("No hay datos válidos para generar el gráfico mensual global.")
        return False
    
    # Crear columna de año-mes
    df_validos['año_mes'] = df_validos['fecha'].dt.to_period('M')
    meses_ordenados = sorted(df_validos['año_mes'].unique())
    
    # Preparar datos para boxplot
    datos_por_mes = []
    etiquetas_meses = []
    for mes in meses_ordenados:
        anchos_mes = df_validos[df_validos['año_mes'] == mes]['ancho'].values
        if len(anchos_mes) > 0:
            datos_por_mes.append(anchos_mes)
            # Formato de etiqueta: Año-Mes (ej. 2023-05)
            etiquetas_meses.append(str(mes))
    
    if not datos_por_mes:
        print("No se pudieron agrupar datos por mes.")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(max(12, len(etiquetas_meses)*0.6), 7))
    
    # Boxplot
    bp = ax.boxplot(datos_por_mes, labels=etiquetas_meses, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
    
    # Leyenda del boxplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Caja: rango intercuartil (Q1-Q3)'),
        plt.Line2D([0], [0], color='red', lw=2, label='Mediana'),
        plt.Line2D([0], [0], color='darkgreen', lw=1, linestyle='-', label='Bigotes (mín/máx sin atípicos)'),
        plt.Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray', markersize=5, linestyle='None', label='Atípicos (puntos grises)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
    ax.set_title('DISTRIBUCIÓN MENSUAL DE ANCHOS (solo instantes abiertos con ancho > 0)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Rotar etiquetas del eje X
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Estadísticas globales
    todos_anchos = df_validos['ancho'].values
    stats_text = (f'Estadísticas globales:\n'
                 f'Total datos: {len(todos_anchos)}\n'
                 f'Promedio global: {np.mean(todos_anchos):.2f} m\n'
                 f'Desv. estándar: {np.std(todos_anchos):.2f} m\n'
                 f'Máximo: {np.max(todos_anchos):.2f} m\n'
                 f'Mínimo: {np.min(todos_anchos):.2f} m')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    nombre_archivo = "Evolucion_promedio_mensual.png"
    ruta_archivo = os.path.join(carpeta_resultados, nombre_archivo)
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Gráfico guardado en: {ruta_archivo}")
    return True

def generar_grafico_evolucion_promedio_mensual(df, carpeta_resultados):
    """
    Genera un gráfico de evolución del promedio mensual del ancho,
    mostrando la curva de promedios mensuales y una banda de desviación estándar.
    (Sin puntos individuales)
    Retorna True si se generó el gráfico.
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO DE EVOLUCIÓN PROMEDIO MENSUAL CON BANDA DE DESVIACIÓN")
    print("="*80)
    
    # Filtrar solo instantes válidos (abiertos con ancho > 0)
    df_validos = df[(df['estado'] == 1) & (df['ancho'] > 0)].copy()
    if len(df_validos) == 0:
        print("No hay datos válidos para generar el gráfico de evolución mensual.")
        return False
    
    # Crear columna de año-mes
    df_validos['año_mes'] = df_validos['fecha'].dt.to_period('M')
    meses_ordenados = sorted(df_validos['año_mes'].unique())
    
    # Preparar datos para el gráfico
    promedios = []
    desviaciones = []
    etiquetas = []
    
    for mes in meses_ordenados:
        df_mes = df_validos[df_validos['año_mes'] == mes]
        anchos_mes = df_mes['ancho'].values
        if len(anchos_mes) > 0:
            promedios.append(np.mean(anchos_mes))
            desviaciones.append(np.std(anchos_mes))
            etiquetas.append(str(mes))   # Formato: YYYY-MM
    
    if not promedios:
        print("No se pudieron calcular promedios mensuales.")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(max(12, len(etiquetas)*0.6), 7))
    
    # Dibujar línea de promedios con marcadores
    x_pos = range(len(etiquetas))
    ax.plot(x_pos, promedios, 'b-o', linewidth=2, markersize=8, 
            label='Promedio mensual', zorder=3)
    
    # Banda de desviación estándar (relleno entre promedio ± desviación)
    prom_array = np.array(promedios)
    desv_array = np.array(desviaciones)
    ax.fill_between(x_pos, prom_array - desv_array, prom_array + desv_array,
                    color='blue', alpha=0.2, label='Desviación estándar', zorder=2)
    
    # Configurar eje X
    ax.set_xticks(x_pos)
    ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
    
    # Título
    ax.set_title('EVOLUCIÓN PROMEDIO MENSUAL DEL ANCHO\n(con banda de desviación estándar)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Cuadrícula
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, marker='o', markersize=8, label='Promedio mensual'),
        Patch(facecolor='blue', alpha=0.2, label='Desviación estándar')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Estadísticas globales
    stats_text = (f'Estadísticas globales (anchos > 0):\n'
                 f'Total datos: {len(df_validos)}\n'
                 f'Promedio global: {np.mean(df_validos["ancho"]):.2f} m\n'
                 f'Desv. estándar: {np.std(df_validos["ancho"]):.2f} m\n'
                 f'Máximo: {np.max(df_validos["ancho"]):.2f} m\n'
                 f'Mínimo: {np.min(df_validos["ancho"]):.2f} m')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    nombre_archivo = "Evolucion_promedio_mensual_con_banda.png"
    ruta_archivo = os.path.join(carpeta_resultados, nombre_archivo)
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Gráfico guardado en: {ruta_archivo}")
    return True

def generar_grafico_barras_estados_mensuales(df, carpeta_resultados):
    """
    Genera un gráfico de barras con la cantidad de estados abierto (verde) y cerrado (rojo)
    por cada mes. Sobre cada barra se muestra el porcentaje correspondiente.
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO DE BARRAS DE ESTADOS MENSUALES")
    print("="*80)
    
    # Copiar y crear columna año-mes
    df_meses = df.copy()
    df_meses['año_mes'] = df_meses['fecha'].dt.to_period('M')
    
    # Agrupar por mes y estado, contar
    grouped = df_meses.groupby(['año_mes', 'estado']).size().unstack(fill_value=0)
    # Asegurar columnas para abierto (1) y cerrado (0)
    if 0 not in grouped.columns:
        grouped[0] = 0
    if 1 not in grouped.columns:
        grouped[1] = 0
    grouped = grouped[[1, 0]]   # orden: abiertos, cerrados
    grouped.columns = ['Abiertos', 'Cerrados']
    
    # Preparar etiquetas de meses (formato YYYY-MM)
    meses = [str(p) for p in grouped.index]
    x = np.arange(len(meses))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(10, len(meses)*0.5), 6))
    
    # Barras
    bars1 = ax.bar(x - width/2, grouped['Abiertos'], width, label='Abiertos', color='green')
    bars2 = ax.bar(x + width/2, grouped['Cerrados'], width, label='Cerrados', color='red')
    
    # Añadir porcentajes sobre las barras
    for i, mes in enumerate(meses):
        total = grouped.loc[mes, 'Abiertos'] + grouped.loc[mes, 'Cerrados']
        if total == 0:
            continue
        
        # Porcentaje de abiertos
        pct_open = (grouped.loc[mes, 'Abiertos'] / total) * 100
        ax.text(x[i] - width/2, grouped.loc[mes, 'Abiertos'], f'{pct_open:.1f}%',
                ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        
        # Porcentaje de cerrados
        pct_closed = (grouped.loc[mes, 'Cerrados'] / total) * 100
        ax.text(x[i] + width/2, grouped.loc[mes, 'Cerrados'], f'{pct_closed:.1f}%',
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    # Configuración de ejes y formato
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Cantidad de instantes', fontsize=12)
    ax.set_title('Cantidad de estados abierto y cerrado por mes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(meses, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    ruta_grafico = os.path.join(carpeta_resultados, 'barras_estados_mensuales.png')
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Gráfico de barras guardado en: {ruta_grafico}")
    return True

def escribir_reporte_final(carpeta_resultados, archivo_anchos, df, config, conteos):
    """
    Escribe un archivo de texto con toda la información y estadísticas generales del proceso.
    """
    ruta_reporte = os.path.join(carpeta_resultados, "reporte_graficos.txt")
    
    # Calcular estadísticas detalladas
    total_instantes = len(df)
    abiertos = np.sum(df['estado'] == 1)
    cerrados = np.sum(df['estado'] == 0)
    df_validos = df[(df['estado'] == 1) & (df['ancho'] > 0)]
    n_validos = len(df_validos)
    
    # Estadísticas por mes
    df['año_mes'] = df['fecha'].dt.to_period('M')
    meses_unicos = sorted(df['año_mes'].unique())
    stats_mensuales = []
    for mes in meses_unicos:
        df_mes = df[df['año_mes'] == mes]
        total_mes = len(df_mes)
        abiertos_mes = np.sum(df_mes['estado'] == 1)
        cerrados_mes = np.sum(df_mes['estado'] == 0)
        df_validos_mes = df_mes[(df_mes['estado'] == 1) & (df_mes['ancho'] > 0)]
        n_validos_mes = len(df_validos_mes)
        if n_validos_mes > 0:
            prom_mes = np.mean(df_validos_mes['ancho'])
            std_mes = np.std(df_validos_mes['ancho'])
            max_mes = np.max(df_validos_mes['ancho'])
            min_mes = np.min(df_validos_mes['ancho'])
        else:
            prom_mes = std_mes = max_mes = min_mes = 0.0
        stats_mensuales.append({
            'mes': str(mes),
            'total': total_mes,
            'abiertos': abiertos_mes,
            'cerrados': cerrados_mes,
            'n_validos': n_validos_mes,
            'prom': prom_mes,
            'std': std_mes,
            'max': max_mes,
            'min': min_mes
        })
    
    # Estadísticas diarias (solo para meses con datos)
    df_validos['dia'] = df_validos['fecha'].dt.day
    df_validos['año_mes'] = df_validos['fecha'].dt.to_period('M')
    stats_diarias = []
    for mes in meses_unicos:
        df_mes = df_validos[df_validos['año_mes'] == mes]
        if len(df_mes) == 0:
            continue
        año = df_mes['fecha'].dt.year.iloc[0]
        mes_num = df_mes['fecha'].dt.month.iloc[0]
        dias_con_datos = sorted(df_mes['dia'].unique())
        for dia in dias_con_datos:
            df_dia = df_mes[df_mes['dia'] == dia]
            stats_diarias.append({
                'mes': str(mes),
                'dia': dia,
                'n': len(df_dia),
                'prom': np.mean(df_dia['ancho']),
                'std': np.std(df_dia['ancho']),
                'max': np.max(df_dia['ancho']),
                'min': np.min(df_dia['ancho'])
            })
    
    # Escribir reporte
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE GENERACIÓN DE GRÁFICOS DE ANCHOS DE DESEMBOCADURA\n")
        f.write("="*80 + "\n")
        f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivo de datos utilizado: {archivo_anchos}\n")
        f.write(f"Parámetros de configuración:\n")
        f.write(f"  • FECHA_INICIO = {config['FECHA_INICIO']}\n")
        f.write(f"  • DIAS_A_GRAFICAR = {config['DIAS_A_GRAFICAR']}\n\n")
        
        # 1. RESUMEN DE DATOS
        f.write("1. RESUMEN DE DATOS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total de instantes analizados: {total_instantes}\n")
        f.write(f"Instantes ABIERTOS (estado=1): {abiertos} ({abiertos/total_instantes*100:.1f}%)\n")
        f.write(f"Instantes CERRADOS (estado=0): {cerrados} ({cerrados/total_instantes*100:.1f}%)\n")
        f.write(f"Instantes con ancho > 0 (válidos para estadísticas): {n_validos}\n\n")
        
        if n_validos > 0:
            f.write("Estadísticas de anchos (solo >0):\n")
            f.write(f"  • Promedio: {np.mean(df_validos['ancho']):.2f} m\n")
            f.write(f"  • Desviación estándar: {np.std(df_validos['ancho']):.2f} m\n")
            f.write(f"  • Máximo: {np.max(df_validos['ancho']):.2f} m\n")
            f.write(f"  • Mínimo: {np.min(df_validos['ancho']):.2f} m\n")
            f.write(f"  • Mediana: {np.median(df_validos['ancho']):.2f} m\n\n")
        
        # 2. GRÁFICOS GENERADOS
        f.write("2. GRÁFICOS GENERADOS\n")
        f.write("-"*80 + "\n")
        f.write(f"Gráficos mensuales individuales: {conteos['mensuales']}\n")
        f.write(f"Gráfico de rango de fechas: {'Sí' if conteos['rango'] else 'No'} (archivo: Evolucion_ancho_{config['DIAS_A_GRAFICAR']}_dias.png)\n")
        f.write(f"Boxplots diarios por mes: {conteos['diarios_individuales']} gráficos individuales (uno por mes), {conteos['diarios_resumen']} resúmenes\n")
        f.write(f"Boxplot mensual global: {'Sí' if conteos['global'] else 'No'} (archivo: Evolucion_promedio_mensual.png)\n")
        f.write(f"Gráfico de evolución promedio mensual con banda de desviación: {'Sí' if conteos['evolucion_promedio'] else 'No'} (archivo: Evolucion_promedio_mensual_con_banda.png)\n\n")
        
        # 3. ESTADÍSTICAS MENSUALES
        f.write("3. ESTADÍSTICAS MENSUALES (solo anchos > 0)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Mes':12} | {'Total':8} | {'Abiertos':8} | {'Cerrados':8} | {'N válidos':8} | {'Prom (m)':8} | {'Std (m)':8} | {'Max (m)':8} | {'Min (m)':8}\n")
        f.write("-"*105 + "\n")
        for s in stats_mensuales:
            f.write(f"{s['mes']:12} | {s['total']:8} | {s['abiertos']:8} | {s['cerrados']:8} | {s['n_validos']:8} | {s['prom']:8.2f} | {s['std']:8.2f} | {s['max']:8.2f} | {s['min']:8.2f}\n")
        f.write("\n")
        
        # 4. ESTADÍSTICAS DIARIAS (por mes)
        f.write("4. ESTADÍSTICAS DIARIAS (solo días con datos, anchos > 0)\n")
        f.write("-"*80 + "\n")
        current_mes = None
        for s in stats_diarias:
            if s['mes'] != current_mes:
                current_mes = s['mes']
                f.write(f"\n  {current_mes}:\n")
            f.write(f"    Día {s['dia']:2d}: n={s['n']:3d}, prom={s['prom']:6.2f} m, std={s['std']:5.2f}, max={s['max']:6.2f}, min={s['min']:6.2f}\n")
        f.write("\n")
        
        # 5. INFORMACIÓN ADICIONAL
        f.write("5. INFORMACIÓN ADICIONAL\n")
        f.write("-"*80 + "\n")
        f.write(f"Los gráficos mensuales se encuentran en la subcarpeta 'Graficas mensuales'.\n")
        f.write(f"Los boxplots diarios por mes se encuentran en la subcarpeta 'boxplots_diarios_mensuales'.\n")
        f.write("="*80 + "\n")
    
    print(f"\nReporte final guardado en: {ruta_reporte}")

def main():
    """
    Función principal que ejecuta todas las generaciones de gráficos.
    """
    print("="*80)
    print("PROCESADOR COMPLETO DE GRÁFICOS DE ANCHOS")
    print("="*80)
    
    # Configurar rutas
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    
    # Buscar el archivo más reciente de anchos
    import glob
    patron_archivos = os.path.join(carpeta_base, 'Resultados_Ancho_Canal_*', 'Data_Ancho_Desde_Limites_*_limpio.nc')
    archivos_encontrados = glob.glob(patron_archivos)
    
    if not archivos_encontrados:
        print("ERROR: No se encontraron archivos de anchos.")
        print(f"Buscando en: {patron_archivos}")
        return
    
    # Seleccionar el archivo más reciente
    archivo_anchos = max(archivos_encontrados, key=os.path.getctime)
    print(f"Archivo de anchos encontrado: {archivo_anchos}")
    
    # Obtener carpeta de resultados del archivo seleccionado
    carpeta_resultados = os.path.dirname(archivo_anchos)
    print(f"Carpeta de resultados: {carpeta_resultados}")
    
    # Abrir dataset para extraer DataFrame (usado en funciones posteriores)
    try:
        ds = xr.open_dataset(archivo_anchos)
        tiempos = ds.time.values
        estados = ds.estado_apertura.values
        anchos = ds.ancho_desembocadura.values
        df = pd.DataFrame({
            'fecha': pd.to_datetime(tiempos),
            'estado': estados,
            'ancho': anchos
        }).sort_values('fecha')
        ds.close()
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return
    
    # Inicializar diccionario de conteos
    conteos = {
        'mensuales': 0,
        'rango': False,
        'diarios_individuales': 0,
        'diarios_resumen': 0,
        'global': False,
        'evolucion_promedio': False
    }
    
    # 1. Generar gráficos mensuales
    conteos['mensuales'] = generar_graficos_mensuales_anchos(archivo_anchos, carpeta_resultados)
    
    # 2. Gráfico de rango de fechas (único gráfico con todos los datos de los N días)
    conteos['rango'] = generar_grafico_rango_fechas(df, carpeta_resultados, FECHA_INICIO, DIAS_A_GRAFICAR)
    
    # 3. Boxplots diarios por mes (un gráfico por mes con boxplots por día)
    diarios = generar_boxplots_diarios_mensuales(df, carpeta_resultados)
    conteos['diarios_individuales'], conteos['diarios_resumen'] = diarios
    
    # 4. Gráfico de promedio mensual global (boxplot)
    conteos['global'] = generar_grafico_promedio_mensual_global(df, carpeta_resultados)
    
    # 5. Gráfico de evolución promedio mensual con banda (sin puntos individuales)
    conteos['evolucion_promedio'] = generar_grafico_evolucion_promedio_mensual(df, carpeta_resultados)

    # Nuevo gráfico de barras de estados mensuales
    generar_grafico_barras_estados_mensuales(df, carpeta_resultados)
    
    # 6. Escribir reporte final con estadísticas
    config = {
        'FECHA_INICIO': FECHA_INICIO,
        'DIAS_A_GRAFICAR': DIAS_A_GRAFICAR
    }
    escribir_reporte_final(carpeta_resultados, archivo_anchos, df, config, conteos)
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)

if __name__ == "__main__":
    main()