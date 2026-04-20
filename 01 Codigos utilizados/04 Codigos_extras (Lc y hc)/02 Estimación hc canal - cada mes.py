#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de la altura de canal (hc) a partir de anchos de desembocadura y datos de marea.
Genera análisis mensual y archivo de resultados detallado.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN DE RUTAS (MODIFICAR SEGÚN CORRESPONDA)
# =============================================================================
RUTA_ANCHOS_NETCDF = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\01 Lee_imagen\Resultados_Ancho_Canal_20240710_ROI1-FINAL\Data_Ancho_Desde_Limites_20240710_limpio.nc"
RUTA_MAREA_NETCDF = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\02 Mareas\marea_reconstruida_utide_2023-2024.nc"
CARPETA_SALIDA = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\06 Codigos extras (Lc y hc)\Resultados Altura Canal hc mensual"
NOMBRE_TXT = "analisis_hc_mensual_detallado.txt"
NOMBRE_FIG = "marea_ancho_intervalos.png"
NOMBRE_FIG_7DIAS = "marea_ancho_intervalos_7dias.png"   # gráfico de 7 días con gaps

# Crear carpeta de salida si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# =============================================================================
# FUNCIÓN PARA CARGAR DATOS DE MAREA (adaptada del código proporcionado)
# =============================================================================
def cargar_datos_marea(archivo):
    """
    Carga datos de marea desde un archivo NetCDF.
    Busca variables como 'marea', 'altura_cahuil_estimada' o la primera
    variable unidimensional con dimensión 'time'.
    Retorna:
        tiempo_marea : np.ndarray de datetime64
        marea        : np.ndarray de alturas (float)
    """
    try:
        ds = xr.open_dataset(archivo)
        
        # Determinar qué variable usar para la altura de marea
        if 'marea' in ds.variables:
            marea_original = ds.marea.values
            variable_usada = 'marea'
        elif 'altura_cahuil_estimada' in ds.variables:
            marea_original = ds.altura_cahuil_estimada.values
            variable_usada = 'altura_cahuil_estimada'
            print("   ADVERTENCIA: Usando 'altura_cahuil_estimada' en lugar de 'marea'.")
        else:
            # Buscar la primera variable que tenga dimensión 'time' y sea unidimensional
            var_encontrada = None
            for var_name in ds.variables:
                if 'time' in ds[var_name].dims and len(ds[var_name].dims) == 1:
                    marea_original = ds[var_name].values
                    variable_usada = var_name
                    var_encontrada = True
                    print(f"   ADVERTENCIA: Usando variable '{var_name}' como marea.")
                    break
            if not var_encontrada:
                raise ValueError("No se encontró una variable adecuada de marea en el archivo.")
        
        # Obtener el tiempo (asumiendo que está en segundos desde 1970-01-01)
        tiempo_segundos = ds.time.values
        # Convertir a datetime64[ns] (pandas lo maneja bien)
        tiempo_dt = pd.to_datetime(tiempo_segundos, unit='s').to_numpy().astype('datetime64[ns]')
        
        if len(tiempo_dt) < 2:
            raise ValueError(f"El archivo de marea tiene solo {len(tiempo_dt)} punto(s). Se necesitan al menos 2.")
        
        ds.close()
        
        print(f"   Datos de marea cargados:")
        print(f"    • Registros: {len(tiempo_dt)}")
        print(f"    • Rango temporal: {pd.to_datetime(tiempo_dt[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(tiempo_dt[-1]).strftime('%Y-%m-%d %H:%M')}")
        print(f"    • Rango de marea: {marea_original.min():.2f} a {marea_original.max():.2f} m")
        print(f"    • Variable usada: {variable_usada}")
        
        return tiempo_dt, marea_original
    
    except Exception as e:
        print(f"   ERROR cargando datos de marea: {e}")
        raise

# =============================================================================
# FUNCIÓN PARA DETECTAR INTERVALOS DE CANAL ABIERTO
# =============================================================================
def encontrar_intervalos(estado):
    """
    Encuentra índices de inicio y fin de cada segmento continuo donde estado == 1.
    Retorna lista de tuplas (inicio, fin) con índices inclusivos.
    """
    intervalos = []
    i = 0
    n = len(estado)
    while i < n:
        if estado[i] == 1:
            inicio = i
            while i < n and estado[i] == 1:
                i += 1
            fin = i - 1
            intervalos.append((inicio, fin))
        else:
            i += 1
    return intervalos

# =============================================================================
# FUNCIÓN AUXILIAR PARA PLOTEAR CON GAPS
# =============================================================================
def plot_with_gaps(ax, x, y, **kwargs):
    """
    Plotea una serie que puede contener NaN, dibujando segmentos continuos
    solo donde no hay NaN.
    """
    # Convertir a arrays de numpy y crear máscara de valores válidos
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(y)
    if not np.any(mask):
        return
    # Encontrar índices donde comienzan y terminan segmentos válidos
    idx = np.where(mask)[0]
    # Buscar cambios consecutivos
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, breaks)
    for seg in segments:
        if len(seg) > 0:
            ax.plot(x[seg], y[seg], **kwargs)

# =============================================================================
# 1. CARGAR DATOS DE ANCHOS
# =============================================================================
print("Cargando archivo de anchos...")
ds_anchos = xr.open_dataset(RUTA_ANCHOS_NETCDF)

# Verificar que las variables necesarias existen
if 'time' not in ds_anchos:
    raise ValueError("El archivo de anchos no contiene la variable 'time'.")
if 'estado_apertura' not in ds_anchos:
    raise ValueError("El archivo de anchos no contiene la variable 'estado_apertura'.")

tiempos_anchos = ds_anchos['time'].values          # datetime64[ns] normalmente
estado = ds_anchos['estado_apertura'].values       # 0/1
# Opcional: cargar ancho si existe para graficar
ancho = ds_anchos['ancho_desembocadura'].values if 'ancho_desembocadura' in ds_anchos else None

ds_anchos.close()

print(f"   Datos de anchos cargados:")
print(f"    • Registros: {len(tiempos_anchos)}")
print(f"    • Rango temporal: {pd.to_datetime(tiempos_anchos[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(tiempos_anchos[-1]).strftime('%Y-%m-%d %H:%M')}")
print(f"    • Estado abierto: {np.sum(estado==1)} instantes")
print(f"    • Estado cerrado: {np.sum(estado==0)} instantes")

# =============================================================================
# 2. CARGAR DATOS DE MAREA E INTERPOLAR A LOS TIEMPOS DE ANCHOS
# =============================================================================
print("\nCargando datos de marea...")
tiempo_marea, marea_original = cargar_datos_marea(RUTA_MAREA_NETCDF)

# Crear interpolador lineal
# Convertir tiempos a segundos desde epoch para interp1d (valores numéricos)
# Usamos pandas para convertir a timestamp en segundos (puede ser int64)
tiempo_marea_num = tiempo_marea.astype('datetime64[s]').astype('int64')
tiempos_anchos_num = tiempos_anchos.astype('datetime64[s]').astype('int64')

interpolador = interp1d(tiempo_marea_num, marea_original,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(marea_original[0], marea_original[-1]))

# Interpolar marea a los tiempos de ancho
marea_interp = interpolador(tiempos_anchos_num)

print(f"\n   Marea interpolada a {len(tiempos_anchos)} instantes de ancho.")
print(f"   Rango de marea interpolada: {marea_interp.min():.2f} a {marea_interp.max():.2f} m")

# =============================================================================
# 3. DETECTAR INTERVALOS DE CANAL ABIERTO (estado == 1)
# =============================================================================
intervalos_idx = encontrar_intervalos(estado)
print(f"\nSe encontraron {len(intervalos_idx)} intervalos de canal abierto.")

# =============================================================================
# 4. CALCULAR AMPLITUDES EN CADA INTERVALO Y ALMACENAR DETALLES
# =============================================================================
resultados_intervalos = []  # lista de diccionarios

for idx, (i_inicio, i_fin) in enumerate(intervalos_idx):
    # Extraer marea interpolada en el intervalo
    marea_intervalo = marea_interp[i_inicio:i_fin+1]
    
    if len(marea_intervalo) >= 2:
        max_m = np.max(marea_intervalo)
        min_m = np.min(marea_intervalo)
        amplitud = max_m - min_m
    else:
        # Si solo un punto, la amplitud es 0
        amplitud = 0.0
        max_m = min_m = marea_intervalo[0]
    
    # Tiempos de inicio y fin
    t_inicio = tiempos_anchos[i_inicio]
    t_fin = tiempos_anchos[i_fin]
    duracion_h = (t_fin - t_inicio) / np.timedelta64(1, 'h')
    
    # Obtener mes (año-mes) para agrupación
    mes_str = pd.to_datetime(t_inicio).strftime('%Y-%m')
    
    resultados_intervalos.append({
        'intervalo_id': idx + 1,
        'mes': mes_str,
        'inicio': t_inicio,
        'fin': t_fin,
        'duracion_h': duracion_h,
        'amplitud_m': amplitud,
        'max_marea': max_m,
        'min_marea': min_m,
        'i_inicio': i_inicio,
        'i_fin': i_fin
    })

# =============================================================================
# 5. AGRUPAR POR MES Y CALCULAR ESTADÍSTICAS MENSUALES (EXCLUYENDO AMPLITUD CERO)
# =============================================================================
# Convertir a DataFrame
df = pd.DataFrame(resultados_intervalos)

# Filtrar solo amplitudes positivas para las estadísticas mensuales
df_pos = df[df['amplitud_m'] > 0].copy()

# Agrupar por mes (usando el DataFrame filtrado)
grupos_mensuales = df_pos.groupby('mes')

resumen_mensual = []
for mes, grupo in grupos_mensuales:
    amps = grupo['amplitud_m'].values
    promedio = np.mean(amps)
    desviacion = np.std(amps, ddof=1) if len(amps) > 1 else 0.0
    max_amp = np.max(amps)
    min_amp = np.min(amps)
    n_int_pos = len(grupo)
    
    # Información del intervalo con máxima amplitud en el mes
    idx_max = np.argmax(amps)
    intervalo_max = grupo.iloc[idx_max]
    
    # También queremos el total de intervalos del mes (incluyendo ceros) para contexto
    total_int_mes = len(df[df['mes'] == mes])
    
    resumen_mensual.append({
        'mes': mes,
        'promedio_hc_m': promedio,
        'desviacion_m': desviacion,
        'max_amp_m': max_amp,
        'min_amp_m': min_amp,
        'n_int_pos': n_int_pos,
        'total_int_mes': total_int_mes,
        'suma_duracion_h': grupo['duracion_h'].sum(),  # suma de horas de intervalos positivos
        'fecha_max_amp': pd.to_datetime(intervalo_max['inicio']).strftime('%Y-%m-%d %H:%M'),
        'amplitud_max_m': intervalo_max['amplitud_m']
    })

# Ordenar por mes
resumen_mensual = sorted(resumen_mensual, key=lambda x: x['mes'])

# =============================================================================
# 6. GUARDAR RESULTADOS EN ARCHIVO TXT
# =============================================================================
ruta_txt = os.path.join(CARPETA_SALIDA, NOMBRE_TXT)
with open(ruta_txt, 'w', encoding='utf-8') as f:
    f.write("ANÁLISIS DE ALTURA DE CANAL (hc) POR MES\n")
    f.write("========================================\n\n")
    f.write(f"Archivo de anchos: {RUTA_ANCHOS_NETCDF}\n")
    f.write(f"Archivo de marea: {RUTA_MAREA_NETCDF}\n")
    f.write(f"Total de intervalos detectados: {len(intervalos_idx)}\n")
    f.write(f"Intervalos con amplitud > 0: {len(df_pos)} (de {len(df)})\n\n")
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1: DETALLE POR INTERVALO (TODOS, INCLUYENDO CERO)
    # -------------------------------------------------------------------------
    f.write("DETALLE DE CADA INTERVALO DE APERTURA\n")
    f.write("-" * 120 + "\n")
    encabezado = (f"{'#':<4} {'Mes':<8} {'Inicio':<20} {'Fin':<20} "
                  f"{'Dur (h)':<8} {'Amplitud (m)':<12} {'Max marea':<10} {'Min marea':<10}\n")
    f.write(encabezado)
    f.write("-" * 120 + "\n")
    
    for r in resultados_intervalos:
        inicio_str = pd.to_datetime(r['inicio']).strftime('%Y-%m-%d %H:%M')
        fin_str = pd.to_datetime(r['fin']).strftime('%Y-%m-%d %H:%M')
        f.write(f"{r['intervalo_id']:<4} {r['mes']:<8} {inicio_str:<20} {fin_str:<20} "
                f"{r['duracion_h']:<8.2f} {r['amplitud_m']:<12.4f} {r['max_marea']:<10.4f} {r['min_marea']:<10.4f}\n")
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2: RESUMEN MENSUAL (SOLO AMPLITUDES > 0)
    # -------------------------------------------------------------------------
    f.write("\n" + "="*120 + "\n")
    f.write("RESUMEN MENSUAL (hc = amplitud promedio por mes, calculado solo con amplitudes > 0)\n")
    f.write("-" * 120 + "\n")
    encabezado2 = (f"{'Mes':<8} {'hc prom (m)':<12} {'Desv (m)':<10} {'Max (m)':<10} "
                   f"{'Min (m)':<10} {'N_int_pos':<9} {'Total_int':<9} {'Suma dur (h)':<12} {'Fecha max amp':<20}\n")
    f.write(encabezado2)
    f.write("-" * 120 + "\n")
    
    for mes in resumen_mensual:
        f.write(f"{mes['mes']:<8} {mes['promedio_hc_m']:<12.4f} {mes['desviacion_m']:<10.4f} "
                f"{mes['max_amp_m']:<10.4f} {mes['min_amp_m']:<10.4f} {mes['n_int_pos']:<9} "
                f"{mes['total_int_mes']:<9} {mes['suma_duracion_h']:<12.2f} {mes['fecha_max_amp']:<20}\n")
    
    f.write("="*120 + "\n")

print(f"\nResultados guardados en: {ruta_txt}")

# =============================================================================
# 7. (OPCIONAL) GENERAR GRÁFICA DE MAREA Y ANCHO CON INTERVALOS
# =============================================================================
try:
    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Eje izquierdo: marea
    color_marea = 'tab:blue'
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Nivel de marea (m)', color=color_marea)
    ax1.plot(tiempos_anchos, marea_interp, color=color_marea, linewidth=1, label='Marea interpolada')
    ax1.tick_params(axis='y', labelcolor=color_marea)

    # Eje derecho: ancho del canal (si está disponible)
    if ancho is not None:
        ax2 = ax1.twinx()
        color_ancho = 'tab:red'
        ax2.set_ylabel('Ancho del canal (m)', color=color_ancho)
        ax2.plot(tiempos_anchos, ancho, color=color_ancho, linewidth=1, label='Ancho', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color_ancho)
    else:
        ax2 = None

    # Sombrear intervalos de canal abierto y marcar puntos de max/min
    for r in resultados_intervalos:
        t0 = r['inicio']
        t1 = r['fin']
        ax1.axvspan(t0, t1, alpha=0.2, color='green', label='Canal abierto' if r['intervalo_id']==1 else "")
        # Marcar puntos de máximo y mínimo dentro del intervalo
        i_ini = r['i_inicio']
        i_fin = r['i_fin']
        # Encontrar índices locales de max y min
        idx_max_local = i_ini + np.argmax(marea_interp[i_ini:i_fin+1])
        idx_min_local = i_ini + np.argmin(marea_interp[i_ini:i_fin+1])
        ax1.plot(tiempos_anchos[idx_max_local], marea_interp[idx_max_local], 'ro', markersize=3)
        ax1.plot(tiempos_anchos[idx_min_local], marea_interp[idx_min_local], 'bo', markersize=3)
        # Opcional: línea punteada uniendo max y min
        ax1.plot([tiempos_anchos[idx_max_local], tiempos_anchos[idx_min_local]],
                 [marea_interp[idx_max_local], marea_interp[idx_min_local]],
                 'k--', linewidth=0.5, alpha=0.5)

    # Formato del eje de tiempo
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Ajustable
    fig.autofmt_xdate()

    # Título y leyenda
    plt.title('Marea y ancho del canal con intervalos de apertura')
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.tight_layout()
    ruta_fig = os.path.join(CARPETA_SALIDA, NOMBRE_FIG)
    plt.savefig(ruta_fig, dpi=150)
    plt.show()
    print(f"Gráfica guardada en: {ruta_fig}")
except Exception as e:
    print(f"No se pudo generar la gráfica: {e}")

# =============================================================================
# 8. GRÁFICA DE DETALLE DE 7 DÍAS (3.5 días antes y después) CON GAPS EN AMBAS CURVAS
# =============================================================================
try:
    # Seleccionar el intervalo con mayor amplitud para centrar la ventana
    if resultados_intervalos:
        # Encontrar el intervalo de mayor amplitud (positiva o cero)
        intervalo_destacado = max(resultados_intervalos, key=lambda x: x['amplitud_m'])
        # Calcular punto medio del intervalo
        t_centro = intervalo_destacado['inicio'] + (intervalo_destacado['fin'] - intervalo_destacado['inicio']) / 2
        # Ventana de 7 días: 3.5 días antes y 3.5 días después del centro
        start_win = t_centro - np.timedelta64(3, 'D') - np.timedelta64(12, 'h')   # 3.5 días
        end_win   = t_centro + np.timedelta64(3, 'D') + np.timedelta64(12, 'h')
        # Ajustar para que no salga del rango total de datos
        start_win = max(start_win, tiempos_anchos[0])
        end_win   = min(end_win, tiempos_anchos[-1])
    else:
        # Si no hay intervalos, usar el primer y último dato (7 días)
        start_win = tiempos_anchos[0]
        end_win   = start_win + np.timedelta64(7, 'D')
        if end_win > tiempos_anchos[-1]:
            end_win = tiempos_anchos[-1]
            start_win = end_win - np.timedelta64(7, 'D')
            start_win = max(start_win, tiempos_anchos[0])

    # Obtener máscara booleana para el subconjunto temporal
    mask = (tiempos_anchos >= start_win) & (tiempos_anchos <= end_win)
    tiempos_sub = tiempos_anchos[mask]
    marea_sub = marea_interp[mask]
    if ancho is not None:
        ancho_sub = ancho[mask]
    else:
        ancho_sub = None

    # Crear figura para el zoom
    fig2, ax1_sub = plt.subplots(figsize=(12, 6))

    # Eje izquierdo: marea (con gaps, solo donde ancho es válido)
    ax1_sub.set_xlabel('Tiempo')
    ax1_sub.set_ylabel('Nivel de marea (m)', color=color_marea)
    if ancho_sub is not None:
        # Usar la máscara de ancho válido para la marea
        valid_mask = ~np.isnan(ancho_sub)
        if np.any(valid_mask):
            plot_with_gaps(ax1_sub, tiempos_sub[valid_mask], marea_sub[valid_mask],
                           color=color_marea, linewidth=1.5, label='Marea interpolada')
    else:
        # Si no hay ancho, mostrar la marea completa (sin gaps)
        ax1_sub.plot(tiempos_sub, marea_sub, color=color_marea, linewidth=1.5, label='Marea interpolada')
    ax1_sub.tick_params(axis='y', labelcolor=color_marea)

    # Eje derecho: ancho (con gaps)
    if ancho_sub is not None:
        ax2_sub = ax1_sub.twinx()
        ax2_sub.set_ylabel('Ancho del canal (m)', color=color_ancho)
        plot_with_gaps(ax2_sub, tiempos_sub, ancho_sub, color=color_ancho, linewidth=1.5, alpha=0.7, label='Ancho')
        ax2_sub.tick_params(axis='y', labelcolor=color_ancho)
    else:
        ax2_sub = None

    # Sombrear solo los intervalos que se superpongan con la ventana
    for r in resultados_intervalos:
        t0 = r['inicio']
        t1 = r['fin']
        # Calcular la intersección del intervalo con la ventana
        t0_intersect = max(t0, start_win)
        t1_intersect = min(t1, end_win)
        if t0_intersect < t1_intersect:  # Hay superposición
            ax1_sub.axvspan(t0_intersect, t1_intersect, alpha=0.2, color='green', 
                            label='Canal abierto' if r['intervalo_id']==1 else "")
            # Marcar máximos y mínimos dentro de la intersección
            # Encontrar índices originales que caen dentro de la ventana para este intervalo
            i_ini = r['i_inicio']
            i_fin = r['i_fin']
            # Buscar dentro de ese rango los índices que también están en la ventana
            idxs_en_ventana = np.where(mask)[0]
            idxs_en_intervalo = np.arange(i_ini, i_fin+1)
            idxs_comunes = np.intersect1d(idxs_en_ventana, idxs_en_intervalo)
            if len(idxs_comunes) > 0:
                # Encontrar el máximo y mínimo de marea en esos índices comunes
                marea_comun = marea_interp[idxs_comunes]
                idx_max_local = idxs_comunes[np.argmax(marea_comun)]
                idx_min_local = idxs_comunes[np.argmin(marea_comun)]
                ax1_sub.plot(tiempos_anchos[idx_max_local], marea_interp[idx_max_local], 'ro', markersize=4)
                ax1_sub.plot(tiempos_anchos[idx_min_local], marea_interp[idx_min_local], 'bo', markersize=4)
                # Línea punteada entre ellos (solo si ambos están en la ventana)
                ax1_sub.plot([tiempos_anchos[idx_max_local], tiempos_anchos[idx_min_local]],
                             [marea_interp[idx_max_local], marea_interp[idx_min_local]],
                             'k--', linewidth=0.8, alpha=0.6)

    # Formato del eje temporal para 7 días (mostrar horas)
    ax1_sub.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1_sub.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig2.autofmt_xdate()

    # Título y leyenda
    plt.title(f'Detalle de 7 días ({pd.to_datetime(start_win).strftime("%Y-%m-%d %H:%M")} a {pd.to_datetime(end_win).strftime("%Y-%m-%d %H:%M")})')
    lines1, labels1 = ax1_sub.get_legend_handles_labels()
    if ax2_sub:
        lines2, labels2 = ax2_sub.get_legend_handles_labels()
        ax1_sub.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1_sub.legend(loc='upper left')

    plt.tight_layout()
    ruta_fig_zoom = os.path.join(CARPETA_SALIDA, NOMBRE_FIG_7DIAS)
    plt.savefig(ruta_fig_zoom, dpi=150)
    plt.show()
    print(f"Gráfica de detalle (7 días) guardada en: {ruta_fig_zoom}")
except Exception as e:
    print(f"No se pudo generar la gráfica de 7 días: {e}")

print("\nProceso completado.")