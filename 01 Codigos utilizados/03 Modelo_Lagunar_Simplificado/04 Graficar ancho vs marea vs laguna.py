#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grafica marea (UTide), ancho de la boca (desde imágenes) y nivel de laguna simulado
(resultados del modelo) en una sola figura con doble eje Y.
- Marea: línea roja continua
- Nivel laguna: línea azul con gaps
- Ancho: puntos verdes (sin línea) con gaps
Eje X con formato en una sola línea (sin saltos)
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# CONFIGURACIÓN DE RUTAS (MODIFICAR SEGÚN CORRESPONDA)
# =============================================================================
# Archivo de anchos (desde procesamiento de imágenes)
RUTA_ANCHOS_NETCDF = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\01 Lee_imagen\Resultados_Ancho_Canal_20240710_ROI1-FINAL\Data_Ancho_Desde_Limites_20240710_limpio.nc"

# Archivo de marea reconstruida (UTide)
RUTA_MAREA_NETCDF = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\02 Mareas\marea_reconstruida_utide_2023-2024.nc"

# Archivo de resultados de la simulación (nivel laguna)
RUTA_SIMULACION_NC = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Resultados_modelo_lagunar_variable-FINAL_2024-07\resultados_simulacion_variable_FINAL_2024-07.nc"

# Carpeta de salida
CARPETA_SALIDA = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Resultados_modelo_lagunar_variable-FINAL_2024-07"
NOMBRE_FIG = "marea_nivel_ancho_combinado.png"

# Crear carpeta de salida si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# =============================================================================
# FUNCIÓN AUXILIAR PARA INSERTAR GAPS
# =============================================================================
def insert_nan_at_gaps(times, values, gap_threshold_minutes=30):
    """
    Inserta NaN en values cuando el gap entre tiempos consecutivos
    supera gap_threshold_minutes.
    Retorna listas (nuevos_tiempos, nuevos_valores) listas para plotear.
    """
    if len(times) == 0:
        return times, values
    times_arr = np.array(times)
    values_arr = np.array(values)
    
    gap_threshold = gap_threshold_minutes  # minutos
    new_times = [times_arr[0]]
    new_vals = [values_arr[0]]
    
    for i in range(1, len(times_arr)):
        gap = (times_arr[i] - times_arr[i-1]) / np.timedelta64(1, 'm')
        if gap > gap_threshold:
            new_times.append(times_arr[i-1] + np.timedelta64(int(gap_threshold), 'm'))
            new_vals.append(np.nan)
            new_times.append(times_arr[i] - np.timedelta64(int(gap_threshold), 'm'))
            new_vals.append(np.nan)
        new_times.append(times_arr[i])
        new_vals.append(values_arr[i])
    
    return new_times, new_vals

# =============================================================================
# CARGAR DATOS DE MAREA
# =============================================================================
print("Cargando datos de marea desde:", RUTA_MAREA_NETCDF)
ds_marea = xr.open_dataset(RUTA_MAREA_NETCDF)
if 'marea' in ds_marea.variables:
    marea = ds_marea.marea.values
else:
    # Buscar la primera variable unidimensional con dimensión time
    var_candidates = [v for v in ds_marea.variables if 'time' in ds_marea[v].dims and len(ds_marea[v].dims) == 1]
    if len(var_candidates) == 0:
        raise ValueError("No se encontró variable de marea en el archivo.")
    marea = ds_marea[var_candidates[0]].values
    print(f"   Usando variable '{var_candidates[0]}' como marea.")
tiempo_marea = ds_marea.time.values
ds_marea.close()

# Convertir tiempo a datetime
tiempo_marea_dt = pd.to_datetime(tiempo_marea)
print(f"   Rango marea: {tiempo_marea_dt.min()} a {tiempo_marea_dt.max()}  ({len(tiempo_marea_dt)} puntos)")

# =============================================================================
# CARGAR DATOS DE ANCHOS
# =============================================================================
print("\nCargando datos de anchos desde:", RUTA_ANCHOS_NETCDF)
ds_anchos = xr.open_dataset(RUTA_ANCHOS_NETCDF)
if 'ancho_desembocadura' not in ds_anchos.variables:
    raise ValueError("El archivo de anchos no contiene 'ancho_desembocadura'.")
ancho = ds_anchos.ancho_desembocadura.values
tiempo_anchos = ds_anchos.time.values
# También cargar estado si se desea (no se usará en gráfico, solo informativo)
estado = ds_anchos.estado_apertura.values if 'estado_apertura' in ds_anchos else None
ds_anchos.close()

tiempo_anchos_dt = pd.to_datetime(tiempo_anchos)
print(f"   Rango anchos: {tiempo_anchos_dt.min()} a {tiempo_anchos_dt.max()}  ({len(tiempo_anchos_dt)} puntos)")

# =============================================================================
# CARGAR DATOS DE SIMULACIÓN (NIVEL LAGUNA)
# =============================================================================
print("\nCargando datos de simulación desde:", RUTA_SIMULACION_NC)
ds_sim = xr.open_dataset(RUTA_SIMULACION_NC)
if 'nivel_laguna' not in ds_sim.variables:
    raise ValueError("El archivo de simulación no contiene 'nivel_laguna'.")
nivel_laguna = ds_sim.nivel_laguna.values
tiempo_sim = ds_sim.time.values
# Obtener paso de tiempo para umbral de gaps
dt_minutes = ds_sim.attrs.get('dt_minutes', 0.25)
ds_sim.close()

tiempo_sim_dt = pd.to_datetime(tiempo_sim)
print(f"   Rango simulación: {tiempo_sim_dt.min()} a {tiempo_sim_dt.max()}  ({len(tiempo_sim_dt)} puntos)")
print(f"   Paso temporal simulación: {dt_minutes} min")

# =============================================================================
# INSERTAR GAPS EN NIVEL LAGUNA Y EN ANCHO (para puntos)
# =============================================================================
gap_threshold = 2 * dt_minutes
t_nivel, nivel_gap = insert_nan_at_gaps(tiempo_sim_dt, nivel_laguna,
                                        gap_threshold_minutes=gap_threshold)
t_ancho, ancho_gap = insert_nan_at_gaps(tiempo_anchos_dt, ancho,
                                        gap_threshold_minutes=gap_threshold)

# =============================================================================
# GENERAR GRÁFICA CON DOBLE EJE Y
# =============================================================================
fig, ax1 = plt.subplots(figsize=(16, 8))

# Eje izquierdo: marea y nivel laguna
color_marea = 'tab:red'
color_nivel = 'tab:blue'
ax1.set_xlabel('Tiempo', fontsize=12)
ax1.set_ylabel('Nivel (m)', color='black', fontsize=12)
ax1.plot(tiempo_marea_dt, marea, color=color_marea, linewidth=1.5,
         label='Marea reconstruida (UTide)', alpha=0.9)
ax1.plot(t_nivel, nivel_gap, color=color_nivel, linewidth=1.5,
         label='Nivel laguna simulado', alpha=0.9)
ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Eje derecho: ancho del canal como puntos verdes (sin línea)
ax2 = ax1.twinx()
color_ancho = 'green'  # puntos verdes
ax2.set_ylabel('Ancho (m)', color='black', fontsize=12)
ax2.plot(t_ancho, ancho_gap, 'o', color=color_ancho, markersize=6,
         label='Ancho de la boca (imágenes)', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='black', labelsize=10)

# Formato del eje de tiempo en una sola línea
locator = mdates.HourLocator(interval=24)
formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate(rotation=45)

# Leyenda combinada dentro de la gráfica (esquina superior derecha)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
           fontsize=10, frameon=True)

# Limitar el eje X al período de simulación
ax1.set_xlim([tiempo_sim_dt.min(), tiempo_sim_dt.max()])

plt.title('Marea, nivel de laguna simulado y ancho de la boca\n(período de simulación)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Guardar figura
ruta_fig = os.path.join(CARPETA_SALIDA, NOMBRE_FIG)
plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
print(f"\nGráfico guardado en: {ruta_fig}")

plt.show()