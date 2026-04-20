import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
archivo_medido = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Nivel Laguna NMM - data completa 2023-2024\Nivel_laguna_ref_NMM.nc"
carpeta_simulaciones = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Laguna Simulados"
archivo_marea = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Marea reconstruida\marea_reconstruida_utide_2023-2024.nc"
carpeta_salida_base = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados_Errores_Completos"

# Archivo de referencia cuyos tiempos se usarán como los intervalos con datos de ancho
archivo_referencia_tiempos_nombre = "resultados_simulacion_variable_FINAL_2024-07.nc"

# Ruta del archivo de anchos reales
RUTA_ANCHOS_NETCDF = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\01 Lee_imagen\Resultados_Ancho_Canal_20240710_ROI1-FINAL\Data_Ancho_Desde_Limites_20240710_limpio.nc"

# Mapeo de nombres de archivo a etiquetas de simulación
mapeo_simulaciones = {
    "resultados_simulacion_constante_FINAL_2024-07": "Simulación 1a (constante continua)",
    "resultados_simulacion_constante_intervalos_FINAL_2024-07": "Simulación 1b (constante intervalos)",
    "resultados_simulacion_variable_FINAL_2024-07": "Simulación 2 (variable 2024-07)",
    "resultados_simulacion_variable_FINAL_2023-05_Qr_cte": "Simulación 3 (variable 2023-05 Qr cte)"
}

# Lista de simulaciones que no deben mostrar ancho (constantes)
SIMULACIONES_SIN_ANCHO = ["Simulación 1a (constante continua)", "Simulación 1b (constante intervalos)"]
# ============================================================

# -------------------------------------------------------------------
# Funciones de carga
# -------------------------------------------------------------------
def cargar_datos_medidos(archivo):
    """
    Carga el archivo NetCDF de datos medidos.
    Retorna DataFrame con índice temporal y columna 'nivel_medido'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    if 'nivel_nmm' in ds.variables:
        nivel = ds['nivel_nmm'].values
    else:
        var_datos = [v for v in ds.variables if v not in ds.dims and v != 'time']
        if len(var_datos) == 0:
            raise ValueError("No se encontró variable de nivel en el archivo de mediciones.")
        nivel = ds[var_datos[0]].values
        print(f"  (mediciones) usando variable '{var_datos[0]}'")
    df = pd.DataFrame({'nivel_medido': nivel}, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def cargar_datos_simulacion(archivo):
    """
    Carga el archivo NetCDF de simulación.
    Retorna DataFrame con índice temporal y columnas 'nivel_simulado' y 'estado_canal'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values

    if 'nivel_laguna' in ds.variables:
        nivel = ds['nivel_laguna'].values
    elif 'nivel_simulado' in ds.variables:
        nivel = ds['nivel_simulado'].values
    else:
        var_datos = [v for v in ds.variables if v not in ds.dims and v != 'time']
        if len(var_datos) == 0:
            raise ValueError("No se encontró variable de nivel en el archivo de simulación.")
        nivel = ds[var_datos[0]].values
        print(f"  (simulación) usando variable '{var_datos[0]}'")

    if 'estado_canal' in ds.variables:
        estado = ds['estado_canal'].values
    else:
        estado = np.full(len(tiempo), np.nan)

    df = pd.DataFrame({
        'nivel_simulado': nivel,
        'estado_canal': estado
    }, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def cargar_datos_marea(archivo):
    """
    Carga el archivo NetCDF de marea reconstruida.
    Retorna DataFrame con índice temporal y columna 'nivel_marea'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    if 'marea' in ds.variables:
        nivel = ds['marea'].values
    else:
        var_datos = [v for v in ds.variables if v not in ds.dims and v != 'time']
        if len(var_datos) == 0:
            raise ValueError("No se encontró variable de marea en el archivo.")
        nivel = ds[var_datos[0]].values
        print(f"  (marea) usando variable '{var_datos[0]}'")
    df = pd.DataFrame({'nivel_marea': nivel}, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def cargar_datos_ancho(archivo):
    """
    Carga el archivo NetCDF de ancho real.
    Retorna DataFrame con índice temporal y columna 'ancho' (y opcionalmente 'estado').
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    # Buscar variable de ancho, priorizando 'ancho_desembocadura'
    posibles = ['ancho_desembocadura', 'ancho', 'ancho_canal', 'width', 'Ancho']
    for var in posibles:
        if var in ds.variables:
            ancho = ds[var].values
            print(f"  (ancho) usando variable '{var}'")
            break
    else:
        raise ValueError("No se encontró variable de ancho en el archivo.")
    
    # Opcionalmente cargar estado si existe
    if 'estado_canal' in ds.variables:
        estado = ds['estado_canal'].values
    else:
        estado = np.full(len(tiempo), np.nan)
    
    df = pd.DataFrame({
        'ancho': ancho,
        'estado_ancho': estado
    }, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

# -------------------------------------------------------------------
# Función de cálculo de métricas
# -------------------------------------------------------------------
def calcular_metricas(obs, sim):
    """
    Calcula MAE, RMSE y sesgo entre dos series (arrays 1D), ignorando NaN.
    Retorna un diccionario con las métricas y el número de puntos usados.
    """
    mascara = ~(np.isnan(obs) | np.isnan(sim))
    obs_valido = obs[mascara]
    sim_valido = sim[mascara]

    if len(obs_valido) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'Sesgo': np.nan, 'n_puntos': 0}

    diferencias = sim_valido - obs_valido
    mae = np.mean(np.abs(diferencias))
    rmse = np.sqrt(np.mean(diferencias**2))
    sesgo = np.mean(diferencias)

    return {'MAE': mae, 'RMSE': rmse, 'Sesgo': sesgo, 'n_puntos': len(obs_valido)}

# -------------------------------------------------------------------
# Funciones de gráficos
# -------------------------------------------------------------------
def insert_nan_at_gaps(times, values, gap_threshold=None):
    """
    Inserta NaN en values cuando el gap entre tiempos consecutivos
    supera gap_threshold (en segundos). Si gap_threshold es None,
    se estima como 2 veces el paso de tiempo mediano.
    Retorna listas (nuevos_tiempos, nuevos_valores) listas para plotear.
    """
    times_arr = times.to_numpy()
    values_arr = values.to_numpy()

    if len(times_arr) == 0:
        return times_arr, values_arr

    if gap_threshold is None:
        diffs = (times_arr[1:] - times_arr[:-1]).astype('timedelta64[s]').astype(int)
        if len(diffs) > 0:
            median_dt = np.median(diffs)
        else:
            median_dt = 3600
        gap_threshold = 2 * median_dt

    new_times = []
    new_vals = []

    for i in range(len(times_arr)):
        if i == 0:
            new_times.append(times_arr[i])
            new_vals.append(values_arr[i])
        else:
            gap = (times_arr[i] - times_arr[i-1]).astype('timedelta64[s]').astype(int)
            if gap > gap_threshold:
                mid_point1 = times_arr[i-1] + np.timedelta64(int(gap_threshold//2), 's')
                mid_point2 = times_arr[i] - np.timedelta64(int(gap_threshold//2), 's')
                new_times.append(mid_point1)
                new_vals.append(np.nan)
                new_times.append(mid_point2)
                new_vals.append(np.nan)
            new_times.append(times_arr[i])
            new_vals.append(values_arr[i])

    return new_times, new_vals

def graficar_comparacion(df_med, df_sim, df_marea, df_ancho=None, titulo=None, archivo_salida=None, show=False,
                         metricas=None, legend_loc='upper right'):
    """
    Grafica las tres series en una misma figura:
        - Medido (azul sólido)
        - Simulado (rojo sólido) con cortes en los saltos temporales
        - Marea reconstruida (verde punteada)
    Si se proporciona df_ancho, se añade un eje Y secundario con los puntos de ancho real:
        - Amarillo: ancho > 0 (abierto) con tamaño 25
        - Rojo: ancho == 0 (cerrado) con tamaño 30
    Las etiquetas del eje X se rotan 45° y se centran en las marcas.
    La leyenda se coloca en la esquina superior derecha.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Medido
    ax.plot(df_med.index, df_med['nivel_medido'], 'b-', label='Medido (NMM)', linewidth=1.5, alpha=0.8)

    # Simulado con gaps
    sim_times, sim_vals = insert_nan_at_gaps(df_sim.index, df_sim['nivel_simulado'])
    ax.plot(sim_times, sim_vals, 'r-', label='Simulado (NMM)', linewidth=1.5, alpha=0.8)

    # Marea
    ax.plot(df_marea.index, df_marea['nivel_marea'], 'g--', label='Marea reconstruida (UTide)', linewidth=1.5, alpha=0.7)

    # Añadir eje Y secundario para ancho real
    if df_ancho is not None and not df_ancho.empty:
        ax2 = ax.twinx()
        mask_abierto = df_ancho['ancho'] > 0
        mask_cerrado = df_ancho['ancho'] == 0

        if mask_abierto.any():
            ax2.scatter(df_ancho.index[mask_abierto], df_ancho['ancho'][mask_abierto],
                        color='yellow', s=25, label='Ancho real (abierto)', alpha=0.6, zorder=3,
                        edgecolors='black', linewidth=0.8)
        if mask_cerrado.any():
            ax2.scatter(df_ancho.index[mask_cerrado], df_ancho['ancho'][mask_cerrado],
                        color='red', s=30, label='Ancho real (cerrado)', alpha=0.6, zorder=3,
                        edgecolors='black', linewidth=0.8)
        ax2.set_ylabel('Ancho (m)')

        # Combinar leyendas de ambos ejes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    # Formato del eje X con rotación 45° y centrado
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('center')

    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Nivel de agua (m)')
    ax.set_title(titulo or 'Comparación de nivel de laguna')
    ax.grid(True, linestyle='--', alpha=0.6)

    if metricas is not None and not np.isnan(metricas['MAE']):
        texto = (f"MAE: {metricas['MAE']:.4f} m\n"
                 f"RMSE: {metricas['RMSE']:.4f} m\n"
                 f"Sesgo: {metricas['Sesgo']:.4f} m\n"
                 f"(n = {metricas['n_puntos']})")
        ax.text(0.02, 0.98, texto, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()

    if archivo_salida:
        fig.savefig(archivo_salida, dpi=150)
        print(f"    Gráfico guardado: {archivo_salida}")

    if show:
        plt.show()
    else:
        plt.close()

def generar_graficos_por_intervalo(df_med, df_sim, df_marea, df_ancho=None, intervalo_dias=4, carpeta=None):
    """
    Genera gráficas separadas para cada intervalo de 'intervalo_dias' días
    a lo largo de todo el período simulado. Las guarda en la carpeta especificada.
    """
    inicio_sim = df_sim.index.min()
    fin_sim = df_sim.index.max()
    fechas_inicio = pd.date_range(start=inicio_sim, end=fin_sim, freq=f'{intervalo_dias}D')

    for i, start in enumerate(fechas_inicio):
        if i < len(fechas_inicio) - 1:
            end = fechas_inicio[i+1]
        else:
            end = fin_sim

        df_med_int = df_med[(df_med.index >= start) & (df_med.index <= end)]
        df_sim_int = df_sim[(df_sim.index >= start) & (df_sim.index <= end)]
        df_marea_int = df_marea[(df_marea.index >= start) & (df_marea.index <= end)]
        df_ancho_int = None
        if df_ancho is not None:
            df_ancho_int = df_ancho[(df_ancho.index >= start) & (df_ancho.index <= end)]

        if len(df_sim_int) == 0:
            continue

        titulo_int = f"Comparación nivel laguna\n{start.strftime('%Y-%m-%d %H:%M')} a {end.strftime('%Y-%m-%d %H:%M')}"
        nombre_archivo = f"intervalo_{start.strftime('%Y%m%d_%H%M')}_a_{end.strftime('%Y%m%d_%H%M')}.png"
        archivo_salida_int = os.path.join(carpeta, nombre_archivo) if carpeta else None

        graficar_comparacion(df_med_int, df_sim_int, df_marea_int, df_ancho=df_ancho_int,
                             titulo=titulo_int, archivo_salida=archivo_salida_int,
                             show=False, legend_loc='upper right')

# -------------------------------------------------------------------
# Procesamiento de un archivo de simulación individual
# -------------------------------------------------------------------
def procesar_archivo_simulacion(ruta_sim, df_med_global, df_marea_global, df_ancho_global,
                                carpeta_salida_sim, ref_tiempos=None, etiqueta_sim=None):
    """
    Procesa un archivo de simulación:
      - Carga sus datos
      - Filtra al período común con mediciones
      - Calcula métricas de error (usando interpolación) sobre la propia malla temporal
      - Si se proporciona ref_tiempos, calcula también métricas sobre esos instantes
      - Genera gráficos (completo e intervalos) incluyendo ancho real si está disponible
    Retorna un diccionario con los resultados de las métricas (propias y de referencia).
    """
    print(f"\nProcesando: {os.path.basename(ruta_sim)}")
    if etiqueta_sim:
        print(f"  Identificada como: {etiqueta_sim}")

    try:
        df_sim = cargar_datos_simulacion(ruta_sim)
    except Exception as e:
        print(f"  ERROR al cargar simulación: {e}")
        return None

    # Período común
    t_inicio = max(df_med_global.index.min(), df_sim.index.min())
    t_fin = min(df_med_global.index.max(), df_sim.index.max())
    if t_inicio >= t_fin:
        print(f"  ADVERTENCIA: Sin superposición temporal. Se omite.")
        return None

    # Filtrar datos al período común
    df_med_filt = df_med_global.loc[t_inicio:t_fin]
    df_sim_filt = df_sim.loc[t_inicio:t_fin]
    df_marea_filt = df_marea_global.loc[t_inicio:t_fin]
    df_ancho_filt = None
    if df_ancho_global is not None:
        df_ancho_filt = df_ancho_global.loc[t_inicio:t_fin]

    # Determinar si debemos incluir ancho en gráficos (no para simulaciones constantes)
    incluir_ancho_en_graficos = True
    if etiqueta_sim in SIMULACIONES_SIN_ANCHO:
        incluir_ancho_en_graficos = False
        print("  (No se incluirá la serie de ancho real porque la simulación es de ancho constante)")

    print(f"  Período común: {t_inicio} a {t_fin}")
    print(f"  Puntos medidos: {len(df_med_filt)}")
    print(f"  Puntos simulados: {len(df_sim_filt)}")
    if df_ancho_filt is not None:
        print(f"  Puntos de ancho: {len(df_ancho_filt)}")

    # --- Cálculo de métricas sobre la propia malla temporal de la simulación ---
    df_med_interp = df_med_filt.reindex(df_sim_filt.index).interpolate(method='linear', limit_area='inside')
    puntos_validos = df_med_interp['nivel_medido'].notna().sum()
    if puntos_validos == 0:
        print(f"  ADVERTENCIA: No se pudo interpolar ningún punto. Se omiten métricas propias.")
        metricas_propias = {'MAE': np.nan, 'RMSE': np.nan, 'Sesgo': np.nan, 'n_puntos': 0}
    else:
        obs = df_med_interp['nivel_medido'].values
        sim = df_sim_filt['nivel_simulado'].values
        metricas_propias = calcular_metricas(obs, sim)
        print(f"  Métricas propias - MAE: {metricas_propias['MAE']:.4f}, RMSE: {metricas_propias['RMSE']:.4f}, Sesgo: {metricas_propias['Sesgo']:.4f} (n={metricas_propias['n_puntos']})")

    # --- Cálculo de métricas sobre la malla temporal de referencia (si se proporciona) ---
    metricas_referencia = None
    if ref_tiempos is not None:
        ref_filtrados = ref_tiempos[(ref_tiempos >= t_inicio) & (ref_tiempos <= t_fin)]
        if len(ref_filtrados) == 0:
            print(f"  ADVERTENCIA: No hay tiempos de referencia dentro del período común.")
            metricas_referencia = {'MAE': np.nan, 'RMSE': np.nan, 'Sesgo': np.nan, 'n_puntos': 0}
        else:
            df_med_ref = df_med_filt.reindex(ref_filtrados).interpolate(method='linear', limit_area='inside')
            df_sim_ref = df_sim_filt.reindex(ref_filtrados).interpolate(method='linear', limit_area='inside')
            puntos_validos_ref = (df_med_ref['nivel_medido'].notna() & df_sim_ref['nivel_simulado'].notna()).sum()
            if puntos_validos_ref == 0:
                print(f"  ADVERTENCIA: No se pudo interpolar ningún punto en la malla de referencia.")
                metricas_referencia = {'MAE': np.nan, 'RMSE': np.nan, 'Sesgo': np.nan, 'n_puntos': 0}
            else:
                obs_ref = df_med_ref['nivel_medido'].values
                sim_ref = df_sim_ref['nivel_simulado'].values
                metricas_referencia = calcular_metricas(obs_ref, sim_ref)
                print(f"  Métricas referencia - MAE: {metricas_referencia['MAE']:.4f}, RMSE: {metricas_referencia['RMSE']:.4f}, Sesgo: {metricas_referencia['Sesgo']:.4f} (n={metricas_referencia['n_puntos']})")

    # --- Generar gráficos ---
    os.makedirs(carpeta_salida_sim, exist_ok=True)

    # Gráfico completo
    nombre_base = os.path.splitext(os.path.basename(ruta_sim))[0]
    archivo_completo = os.path.join(carpeta_salida_sim, f"{nombre_base}_completo.png")
    titulo = f"Comparación nivel laguna\n{etiqueta_sim if etiqueta_sim else nombre_base}"
    graficar_comparacion(df_med_filt, df_sim_filt, df_marea_filt,
                         df_ancho=df_ancho_filt if incluir_ancho_en_graficos else None,
                         titulo=titulo, archivo_salida=archivo_completo, show=False,
                         metricas=metricas_propias)

    # Gráficos por intervalos de 4 días
    generar_graficos_por_intervalo(df_med_filt, df_sim_filt, df_marea_filt,
                                   df_ancho=df_ancho_filt if incluir_ancho_en_graficos else None,
                                   intervalo_dias=4, carpeta=carpeta_salida_sim)

    return {
        'archivo': os.path.basename(ruta_sim),
        'etiqueta': etiqueta_sim,
        'periodo': f"{t_inicio} a {t_fin}",
        'metricas_propias': metricas_propias,
        'metricas_referencia': metricas_referencia
    }

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    # Verificar existencia de archivos base
    for ruta, nombre in [(archivo_medido, "mediciones"), (archivo_marea, "marea")]:
        if not os.path.exists(ruta):
            print(f"ERROR: No se encuentra el archivo de {nombre}:\n{ruta}")
            return

    if not os.path.isdir(carpeta_simulaciones):
        print(f"ERROR: No se encuentra la carpeta de simulaciones:\n{carpeta_simulaciones}")
        return

    # Crear carpeta de salida global
    os.makedirs(carpeta_salida_base, exist_ok=True)
    print(f"Los resultados se guardarán en: {os.path.abspath(carpeta_salida_base)}")

    # Cargar datos de medición y marea
    print("\nCargando datos de medición...")
    df_med = cargar_datos_medidos(archivo_medido)
    print(f"  Rango: {df_med.index.min()} a {df_med.index.max()}  ({len(df_med)} puntos)")

    print("Cargando datos de marea...")
    df_marea = cargar_datos_marea(archivo_marea)
    print(f"  Rango: {df_marea.index.min()} a {df_marea.index.max()}  ({len(df_marea)} puntos)")

    # Cargar datos de ancho real
    print("Cargando datos de ancho real...")
    if os.path.exists(RUTA_ANCHOS_NETCDF):
        try:
            df_ancho = cargar_datos_ancho(RUTA_ANCHOS_NETCDF)
            print(f"  Rango: {df_ancho.index.min()} a {df_ancho.index.max()}  ({len(df_ancho)} puntos)")
        except Exception as e:
            print(f"  ERROR al cargar ancho: {e}")
            df_ancho = None
    else:
        print(f"  ADVERTENCIA: No se encuentra el archivo de ancho:\n{RUTA_ANCHOS_NETCDF}")
        df_ancho = None

    # Cargar archivo de referencia para obtener sus tiempos
    ref_tiempos = None
    ruta_referencia = os.path.join(carpeta_simulaciones, archivo_referencia_tiempos_nombre)
    if os.path.exists(ruta_referencia):
        print("\nCargando archivo de referencia para tiempos (intervalos de ancho)...")
        try:
            df_ref = cargar_datos_simulacion(ruta_referencia)
            ref_tiempos = df_ref.index
            print(f"  Tiempos de referencia: {ref_tiempos.min()} a {ref_tiempos.max()}  ({len(ref_tiempos)} puntos)")
        except Exception as e:
            print(f"  ERROR al cargar archivo de referencia: {e}")
            ref_tiempos = None
    else:
        print(f"\nADVERTENCIA: No se encuentra el archivo de referencia '{archivo_referencia_tiempos_nombre}' en la carpeta de simulaciones.")
        print("Se continuará sin calcular métricas sobre tiempos de referencia.")

    # Listar archivos de simulación
    patron = os.path.join(carpeta_simulaciones, "*.nc")
    archivos_sim = glob.glob(patron)
    if not archivos_sim:
        print(f"No se encontraron archivos .nc en {carpeta_simulaciones}")
        return

    print(f"\nSe encontraron {len(archivos_sim)} archivos de simulación.")

    resultados = []

    # Procesar cada simulación
    for i, ruta_sim in enumerate(archivos_sim, 1):
        print(f"\n--- Simulación {i}/{len(archivos_sim)} ---")
        nombre_archivo = os.path.basename(ruta_sim)
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        etiqueta = mapeo_simulaciones.get(nombre_sin_ext, nombre_sin_ext)

        carpeta_sim = os.path.join(carpeta_salida_base, nombre_sin_ext)
        res = procesar_archivo_simulacion(ruta_sim, df_med, df_marea, df_ancho,
                                          carpeta_sim, ref_tiempos=ref_tiempos,
                                          etiqueta_sim=etiqueta)
        if res is not None:
            resultados.append(res)

    # Guardar todas las métricas en un archivo de texto
    if resultados:
        archivo_metricas = os.path.join(carpeta_salida_base, "metricas_error_todas_simulaciones.txt")
        with open(archivo_metricas, 'w', encoding='utf-8') as f:
            f.write("MÉTRICAS DE ERROR PARA CADA SIMULACIÓN\n")
            f.write("=======================================\n")
            f.write(f"Archivo de mediciones: {archivo_medido}\n")
            f.write(f"Archivo de marea: {archivo_marea}\n")
            f.write(f"Carpeta de simulaciones: {carpeta_simulaciones}\n")
            if ref_tiempos is not None:
                f.write(f"Tiempos de referencia (intervalos de ancho) tomados de: {archivo_referencia_tiempos_nombre}\n")
            f.write("\n")

            for res in resultados:
                if res['etiqueta'] == "Simulación 1a (constante continua)":
                    nota = "NOTA: Para esta simulación (continua), la métrica que debe considerarse es la calculada sobre la malla de referencia (intervalos).\n"
                else:
                    nota = ""

                f.write(f"Simulación: {res['etiqueta']} ({res['archivo']})\n")
                f.write(f"  Período común: {res['periodo']}\n")
                f.write(nota)

                mp = res['metricas_propias']
                f.write(f"  Métricas sobre malla propia:\n")
                f.write(f"    Puntos utilizados: {mp['n_puntos']}\n")
                f.write(f"    MAE  : {mp['MAE']:.6f} m\n")
                f.write(f"    RMSE : {mp['RMSE']:.6f} m\n")
                f.write(f"    Sesgo: {mp['Sesgo']:.6f} m\n")

                if res['metricas_referencia'] is not None:
                    mr = res['metricas_referencia']
                    f.write(f"  Métricas sobre malla de referencia (intervalos de ancho):\n")
                    f.write(f"    Puntos utilizados: {mr['n_puntos']}\n")
                    f.write(f"    MAE  : {mr['MAE']:.6f} m\n")
                    f.write(f"    RMSE : {mr['RMSE']:.6f} m\n")
                    f.write(f"    Sesgo: {mr['Sesgo']:.6f} m\n")
                else:
                    f.write(f"  (No se calcularon métricas sobre malla de referencia)\n")
                f.write("\n")

        print(f"\nMétricas guardadas en: {archivo_metricas}")
    else:
        print("\nNo se generaron métricas para ningún archivo.")

    print("\nProceso completado.")

if __name__ == "__main__":
    main()