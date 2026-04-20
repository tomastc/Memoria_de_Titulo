import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURACIÓN DE RUTAS (MODIFICAR SEGÚN CORRESPONDA)
# ============================================================
archivo_medido = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Nivel Laguna NMM - data completa 2023-2024\Nivel_laguna_ref_NMM.nc"
archivo_marea_utide = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Marea reconstruida\marea_reconstruida_utide_2023-2024.nc"
archivo_marea_estimada = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\02 Mareas\Resultados_mareas\Resultado_mareas_estimadas_10min.nc"

# Carpeta donde se guardarán los gráficos
carpeta_salida = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Comparacion_niveles_marea_vs_laguna_medido"
# ============================================================

def insertar_nan_en_gaps(tiempos, valores, umbral_horas=48):
    """
    Inserta NaN en 'valores' cuando el gap entre tiempos consecutivos
    supera 'umbral_horas' (en horas). Retorna arrays de tiempos y valores
    listos para graficar (con NaN intercalados).
    """
    tiempos_arr = tiempos.to_numpy()          # array de datetime64
    valores_arr = valores.to_numpy()
    
    if len(tiempos_arr) == 0:
        return tiempos_arr, valores_arr
    
    umbral_seg = umbral_horas * 3600
    
    nuevos_tiempos = []
    nuevos_valores = []
    
    for i in range(len(tiempos_arr)):
        if i == 0:
            nuevos_tiempos.append(tiempos_arr[i])
            nuevos_valores.append(valores_arr[i])
        else:
            gap = (tiempos_arr[i] - tiempos_arr[i-1]).astype('timedelta64[s]').astype(int)
            if gap > umbral_seg:
                punto_medio1 = tiempos_arr[i-1] + np.timedelta64(int(umbral_seg//2), 's')
                punto_medio2 = tiempos_arr[i] - np.timedelta64(int(umbral_seg//2), 's')
                nuevos_tiempos.append(punto_medio1)
                nuevos_valores.append(np.nan)
                nuevos_tiempos.append(punto_medio2)
                nuevos_valores.append(np.nan)
            nuevos_tiempos.append(tiempos_arr[i])
            nuevos_valores.append(valores_arr[i])
    
    return np.array(nuevos_tiempos), np.array(nuevos_valores)

def cargar_medido(archivo):
    """Carga nivel medido (referido a NMM) y retorna DataFrame con índice temporal."""
    ds = xr.open_dataset(archivo)
    if 'nivel_nmm' in ds.variables:
        nivel = ds['nivel_nmm'].values
    else:
        var = [v for v in ds.variables if v not in ds.dims and v != 'time'][0]
        nivel = ds[var].values
        print(f"Usando variable '{var}' como nivel medido.")
    df = pd.DataFrame({'nivel_medido': nivel}, index=pd.DatetimeIndex(ds['time'].values))
    ds.close()
    return df

def cargar_marea_utide(archivo):
    """Carga nivel de marea reconstruida con UTide (variable 'marea' o primera variable)."""
    ds = xr.open_dataset(archivo)
    if 'marea' in ds.variables:
        nivel = ds['marea'].values
    else:
        var = [v for v in ds.variables if v not in ds.dims and v != 'time'][0]
        nivel = ds[var].values
        print(f"Usando variable '{var}' como marea UTide.")
    df = pd.DataFrame({'nivel_marea': nivel}, index=pd.DatetimeIndex(ds['time'].values))
    ds.close()
    return df

def cargar_marea_estimada(archivo):
    """
    Carga la marea estimada directa (variable 'altura_cahuil_estimada')
    del archivo de resultados cada 10 minutos.
    """
    ds = xr.open_dataset(archivo)
    if 'altura_cahuil_estimada' in ds.variables:
        nivel = ds['altura_cahuil_estimada'].values
    else:
        # Fallback: tomar la primera variable de datos
        var = [v for v in ds.variables if v not in ds.dims and v != 'time'][0]
        nivel = ds[var].values
        print(f"Variable 'altura_cahuil_estimada' no encontrada. Usando '{var}' como marea estimada.")
    df = pd.DataFrame({'nivel_marea_estimada': nivel}, index=pd.DatetimeIndex(ds['time'].values))
    ds.close()
    return df

def graficar_comparacion(tiempos_med, valores_med, df_marea_filt, titulo, nombre_archivo, color_marea='g--', etiqueta_marea='Marea'):
    """
    Función auxiliar para generar una gráfica con nivel medido y una serie de marea.
    """
    plt.figure(figsize=(14, 6))
    
    # Nivel medido con gaps
    plt.plot(tiempos_med, valores_med, 'b-', linewidth=1.2, label='Nivel laguna medido (NMM)')
    
    # Marea (puede ser UTide o estimada)
    plt.plot(df_marea_filt.index, df_marea_filt.iloc[:, 0], color_marea, linewidth=1.2, label=etiqueta_marea)
    
    # Formato del eje x: cada 1 mes con formato YYYY-MM
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    
    plt.xlabel('Fecha')
    plt.ylabel('Nivel de agua (m)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Guardar
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
    plt.savefig(ruta_salida, dpi=150)
    print(f"Gráfico guardado en: {ruta_salida}")
    plt.close()

def main():
    os.makedirs(carpeta_salida, exist_ok=True)

    # Cargar datos medidos (una sola vez)
    print("Cargando nivel medido...")
    df_med = cargar_medido(archivo_medido)
    print(f"  Rango: {df_med.index.min()} a {df_med.index.max()}  ({len(df_med)} puntos)")

    # Definir ventana temporal basada en los datos medidos
    inicio = df_med.index.min()
    fin = df_med.index.max()
    print(f"\nVentana de análisis (basada en datos medidos): {inicio} a {fin}")

    # Procesar nivel medido con gaps
    tiempos_med, valores_med = insertar_nan_en_gaps(df_med.index, df_med['nivel_medido'], umbral_horas=48)

    # ------------------------------------------------------------
    # Gráfico 1: Comparación con marea UTide
    # ------------------------------------------------------------
    print("\nProcesando marea reconstruida (UTide)...")
    df_marea_utide = cargar_marea_utide(archivo_marea_utide)
    print(f"  Rango: {df_marea_utide.index.min()} a {df_marea_utide.index.max()}  ({len(df_marea_utide)} puntos)")

    # Filtrar a la ventana de datos medidos
    df_marea_utide_filt = df_marea_utide.loc[inicio:fin]
    if len(df_marea_utide_filt) == 0:
        print("  Advertencia: No hay datos de marea UTide en la ventana.")
    else:
        graficar_comparacion(
            tiempos_med, valores_med,
            df_marea_utide_filt,
            titulo='Comparación: Nivel medido de laguna vs. Marea reconstruida (UTide)',
            nombre_archivo='comparacion_nivel_medido_vs_marea_utide.png',
            color_marea='g--',
            etiqueta_marea='Marea reconstruida (UTide)'
        )

    # ------------------------------------------------------------
    # Gráfico 2: Comparación con marea estimada directa
    # ------------------------------------------------------------
    print("\nProcesando marea estimada directa...")
    df_marea_estimada = cargar_marea_estimada(archivo_marea_estimada)
    print(f"  Rango: {df_marea_estimada.index.min()} a {df_marea_estimada.index.max()}  ({len(df_marea_estimada)} puntos)")

    # Filtrar a la ventana de datos medidos
    df_marea_estimada_filt = df_marea_estimada.loc[inicio:fin]
    if len(df_marea_estimada_filt) == 0:
        print("  Advertencia: No hay datos de marea estimada en la ventana.")
    else:
        graficar_comparacion(
            tiempos_med, valores_med,
            df_marea_estimada_filt,
            titulo='Comparación: Nivel medido de laguna vs. Marea estimada directa',
            nombre_archivo='comparacion_nivel_medido_vs_marea_estimada.png',
            color_marea='m-',  # magenta sólido para distinguir
            etiqueta_marea='Marea estimada (directa)'
        )

    print("\nProceso completado.")

if __name__ == "__main__":
    main()