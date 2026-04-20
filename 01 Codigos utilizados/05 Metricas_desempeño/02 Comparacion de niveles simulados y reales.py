import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURACIÓN MANUAL DE RUTAS
# ============================================================
# Especifica aquí las rutas completas a tus archivos NetCDF
archivo_medido = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Nivel Laguna NMM - data completa 2023-2024\Nivel_laguna_ref_NMM.nc"  
archivo_simulacion = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Laguna Simulados\resultados_simulacion_constante_FINAL_2024-07.nc"
archivo_marea = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Marea reconstruida\marea_reconstruida_utide_2023-2024.nc"

# Carpeta donde se guardarán los gráficos
carpeta_salida = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Comparacion de niveles laguna_constante_2024-07"
# ============================================================

def cargar_datos_medidos(archivo):
    """
    Carga el archivo NetCDF de datos medidos.
    Se espera que contenga:
        - variable 'time' con unidades decodificables (segundos desde 1970-01-01)
        - variable 'nivel_nmm' con el nivel de agua referido a NMM (m)
    Retorna un DataFrame de pandas con índice temporal y columna 'nivel_medido'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    nivel = ds['nivel_nmm'].values
    df = pd.DataFrame({'nivel_medido': nivel}, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def cargar_datos_simulacion(archivo):
    """
    Carga el archivo NetCDF de simulación.
    Se espera que contenga:
        - coordenada 'time' (decodificable a datetime)
        - variable 'nivel_laguna' (nivel simulado)
        - variable 'estado_canal' (0=cerrado, 1=abierto)
    Retorna un DataFrame de pandas con índice temporal y columnas 'nivel_simulado' y 'estado_canal'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    nivel = ds['nivel_laguna'].values
    estado = ds['estado_canal'].values
    df = pd.DataFrame({
        'nivel_simulado': nivel,
        'estado_canal': estado
    }, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def cargar_datos_marea(archivo):
    """
    Carga el archivo NetCDF de marea reconstruida con UTide.
    Se asume que contiene:
        - coordenada 'time' (decodificable a datetime)
        - variable 'marea' con el nivel de marea (m)
    Retorna un DataFrame de pandas con índice temporal y columna 'nivel_marea'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    # Intentar con el nombre 'marea', si no, buscar la primera variable de datos
    if 'marea' in ds.variables:
        nivel = ds['marea'].values
    else:
        # Si no hay variable 'marea', tomar la primera variable que no sea coordenada
        var_names = [v for v in ds.variables if v not in ds.dims and v != 'time']
        if len(var_names) == 0:
            raise ValueError("No se encontró una variable de datos en el archivo de marea.")
        nivel = ds[var_names[0]].values
        print(f"Usando variable '{var_names[0]}' como nivel de marea.")
    df = pd.DataFrame({'nivel_marea': nivel}, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def graficar_comparacion(df_med, df_sim, df_marea, titulo=None, archivo_salida=None, show=True):
    """
    Grafica las tres series en una misma figura:
        - Medido (azul)
        - Simulado (rojo) **solo en los intervalos simulados (sin conexión en los saltos)**
        - Marea reconstruida (verde, punteada)
    Además, marca con puntos rojos los instantes donde el canal está cerrado (estado_canal == 0).
    Las etiquetas del eje X se muestran cada 1 día.
    Si se proporciona archivo_salida, guarda la figura.
    Si show es True, muestra la figura en pantalla (por defecto True).
    """
    # ------------------------------------------------------------
    # Función auxiliar para insertar NaN en saltos temporales grandes
    # ------------------------------------------------------------
    def insert_nan_at_gaps(times, values, gap_threshold=None):
        """
        Inserta NaN en values cuando el gap entre tiempos consecutivos
        supera gap_threshold (en segundos). Si gap_threshold es None,
        se estima como 2 veces el paso de tiempo mediano.
        Retorna listas (nuevos_tiempos, nuevos_valores) listas para plotear.
        """
        # Convertir a arrays de numpy para evitar problemas de indexación de pandas
        times_arr = times.to_numpy()           # array de datetime64
        values_arr = values.to_numpy()          # array de floats

        if len(times_arr) == 0:
            return times_arr, values_arr

        if gap_threshold is None:
            # Calcular paso de tiempo mediano en segundos
            diffs = (times_arr[1:] - times_arr[:-1]).astype('timedelta64[s]').astype(int)
            if len(diffs) > 0:
                median_dt = np.median(diffs)
            else:
                median_dt = 3600  # fallback 1 hora
            gap_threshold = 2 * median_dt  # segundos

        new_times = []
        new_vals = []

        for i in range(len(times_arr)):
            if i == 0:
                new_times.append(times_arr[i])
                new_vals.append(values_arr[i])
            else:
                gap = (times_arr[i] - times_arr[i-1]).astype('timedelta64[s]').astype(int)
                if gap > gap_threshold:
                    # Insertar NaN para romper la línea
                    # Se añaden dos puntos ficticios (con NaN) alrededor del gap
                    mid_point1 = times_arr[i-1] + np.timedelta64(int(gap_threshold//2), 's')
                    mid_point2 = times_arr[i] - np.timedelta64(int(gap_threshold//2), 's')
                    new_times.append(mid_point1)
                    new_vals.append(np.nan)
                    new_times.append(mid_point2)
                    new_vals.append(np.nan)
                new_times.append(times_arr[i])
                new_vals.append(values_arr[i])

        return new_times, new_vals

    plt.figure(figsize=(12, 6))

    # Serie medida (continua)
    plt.plot(df_med.index, df_med['nivel_medido'], 'b-', label='Medido (NMM)', linewidth=1.5, alpha=0.8)

    # Serie simulada con gaps
    sim_times, sim_vals = insert_nan_at_gaps(df_sim.index, df_sim['nivel_simulado'])
    plt.plot(sim_times, sim_vals, 'r-', label='Simulado (NMM)', linewidth=1.5, alpha=0.8)

    # Marea reconstruida (continua)
    plt.plot(df_marea.index, df_marea['nivel_marea'], 'g--', label='Marea reconstruida (UTide)', linewidth=1.5, alpha=0.7)

    # Puntos de canal cerrado (estado_canal == 0)
    cerrados = df_sim[df_sim['estado_canal'] == 0]
    if not cerrados.empty:
        plt.scatter(cerrados.index, cerrados['nivel_simulado'],
                    color='red', s=20, zorder=5, label='Canal cerrado (simulación)')

    # Configurar eje X con ticks cada 1 día
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('Tiempo')
    plt.ylabel('Nivel de agua (m)')
    plt.title(titulo or 'Comparación de nivel de laguna: medido vs simulado vs marea')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if archivo_salida:
        plt.savefig(archivo_salida, dpi=150)
        print(f"Gráfico guardado en: {archivo_salida}")

    if show:
        plt.show()
    else:
        plt.close()  # Cerrar la figura para no acumular memoria

def generar_graficos_por_intervalo(df_med, df_sim, df_marea, intervalo_dias=4, carpeta=None):
    """
    Genera gráficas separadas para cada intervalo de 'intervalo_dias' días
    a lo largo de todo el período simulado.
    Las guarda en la carpeta especificada con nombres que indican el rango.
    """
    inicio_sim = df_sim.index.min()
    fin_sim = df_sim.index.max()

    # Crear rangos de fechas cada 'intervalo_dias' días
    fechas_inicio = pd.date_range(start=inicio_sim, end=fin_sim, freq=f'{intervalo_dias}D')

    for i, start in enumerate(fechas_inicio):
        if i < len(fechas_inicio) - 1:
            end = fechas_inicio[i+1]
        else:
            end = fin_sim

        # Filtrar datos en el intervalo [start, end]
        df_med_int = df_med[(df_med.index >= start) & (df_med.index <= end)]
        df_sim_int = df_sim[(df_sim.index >= start) & (df_sim.index <= end)]
        df_marea_int = df_marea[(df_marea.index >= start) & (df_marea.index <= end)]

        if len(df_sim_int) == 0:
            continue

        titulo_int = f"Comparación nivel laguna\n{start.strftime('%Y-%m-%d %H:%M')} a {end.strftime('%Y-%m-%d %H:%M')}"
        nombre_archivo = f"comparacion_{start.strftime('%Y%m%d_%H%M')}_a_{end.strftime('%Y%m%d_%H%M')}.png"
        archivo_salida_int = os.path.join(carpeta, nombre_archivo) if carpeta else None

        graficar_comparacion(df_med_int, df_sim_int, df_marea_int,
                             titulo=titulo_int, archivo_salida=archivo_salida_int, show=False)

    print(f"Se generaron {len(fechas_inicio)} gráficos de intervalos de {intervalo_dias} días.")

def main():
    # Verificar que los archivos existan
    if not os.path.exists(archivo_medido):
        print(f"ERROR: No se encuentra el archivo de mediciones:\n{archivo_medido}")
        return
    if not os.path.exists(archivo_simulacion):
        print(f"ERROR: No se encuentra el archivo de simulación:\n{archivo_simulacion}")
        return
    if not os.path.exists(archivo_marea):
        print(f"ERROR: No se encuentra el archivo de marea:\n{archivo_marea}")
        return

    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"Los resultados se guardarán en: {os.path.abspath(carpeta_salida)}")

    # Cargar datos
    print("Cargando datos medidos...")
    df_med = cargar_datos_medidos(archivo_medido)
    print(f"  Rango temporal original: {df_med.index.min()} a {df_med.index.max()}")
    print(f"  Número de puntos: {len(df_med)}")

    print("Cargando datos de simulación...")
    df_sim = cargar_datos_simulacion(archivo_simulacion)
    print(f"  Rango temporal original: {df_sim.index.min()} a {df_sim.index.max()}")
    print(f"  Número de puntos: {len(df_sim)}")

    print("Cargando datos de marea reconstruida...")
    df_marea = cargar_datos_marea(archivo_marea)
    print(f"  Rango temporal original: {df_marea.index.min()} a {df_marea.index.max()}")
    print(f"  Número de puntos: {len(df_marea)}")

    # ------------------------------------------------------------
    # Filtrar para mostrar solo la ventana de simulación (exactamente el rango de la simulación)
    # ------------------------------------------------------------
    inicio_sim = df_sim.index.min()
    fin_sim = df_sim.index.max()

    print(f"Ventana de visualización (simulación): {inicio_sim} a {fin_sim}")

    # Aplicar filtro a los tres DataFrames
    df_med_filtrado = df_med[(df_med.index >= inicio_sim) & (df_med.index <= fin_sim)]
    df_sim_filtrado = df_sim[(df_sim.index >= inicio_sim) & (df_sim.index <= fin_sim)]
    df_marea_filtrado = df_marea[(df_marea.index >= inicio_sim) & (df_marea.index <= fin_sim)]

    print(f"Datos medidos en ventana: {len(df_med_filtrado)} puntos")
    print(f"Datos simulados en ventana: {len(df_sim_filtrado)} puntos")
    print(f"Datos de marea en ventana: {len(df_marea_filtrado)} puntos")

    if len(df_sim_filtrado) == 0:
        print("ERROR: No hay datos simulados en la ventana. Abortando.")
        return
    if len(df_med_filtrado) == 0:
        print("ADVERTENCIA: No hay datos medidos en la ventana seleccionada.")
    if len(df_marea_filtrado) == 0:
        print("ADVERTENCIA: No hay datos de marea en la ventana seleccionada.")

    # Generar nombre de archivo de salida
    nombre_med = os.path.splitext(os.path.basename(archivo_medido))[0]
    nombre_sim = os.path.splitext(os.path.basename(archivo_simulacion))[0]
    nombre_grafico = f"comparacion_{nombre_med}_vs_{nombre_sim}_con_marea.png"
    archivo_salida = os.path.join(carpeta_salida, nombre_grafico)

    titulo = f"Comparación nivel laguna\n{nombre_med} vs {nombre_sim}"

    # Graficar con los datos filtrados (gráfico completo)
    graficar_comparacion(df_med_filtrado, df_sim_filtrado, df_marea_filtrado,
                         titulo=titulo, archivo_salida=archivo_salida, show=True)

    # Generar gráficas cada 4 días
    print("\nGenerando gráficas por intervalos de 4 días...")
    generar_graficos_por_intervalo(df_med_filtrado, df_sim_filtrado, df_marea_filtrado,
                                   intervalo_dias=4, carpeta=carpeta_salida)

    print("Proceso completado.")

if __name__ == "__main__":
    main()