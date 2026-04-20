import xarray as xr
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURACIÓN MANUAL DE RUTAS (igual que en el código anterior)
# ============================================================
archivo_medido = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Nivel Laguna NMM - data completa 2023-2024\Nivel_laguna_ref_NMM.nc"
archivo_simulacion = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Laguna Simulados\resultados_simulacion_variable_FINAL_2023-05_Qr_cte.nc"
carpeta_salida = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados_metricas_error_variable_2023-05"
# ============================================================

def cargar_datos_medidos(archivo):
    """
    Carga el archivo NetCDF de datos medidos.
    Retorna un DataFrame con índice temporal y columna 'nivel_medido'.
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
    Retorna un DataFrame con índice temporal y columna 'nivel_simulado'.
    """
    ds = xr.open_dataset(archivo)
    tiempo = ds['time'].values
    nivel = ds['nivel_laguna'].values
    df = pd.DataFrame({'nivel_simulado': nivel}, index=pd.DatetimeIndex(tiempo, name='tiempo'))
    ds.close()
    return df

def calcular_metricas(obs, sim):
    """
    Calcula MAE, RMSE y sesgo (bias) entre dos series (arrays 1D).
    Ignora valores NaN.
    Retorna un diccionario con las métricas y el número de puntos usados.
    """
    mascara_valida = ~(np.isnan(obs) | np.isnan(sim))
    obs_valido = obs[mascara_valida]
    sim_valido = sim[mascara_valida]
    
    if len(obs_valido) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'Sesgo': np.nan, 'n_puntos': 0}
    
    diferencias = sim_valido - obs_valido
    mae = np.mean(np.abs(diferencias))
    rmse = np.sqrt(np.mean(diferencias**2))
    sesgo = np.mean(diferencias)
    
    return {'MAE': mae, 'RMSE': rmse, 'Sesgo': sesgo, 'n_puntos': len(obs_valido)}

def main():
    # Verificar archivos
    if not os.path.exists(archivo_medido):
        print(f"ERROR: No se encuentra el archivo de mediciones:\n{archivo_medido}")
        return
    if not os.path.exists(archivo_simulacion):
        print(f"ERROR: No se encuentra el archivo de simulación:\n{archivo_simulacion}")
        return

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"Los resultados se guardarán en: {os.path.abspath(carpeta_salida)}")

    # Cargar datos
    print("Cargando datos medidos...")
    df_med = cargar_datos_medidos(archivo_medido)
    print(f"  Rango temporal: {df_med.index.min()} a {df_med.index.max()}")
    print(f"  Número de puntos: {len(df_med)}")

    print("Cargando datos de simulación...")
    df_sim = cargar_datos_simulacion(archivo_simulacion)
    print(f"  Rango temporal: {df_sim.index.min()} a {df_sim.index.max()}")
    print(f"  Número de puntos: {len(df_sim)}")

    # Determinar período común
    t_inicio = max(df_med.index.min(), df_sim.index.min())
    t_fin = min(df_med.index.max(), df_sim.index.max())
    
    if t_inicio >= t_fin:
        print("ERROR: No hay superposición temporal entre las series.")
        return
    
    print(f"\nPeríodo común: {t_inicio} a {t_fin}")
    
    # Filtrar datos al período común
    df_med_comun = df_med.loc[t_inicio:t_fin]
    df_sim_comun = df_sim.loc[t_inicio:t_fin]
    
    print(f"  Puntos medidos en período común: {len(df_med_comun)}")
    print(f"  Puntos simulados en período común: {len(df_sim_comun)}")

    # Interpolar mediciones a los instantes de simulación
    # Se crea un DataFrame con índice igual a los tiempos de simulación y se interpolar linealmente
    df_med_interp = df_med_comun.reindex(df_sim_comun.index).interpolate(method='linear', limit_area='inside')
    
    # Verificar cuántos valores se pudieron interpolar
    puntos_validos = df_med_interp['nivel_medido'].notna().sum()
    print(f"  Puntos con interpolación válida: {puntos_validos} de {len(df_sim_comun)}")
    
    if puntos_validos == 0:
        print("ERROR: No se pudo interpolar ningún valor de medición en los tiempos de simulación.")
        return

    # Extraer series para comparación
    obs = df_med_interp['nivel_medido'].values
    sim = df_sim_comun['nivel_simulado'].values

    # Calcular métricas
    metricas = calcular_metricas(obs, sim)
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS DE MÉTRICAS DE ERROR")
    print("="*50)
    print(f"Número de puntos utilizados: {metricas['n_puntos']}")
    print(f"MAE   (Error Absoluto Medio): {metricas['MAE']:.4f} m")
    print(f"RMSE  (Raíz del Error Cuadrático Medio): {metricas['RMSE']:.4f} m")
    print(f"Sesgo (Media de la diferencia Sim - Obs): {metricas['Sesgo']:.4f} m")
    print("="*50)

    # Guardar resultados en archivo de texto
    nombre_med = os.path.splitext(os.path.basename(archivo_medido))[0]
    nombre_sim = os.path.splitext(os.path.basename(archivo_simulacion))[0]
    archivo_resultados = os.path.join(carpeta_salida, f"metricas_error_{nombre_med}_vs_{nombre_sim}.txt")
    
    with open(archivo_resultados, 'w') as f:
        f.write("MÉTRICAS DE ERROR - NIVEL DE LAGUNA\n")
        f.write(f"Archivo mediciones: {archivo_medido}\n")
        f.write(f"Archivo simulación: {archivo_simulacion}\n")
        f.write(f"Período común: {t_inicio} a {t_fin}\n")
        f.write(f"Número de puntos (tiempos de simulación con interpolación): {metricas['n_puntos']}\n")
        f.write(f"MAE  : {metricas['MAE']:.6f} m\n")
        f.write(f"RMSE : {metricas['RMSE']:.6f} m\n")
        f.write(f"Sesgo: {metricas['Sesgo']:.6f} m\n")
    
    print(f"\nResultados guardados en: {archivo_resultados}")

if __name__ == "__main__":
    main()