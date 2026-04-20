"""
Script para graficar comparación entre la marea estimada en Cáhuil (desde NetCDF)
y la marea reconstruida mediante UTIDE (desde archivo NetCDF específico).
Incluye un panel inferior con zoom de 10 días y cálculo de métricas de error (MAE, RMSE, SESGO).
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

# ========================= CONFIGURACIÓN DE RUTAS =========================
ARCHIVO_RESULTADOS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas/Resultado_mareas_estimadas_10min.nc'
ARCHIVO_UTIDE = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/marea_reconstruida_utide_2023-2024.nc'
CARPETA_RESULTADOS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas'

# ==================== CONFIGURACIÓN DEL ZOOM ==============================
DIAS_ZOOM = 20
FECHA_INICIO_ZOOM = '2024-07-10'
# =========================================================================

def cargar_utide_desde_nc(ruta_nc):
    """
    Carga un archivo NetCDF de UTIDE y extrae tiempo y altura.
    Devuelve DataFrame con columnas 'time' y 'altura_utide'.
    """
    if not os.path.isfile(ruta_nc):
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta_nc}")
    
    ds = xr.open_dataset(ruta_nc)
    print("Variables disponibles en el archivo UTIDE:", list(ds.variables))
    print("Coordenadas:", list(ds.coords))
    
    posibles_altura = ['altura', 'tide', 'ssh', 'marea', 'reconstruida', 'water_level', 'eta', 'height']
    var_altura = None
    for v in posibles_altura:
        if v in ds.variables:
            var_altura = v
            break
    if var_altura is None:
        data_vars = [v for v in ds.variables if v not in ds.coords]
        if data_vars:
            var_altura = data_vars[0]
            print(f"No se encontró una variable de altura obvia. Se usará '{var_altura}'")
        else:
            raise ValueError("No se encontró ninguna variable de datos en el NetCDF UTIDE")
    
    if 'time' in ds.coords:
        tiempo = pd.to_datetime(ds['time'].values)
    else:
        time_coord = None
        for coord in ds.coords:
            if 'time' in coord.lower():
                time_coord = coord
                break
        if time_coord is None:
            for var in ds.variables:
                if np.issubdtype(ds[var].dtype, np.datetime64):
                    time_coord = var
                    break
        if time_coord is None:
            raise ValueError("No se pudo identificar una coordenada de tiempo en el NetCDF UTIDE")
        tiempo = pd.to_datetime(ds[time_coord].values)
    
    df = pd.DataFrame({
        'time': tiempo,
        'altura_utide': ds[var_altura].values
    })
    ds.close()
    return df

def calcular_metricas_error(df_cahuil, df_utide, tolerancia='5min'):
    """
    Alinea las dos series temporales (Cáhuil y UTIDE) mediante merge_asof,
    calcula residuos (UTIDE - Cáhuil) y retorna MAE, RMSE y SESGO.
    """
    df_cahuil = df_cahuil.sort_values('time').copy()
    df_utide = df_utide.sort_values('time').copy()
    
    df_merged = pd.merge_asof(
        df_cahuil,
        df_utide,
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(tolerancia)
    )
    df_merged = df_merged.dropna(subset=['altura_utide'])
    
    if len(df_merged) == 0:
        raise ValueError("No se pudo alinear ninguna observación entre Cáhuil y UTIDE.")
    
    df_merged['residual'] = df_merged['altura_utide'] - df_merged['cahuil_estimada']
    
    mae = np.mean(np.abs(df_merged['residual']))
    rmse = np.sqrt(np.mean(df_merged['residual']**2))
    sesgo = np.mean(df_merged['residual'])
    
    print("\n===== MÉTRICAS DE ERROR (UTIDE vs Cáhuil) =====")
    print(f"MAE  (Error Absoluto Medio): {mae:.4f} m")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f} m")
    print(f"SESGO (Error sistemático medio): {sesgo:.4f} m")
    print("===============================================\n")
    
    return mae, rmse, sesgo

def graficar_comparacion_cahuil_utide():
    """
    Carga los datos, calcula errores y genera figura con dos paneles.
    Panel superior: serie completa con recuadro de métricas.
    Panel inferior: zoom de 10 días.
    """
    # Cargar datos Cáhuil
    print("Cargando datos de Cáhuil estimada desde NetCDF...")
    if not os.path.isfile(ARCHIVO_RESULTADOS):
        raise FileNotFoundError(f"No se encuentra el archivo: {ARCHIVO_RESULTADOS}")
    ds_cahuil = xr.open_dataset(ARCHIVO_RESULTADOS)
    if 'altura_cahuil_estimada' not in ds_cahuil.variables:
        raise KeyError(f"Variable 'altura_cahuil_estimada' no encontrada. Disponibles: {list(ds_cahuil.variables)}")
    df_cahuil = ds_cahuil[['time', 'altura_cahuil_estimada']].to_dataframe().reset_index()
    df_cahuil = df_cahuil.rename(columns={'altura_cahuil_estimada': 'cahuil_estimada'})
    ds_cahuil.close()
    print(f"Datos Cáhuil: {len(df_cahuil)} registros, {df_cahuil['time'].min()} a {df_cahuil['time'].max()}")
    
    # Cargar datos UTIDE
    print("Cargando datos UTIDE...")
    df_utide = cargar_utide_desde_nc(ARCHIVO_UTIDE)
    print(f"Datos UTIDE: {len(df_utide)} registros, {df_utide['time'].min()} a {df_utide['time'].max()}")
    
    # Calcular métricas de error
    mae, rmse, sesgo = calcular_metricas_error(df_cahuil, df_utide, tolerancia='5min')
    
    # Preparar datos del zoom
    fecha_zoom_inicio = pd.to_datetime(FECHA_INICIO_ZOOM)
    fecha_zoom_fin = fecha_zoom_inicio + pd.Timedelta(days=DIAS_ZOOM)
    mask_cahuil_zoom = (df_cahuil['time'] >= fecha_zoom_inicio) & (df_cahuil['time'] <= fecha_zoom_fin)
    mask_utide_zoom = (df_utide['time'] >= fecha_zoom_inicio) & (df_utide['time'] <= fecha_zoom_fin)
    df_cahuil_zoom = df_cahuil.loc[mask_cahuil_zoom].copy()
    df_utide_zoom = df_utide.loc[mask_utide_zoom].copy()
    
    # Crear figura con dos paneles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    
    # ==================== PANEL SUPERIOR (serie completa) ====================
    ax1.plot(df_cahuil['time'], df_cahuil['cahuil_estimada'],
             label='Cáhuil estimada', color='red', alpha=0.5, linewidth=1.5)
    ax1.plot(df_utide['time'], df_utide['altura_utide'],
             label='UTIDE reconstruida', color='blue', marker='.', markersize=1, linestyle='-', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Altura relativa (m)')
    ax1.set_title('Comparación: Marea estimada en Cáhuil vs reconstrucción UTIDE (serie completa)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recuadro con métricas (solo MAE, RMSE, Sesgo)
    texto_metricas = f"MAE: {mae:.4f} m\nRMSE: {rmse:.4f} m\nSesgo: {sesgo:.4f} m"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, texto_metricas, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=props)
    
    # ==================== PANEL INFERIOR (zoom) ====================
    ax2.plot(df_cahuil_zoom['time'], df_cahuil_zoom['cahuil_estimada'],
             label='Cáhuil estimada', color='red', alpha=0.7, linewidth=1.5)
    ax2.plot(df_utide_zoom['time'], df_utide_zoom['altura_utide'],
             label='UTIDE reconstruida', color='blue', marker='.', markersize=2, linestyle='-', linewidth=0.8, alpha=0.9)
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Altura relativa (m)')
    ax2.set_title(f'Zoom: {DIAS_ZOOM} días ({fecha_zoom_inicio.strftime("%Y-%m-%d")} a {fecha_zoom_fin.strftime("%Y-%m-%d")})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([fecha_zoom_inicio, fecha_zoom_fin])
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_guardado = os.path.join(CARPETA_RESULTADOS, f"comparacion_cahuil_utide_{timestamp}.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_guardado}")
    plt.show()

if __name__ == "__main__":
    graficar_comparacion_cahuil_utide()