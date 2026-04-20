"""
Script para graficar resultados de mareas a partir del archivo NetCDF generado.
Gráfica 1: Datos brutos y promedios móviles de Boyeruca y San Antonio.
Gráfica 2: Mareas relativas de Boyeruca, San Antonio y Cáhuil estimada.
Gráfica 3: Zoom de las mareas relativas para un día específico (Dia_zoom).
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os

# ========================= CONFIGURACIÓN DE RUTAS =========================
ARCHIVO_RESULTADOS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas/Resultado_mareas_estimadas_10min.nc'
GRAFICO_SALIDA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas/Grafico_resultados_mareas_10min.png'
# Día para el zoom (formato YYYY-MM-DD)
Dia_zoom = "2024-07-10"
# =========================================================================

def graficar_resultados_desde_netcdf(ruta_netcdf, ruta_grafico, dia_zoom):
    """
    Carga el archivo NetCDF con los resultados y genera las gráficas.
    """
    # Cargar dataset
    ds = xr.open_dataset(ruta_netcdf)
    
    # Convertir a DataFrame para facilitar el manejo
    df = ds.to_dataframe().reset_index()
    
    # Convertir columna time a datetime si no lo está
    df['time'] = pd.to_datetime(df['time'])
    
    # Crear figura con tres subplots verticales
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # ----- GRÁFICA 1: Datos brutos y promedios móviles -----
    ax1.plot(df['time'], df['boyeruca_bruto'],
             label='Boyeruca bruto', linewidth=0.8, alpha=0.6, color='C0')
    ax1.plot(df['time'], df['san_antonio_bruto'],
             label='San Antonio bruto', linewidth=0.8, alpha=0.6, color='C1')
    ax1.plot(df['time'], df['promedio_movil_boyeruca'],
             label='Prom. móvil Boyeruca', linewidth=1.5, color='C0', linestyle='--')
    ax1.plot(df['time'], df['promedio_movil_san_antonio'],
             label='Prom. móvil San Antonio', linewidth=1.5, color='C1', linestyle='--')
    ax1.set_ylabel('Altura (m)')
    ax1.set_title('Datos brutos y promedios móviles de 7 días (resolución 10 min)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----- GRÁFICA 2: Mareas relativas (todo el período) -----
    ax2.plot(df['time'], df['altura_boyeruca_relativa'],
             label='Boyeruca (23.25 km)', linewidth=1.0, color='C0')
    ax2.plot(df['time'], df['altura_san_antonio_relativa'],
             label='San Antonio (106.71 km)', linewidth=1.0, color='C1')
    ax2.plot(df['time'], df['altura_cahuil_estimada'],
             label='Cáhuil estimada', linewidth=1.5, color='red', linestyle='-')
    ax2.set_ylabel('Altura relativa (m)')
    ax2.set_xlabel('Tiempo')
    ax2.set_title('Mareas relativas al promedio móvil de 7 días (todo el período)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ----- GRÁFICA 3: Zoom de mareas relativas para el día especificado -----
    # Convertir día a fecha para filtrado
    fecha_zoom = pd.to_datetime(dia_zoom).date()
    # Filtrar datos para ese día
    mask = df['time'].dt.date == fecha_zoom
    df_zoom = df.loc[mask].copy()
    
    if df_zoom.empty:
        print(f"Advertencia: No hay datos para el día {dia_zoom}. El gráfico de zoom estará vacío.")
    else:
        ax3.plot(df_zoom['time'], df_zoom['altura_boyeruca_relativa'],
                 label='Boyeruca (23.25 km)', linewidth=1.0, color='C0')
        ax3.plot(df_zoom['time'], df_zoom['altura_san_antonio_relativa'],
                 label='San Antonio (106.71 km)', linewidth=1.0, color='C1')
        ax3.plot(df_zoom['time'], df_zoom['altura_cahuil_estimada'],
                 label='Cáhuil estimada', linewidth=1.5, color='red', linestyle='-')
    
    ax3.set_ylabel('Altura relativa (m)')
    ax3.set_xlabel('Tiempo')
    ax3.set_title(f'Zoom: Mareas relativas para el día {dia_zoom}')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar con alta resolución
    os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {ruta_grafico}")
    plt.show()

if __name__ == "__main__":
    graficar_resultados_desde_netcdf(ARCHIVO_RESULTADOS, GRAFICO_SALIDA, Dia_zoom)