import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def leer_archivo_marea(ruta_archivo):
    """Lee un archivo de marea con el formato específico"""
    try:
        # Leer el archivo con el formato conocido
        df = pd.read_csv(ruta_archivo, sep='\t', skiprows=2, header=None,
                        names=['datetime', 'prs', 'rad'], parse_dates=['datetime'])
        
        # Convertir a numérico y manejar valores no numéricos
        df['prs'] = pd.to_numeric(df['prs'], errors='coerce')
        df['rad'] = pd.to_numeric(df['rad'], errors='coerce')
        
        # Eliminar filas con NaN en datetime
        df = df.dropna(subset=['datetime'])
        
        return df
    
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return pd.DataFrame()

def procesar_archivos_marea(data_folder):
    """Procesa todos los archivos de marea en las subcarpetas y los combina"""
    # Listar y ordenar carpetas por fecha
    carpetas = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    
    # Ordenar carpetas por fecha
    def obtener_fecha_inicio(nombre_carpeta):
        try:
            return datetime.strptime(nombre_carpeta.split(' hasta ')[0], '%Y-%m-%d')
        except:
            return datetime.min
    
    carpetas_ordenadas = sorted(carpetas, key=obtener_fecha_inicio)
    
    # Listas para almacenar datos
    datos_san_antonio, datos_boyeruca = [], []
    
    for carpeta in carpetas_ordenadas:
        ruta_carpeta = os.path.join(data_folder, carpeta)
        
        # Procesar San Antonio
        archivo_sa = os.path.join(ruta_carpeta, "San_Antonio.txt")
        if os.path.exists(archivo_sa):
            df_sa = leer_archivo_marea(archivo_sa)
            if not df_sa.empty:
                print(f"San Antonio en {carpeta}: {len(df_sa)} registros")
                datos_san_antonio.append(df_sa)
            else:
                print(f"Advertencia: No se pudieron leer datos de {archivo_sa}")
        
        # Procesar Boyeruca
        archivo_boy = os.path.join(ruta_carpeta, "Boyeruca.txt")
        if os.path.exists(archivo_boy):
            df_boy = leer_archivo_marea(archivo_boy)
            if not df_boy.empty:
                print(f"Boyeruca en {carpeta}: {len(df_boy)} registros")
                datos_boyeruca.append(df_boy)
            else:
                print(f"Advertencia: No se pudieron leer datos de {archivo_boy}")
    
    # Combinar todos los datos
    df_sa_completo = pd.concat(datos_san_antonio, ignore_index=True) if datos_san_antonio else pd.DataFrame()
    df_boy_completo = pd.concat(datos_boyeruca, ignore_index=True) if datos_boyeruca else pd.DataFrame()
    
    # Eliminar duplicados y ordenar
    for df in [df_sa_completo, df_boy_completo]:
        if not df.empty:
            df.drop_duplicates('datetime', inplace=True)
            df.sort_values('datetime', inplace=True)
            df.reset_index(drop=True, inplace=True)
    
    return df_sa_completo, df_boy_completo

def guardar_netcdf(df, output_path, estacion_nombre):
    """Guarda los datos en formato NetCDF"""
    if df.empty:
        print(f"No hay datos para {estacion_nombre}")
        return
    
    # Asegurar que los datos son numéricos
    df['prs'] = pd.to_numeric(df['prs'], errors='coerce')
    df['rad'] = pd.to_numeric(df['rad'], errors='coerce')
    
    # Eliminar filas con valores NaN
    df = df.dropna()
    
    # Crear Dataset de xarray
    ds = xr.Dataset(
        {
            'prs': (['time'], df['prs'].astype(np.float32).values),
            'rad': (['time'], df['rad'].astype(np.float32).values)
        },
        coords={
            'time': df['datetime'].values
        }
    )
    
    # Agregar atributos
    ds.attrs = {
        'title': f'Serie temporal de marea - {estacion_nombre}',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Datos de marea brutos',
        'station': estacion_nombre
    }
    
    ds['prs'].attrs = {
        'units': 'meters',
        'long_name': 'Water surface height above datum (pressure sensor)'
    }
    
    ds['rad'].attrs = {
        'units': 'meters',
        'long_name': 'Water surface height above datum (radar sensor)'
    }
    
    # Guardar a NetCDF
    ds.to_netcdf(output_path)
    print(f"Datos de {estacion_nombre} guardados en: {output_path}")

def graficar_series_completas(df_sa, df_boy):
    """Grafica las series temporales completas de ambas estaciones con muestreo para mejor rendimiento"""
    if df_sa.empty or df_boy.empty:
        print("No hay datos suficientes para graficar")
        return
    
    # Muestrear los datos para hacer la gráfica manejable
    # Si hay más de 10,000 puntos, muestrear para mantener aproximadamente 10,000 puntos
    muestreo_sa = max(1, len(df_sa) // 10000)
    muestreo_boy = max(1, len(df_boy) // 10000)
    
    df_sa_muestreado = df_sa.iloc[::muestreo_sa]
    df_boy_muestreado = df_boy.iloc[::muestreo_boy]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Graficar San Antonio
    ax1.plot(df_sa_muestreado['datetime'], df_sa_muestreado['prs'], label='Presión', color='blue', alpha=0.7, linewidth=0.5)
    ax1.plot(df_sa_muestreado['datetime'], df_sa_muestreado['rad'], label='Radar', color='red', alpha=0.7, linewidth=0.5)
    ax1.set_title(f'San Antonio - Serie Temporal Completa (muestreado 1:{muestreo_sa})')
    ax1.set_ylabel('Altura (m)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Graficar Boyeruca
    ax2.plot(df_boy_muestreado['datetime'], df_boy_muestreado['prs'], label='Presión', color='blue', alpha=0.7, linewidth=0.5)
    ax2.plot(df_boy_muestreado['datetime'], df_boy_muestreado['rad'], label='Radar', color='red', alpha=0.7, linewidth=0.5)
    ax2.set_title(f'Boyeruca - Serie Temporal Completa (muestreado 1:{muestreo_boy})')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Altura (m)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Formatear eje de tiempo
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/series_temporales_completas.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Configuración inicial
    data_folder = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Data_mareas'
    output_san_antonio = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Serie_bruta_Zm_San_Antonio.nc'
    output_boyeruca = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Serie_bruta_Zm_Boyeruca.nc'
    
    # Procesar archivos
    print("Procesando archivos de marea...")
    df_sa, df_boy = procesar_archivos_marea(data_folder)
    
    # Verificar que hay datos
    if df_sa.empty or df_boy.empty:
        print("Error: No se encontraron datos suficientes")
    else:
        print(f"San Antonio total: {len(df_sa)} registros")
        print(f"Boyeruca total: {len(df_boy)} registros")
        
        # Guardar resultados
        guardar_netcdf(df_sa, output_san_antonio, "San_Antonio")
        guardar_netcdf(df_boy, output_boyeruca, "Boyeruca")
        
        # Graficar series completas
        print("Generando gráficos...")
        graficar_series_completas(df_sa, df_boy)
        
        print("Procesamiento completado!")