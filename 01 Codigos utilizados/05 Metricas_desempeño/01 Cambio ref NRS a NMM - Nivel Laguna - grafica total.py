import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
from datetime import datetime, timedelta

# =====================================================
# CONFIGURACIÓN (modifica estos parámetros según tu caso)
# =====================================================
# Archivos CSV de entrada (2023 y 2024)
ARCHIVO_CSV_2023 = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Nivel Laguna medidos\2023-Cahuil-Barrancas_Nivel_Salinidad.csv"
ARCHIVO_CSV_2024 = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Nivel Laguna medidos\2024-Cahuil-LaBalsa_Nivel_Salinidad.csv"

# Nombres de las columnas de nivel en cada archivo (difieren por estación)
COLUMNA_NIVEL_2023 = 'Nivel H2O [m] Barrancas'
COLUMNA_NIVEL_2024 = 'Nivel H2O [m] LaBalsa'

NRS = 0.9257                             # Valor fijo de NRS (en metros, sacado de analisas armónico matlab.)
INICIO = "01-01-23 00:00"                # Instante inicial (dd-mm-yy HH:MM)
FIN = "30-12-24 23:50"                   # Instante final (dd-mm-yy HH:MM)
CARPETA_SALIDA = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\05 Metricas de Error\Resultados Nivel Laguna NMM - data completa 2023-2024"  # Carpeta donde se guardarán los resultados

# Parámetros para la ventana de tiempo adicional
INICIO_VENTANA = "01-07-24 00:00"        # Inicio de la ventana para la segunda gráfica
DIAS_VENTANA = 30                         # Número de días a incluir en la ventana
# =====================================================

# Crear la carpeta de salida si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Lista de archivos a procesar con sus respectivas columnas de nivel
archivos_csv = [
    (ARCHIVO_CSV_2023, COLUMNA_NIVEL_2023),
    (ARCHIVO_CSV_2024, COLUMNA_NIVEL_2024)
]

# DataFrame que acumulará los datos de ambos años
df_combinado = pd.DataFrame()

# Procesar cada archivo
for ruta_csv, columna_nivel in archivos_csv:
    print(f"\nProcesando archivo: {ruta_csv}")
    
    # 1. Leer el CSV
    df = pd.read_csv(ruta_csv)
    
    # Verificar nombres de columnas (pueden tener espacios)
    df.columns = df.columns.str.strip()  # eliminar espacios al inicio/fin
    
    # Renombrar solo las columnas de fecha y nivel (ignoramos salinidad)
    # La columna de fecha se llama 'Date UTC' en ambos archivos
    df.rename(columns={
        'Date UTC': 'fecha',
        columna_nivel: 'nivel_nrs'
    }, inplace=True)
    
    # 2. Convertir la columna de fecha a datetime
    print("Convirtiendo fechas...")
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%y %H:%M', errors='coerce')
    # Eliminar filas con fecha no válida (por si acaso)
    df.dropna(subset=['fecha'], inplace=True)
    
    # 3. Filtrar por rango de fechas principal
    fecha_inicio = pd.to_datetime(INICIO, format='%d-%m-%y %H:%M')
    fecha_fin = pd.to_datetime(FIN, format='%d-%m-%y %H:%M')
    mask = (df['fecha'] >= fecha_inicio) & (df['fecha'] <= fecha_fin)
    df_filtrado = df.loc[mask].copy()
    
    if df_filtrado.empty:
        print(f"¡Advertencia! No hay datos en el rango especificado para {ruta_csv}.")
        continue
    else:
        print(f"Datos encontrados en {ruta_csv}: {len(df_filtrado)} registros.")
    
    # 4. Calcular nivel referido a NMM
    df_filtrado['nivel_nmm'] = df_filtrado['nivel_nrs'] - NRS
    
    # Agregar al DataFrame combinado
    df_combinado = pd.concat([df_combinado, df_filtrado], ignore_index=True)

# Ordenar por fecha (por si acaso los datos no vienen en orden)
if not df_combinado.empty:
    df_combinado.sort_values('fecha', inplace=True)
    df_combinado.reset_index(drop=True, inplace=True)
    print(f"\nTotal de registros combinados (2023-2024): {len(df_combinado)}")
else:
    print("¡No se encontraron datos en ninguno de los archivos!")

# 5. Guardar en NetCDF (solo niveles, sin salinidad)
if not df_combinado.empty:
    print("Guardando archivo NetCDF...")
    archivo_nc = os.path.join(CARPETA_SALIDA, "Nivel_laguna_ref_NMM.nc")
    
    # Crear archivo NetCDF
    with nc.Dataset(archivo_nc, 'w', format='NETCDF4') as ds:
        # Crear dimensión tiempo
        ds.createDimension('time', None)  # ilimitada
        
        # Crear variable tiempo (numérica, en segundos desde 1970-01-01)
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.units = 'seconds since 1970-01-01'
        time_var.calendar = 'gregorian'
        time_var.long_name = 'Tiempo (segundos desde 1970-01-01)'
        
        # Convertir fechas a segundos desde 1970
        referencia = datetime(1970, 1, 1)
        segundos = [(t - referencia).total_seconds() for t in df_combinado['fecha']]
        time_var[:] = segundos
        
        # Guardar también la fecha como string (para legibilidad)
        str_var = ds.createVariable('time_str', str, ('time',))
        str_var.long_name = 'Fecha y hora (formato ISO)'
        str_var[:] = df_combinado['fecha'].dt.strftime('%Y-%m-%d %H:%M').values
        
        # Crear variable nivel NRS
        nivel_nrs_var = ds.createVariable('nivel_nrs', 'f4', ('time',))
        nivel_nrs_var.units = 'm'
        nivel_nrs_var.long_name = 'Nivel de agua referido a NRS'
        nivel_nrs_var[:] = df_combinado['nivel_nrs'].values
        
        # Crear variable nivel NMM
        nivel_nmm_var = ds.createVariable('nivel_nmm', 'f4', ('time',))
        nivel_nmm_var.units = 'm'
        nivel_nmm_var.long_name = 'Nivel de agua referido a NMM'
        nivel_nmm_var[:] = df_combinado['nivel_nmm'].values
        
        # Atributos globales
        ds.title = 'Nivel de laguna referido a NMM'
        ds.source = f'Datos combinados de: {ARCHIVO_CSV_2023} y {ARCHIVO_CSV_2024}'
        ds.history = f'Creado el {datetime.now().isoformat()}'
        ds.comment = f'Conversión usando NRS = {NRS} m'
    
    print(f"Archivo NetCDF guardado en: {archivo_nc}")

# 6. Graficar (primer gráfico: rango completo)
if not df_combinado.empty:
    print("Generando gráfico del período completo (con puntos)...")
    plt.figure(figsize=(12, 6))
    # Usar marcador '.' para puntos, sin línea
    plt.plot(df_combinado['fecha'], df_combinado['nivel_nrs'], '.', label='Nivel (ref NRS)', color='blue', markersize=2)
    plt.plot(df_combinado['fecha'], df_combinado['nivel_nmm'], '.', label='Nivel (ref NMM)', color='red', markersize=2)
    plt.xlabel('Fecha')
    plt.ylabel('Nivel de agua [m]')
    plt.title(f'Comparación de niveles: NRS vs NMM\nPeriodo: {INICIO} a {FIN}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar gráfico en alta calidad
    archivo_png = os.path.join(CARPETA_SALIDA, "comparacion_niveles_completo.png")
    plt.savefig(archivo_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico completo guardado en: {archivo_png}")
else:
    print("No se generó gráfico del período completo por falta de datos.")

# 7. Graficar ventana de tiempo adicional
if not df_combinado.empty:
    print("Generando gráfico de la ventana de tiempo (con puntos)...")
    # Calcular fecha fin de la ventana
    fecha_inicio_ventana = pd.to_datetime(INICIO_VENTANA, format='%d-%m-%y %H:%M')
    fecha_fin_ventana = fecha_inicio_ventana + timedelta(days=DIAS_VENTANA)
    
    # Filtrar datos dentro de la ventana
    mask_ventana = (df_combinado['fecha'] >= fecha_inicio_ventana) & (df_combinado['fecha'] <= fecha_fin_ventana)
    df_ventana = df_combinado.loc[mask_ventana].copy()
    
    if df_ventana.empty:
        print(f"Advertencia: No hay datos en la ventana {INICIO_VENTANA} a {fecha_fin_ventana.strftime('%d-%m-%y %H:%M')}")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(df_ventana['fecha'], df_ventana['nivel_nrs'], '.', label='Nivel (ref NRS)', color='blue', markersize=4)
        plt.plot(df_ventana['fecha'], df_ventana['nivel_nmm'], '.', label='Nivel (ref NMM)', color='red', markersize=4)
        plt.xlabel('Fecha')
        plt.ylabel('Nivel de agua [m]')
        plt.title(f'Comparación de niveles: NRS vs NMM\nVentana: {INICIO_VENTANA} a {(fecha_inicio_ventana + timedelta(days=DIAS_VENTANA)).strftime("%d-%m-%y %H:%M")}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico en alta calidad
        archivo_png_ventana = os.path.join(CARPETA_SALIDA, f"comparacion_niveles_ventana_{DIAS_VENTANA}dias.png")
        plt.savefig(archivo_png_ventana, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de ventana guardado en: {archivo_png_ventana}")
else:
    print("No se generó gráfico de ventana por falta de datos en el rango principal.")

print("\n¡Proceso completado!")