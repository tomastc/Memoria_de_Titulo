import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import itertools
import time
from typing import Optional

def obtener_matriz_coordenadas(ruta_archivo: str):
    """
    Lee las coordenadas desde el archivo TXT
    """
    estaciones = []
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith('#') and not linea.startswith('---'):
                    partes = linea.split(';')
                    if len(partes) == 3:
                        nombre = partes[0].strip()
                        lat = float(partes[1].strip())
                        lon = float(partes[2].strip())
                        estaciones.append((nombre, (lat, lon)))
        return estaciones
    except Exception as e:
        print(f"Error al leer coordenadas: {e}")
        return []

def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula del haversine
    """
    # Radio de la Tierra en km
    r = 6371.0
    
    # Convertir grados a radianes
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Diferencias
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Fórmula del haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return r * c

def calcular_distancias_estaciones(ruta_archivo: str) -> Optional[pd.DataFrame]:
    """
    Calcula las distancias entre todos los pares de estaciones
    """
    estaciones = obtener_matriz_coordenadas(ruta_archivo)
    
    if not estaciones or len(estaciones) < 2:
        print("Error: Se necesitan al menos 2 estaciones válidas para calcular distancias.")
        return None
    
    resultados = []
    
    for (estacion1, estacion2) in itertools.combinations(estaciones, 2):
        try:
            nombre1, (lat1, lon1) = estacion1
            nombre2, (lat2, lon2) = estacion2
            
            dist = distancia_haversine(lat1, lon1, lat2, lon2)
            resultados.append({
                'Estacion_Origen': nombre1,
                'Estacion_Destino': nombre2,
                'Distancia_km': round(dist, 4)
            })
        except (ValueError, TypeError) as e:
            print(f"Error al procesar el par ({nombre1}, {nombre2}): {e}")
            continue
            
    df = pd.DataFrame(resultados)
    
    if df.empty:
        print("No se pudieron calcular distancias para ningún par de estaciones.")
        return None
    
    df = df.sort_values(by=['Estacion_Origen', 'Estacion_Destino']).reset_index(drop=True)
    return df

def cargar_datos_netcdf(archivo_boyeruca, archivo_san_antonio):
    """
    Carga los archivos NetCDF de Boyeruca y San Antonio
    """
    try:
        ds_boyeruca = xr.open_dataset(archivo_boyeruca)
        ds_san_antonio = xr.open_dataset(archivo_san_antonio)
        print("Datos NetCDF cargados exitosamente")
        return ds_boyeruca, ds_san_antonio
    except Exception as e:
        raise ValueError(f"Error al cargar archivos NetCDF: {str(e)}")

def extraer_serie_temporal(ds, nombre_estacion):
    """
    Extrae la serie temporal de marea de un dataset NetCDF
    """
    # Buscar la variable de altura (puede ser 'Zm', 'altura', 'marea', etc.)
    variables_posibles = ['Zm', 'altura', 'marea', 'water_level', 'sea_surface_height']
    variable_encontrada = None
    
    for var in variables_posibles:
        if var in ds.variables:
            variable_encontrada = var
            break
    
    if variable_encontrada is None:
        # Tomar la primera variable de datos que no sea una coordenada
        data_vars = list(ds.data_vars)
        if data_vars:
            variable_encontrada = data_vars[0]
            print(f"Usando variable '{variable_encontrada}' para {nombre_estacion}")
        else:
            raise ValueError(f"No se encontraron variables de datos en el dataset de {nombre_estacion}")
    
    datos = ds[variable_encontrada].values
    
    # Extraer tiempos
    if 'time' in ds.coords:
        tiempos = pd.to_datetime(ds['time'].values)
    else:
        # Buscar coordenada de tiempo
        for coord in ds.coords:
            if 'time' in coord.lower():
                tiempos = pd.to_datetime(ds[coord].values)
                break
        else:
            raise ValueError(f"No se encontró coordenada de tiempo en el dataset de {nombre_estacion}")
    
    return pd.Series(datos, index=tiempos, name=nombre_estacion)

def calcular_promedio_movil_7dias(serie, fecha_central):
    """
    Calcula el promedio móvil de 7 días (3 días antes + 3 días después + día central)
    """
    fecha_inicio = fecha_central - timedelta(days=3)
    fecha_fin = fecha_central + timedelta(days=3)
    
    # Filtrar serie para el rango de 7 días
    mascara = (serie.index >= fecha_inicio) & (serie.index <= fecha_fin)
    datos_7dias = serie[mascara]
    
    if len(datos_7dias) == 0:
        return np.nan
    
    return datos_7dias.mean()

def interpolar_minuto_exacto(serie, tiempo_objetivo, ventana_minutos=10):
    """
    Interpola linealmente para obtener el valor en un tiempo exacto
    usando datos dentro de una ventana de tiempo alrededor del tiempo objetivo
    """
    # Crear rango de tiempo alrededor del tiempo objetivo
    tiempo_antes = tiempo_objetivo - timedelta(minutes=ventana_minutos)
    tiempo_despues = tiempo_objetivo + timedelta(minutes=ventana_minutos)
    
    # Buscar datos cercanos en la ventana
    datos_cercanos = serie[(serie.index >= tiempo_antes) & (serie.index <= tiempo_despues)]
    
    if len(datos_cercanos) < 2:
        # Si no hay suficientes datos, intentar con una ventana más amplia
        tiempo_antes = tiempo_objetivo - timedelta(minutes=30)
        tiempo_despues = tiempo_objetivo + timedelta(minutes=30)
        datos_cercanos = serie[(serie.index >= tiempo_antes) & (serie.index <= tiempo_despues)]
        
        if len(datos_cercanos) < 2:
            return np.nan
    
    # Ordenar por tiempo
    datos_cercanos = datos_cercanos.sort_index()
    
    # Interpolación lineal
    try:
        valor_interpolado = np.interp(
            tiempo_objetivo.timestamp(), 
            [t.timestamp() for t in datos_cercanos.index], 
            datos_cercanos.values
        )
        return valor_interpolado
    except:
        return np.nan

def procesar_cada_10_minutos(serie_boyeruca, serie_san_antonio):
    """
    Procesa los datos cada 10 minutos (00:00:00, 00:10:00, 00:20:00, etc.)
    en todo el dominio común entre ambas estaciones
    """
    # Determinar rango de fechas común (redondeado a minutos)
    fecha_inicio = max(serie_boyeruca.index.min(), serie_san_antonio.index.min())
    fecha_fin = min(serie_boyeruca.index.max(), serie_san_antonio.index.max())
    
    # Redondear al múltiplo de 10 minutos más cercano
    # Para el inicio: redondear hacia arriba al próximo múltiplo de 10 minutos
    minutos_inicio = fecha_inicio.minute
    segundos_inicio = fecha_inicio.second
    microsegundos_inicio = fecha_inicio.microsecond
    
    # Calcular los segundos para sumar para llegar al próximo múltiplo de 10 minutos
    minutos_a_sumar = (10 - (minutos_inicio % 10)) % 10
    if minutos_a_sumar == 0 and (segundos_inicio > 0 or microsegundos_inicio > 0):
        minutos_a_sumar = 10
    
    fecha_inicio = fecha_inicio.replace(second=0, microsecond=0) + timedelta(minutes=minutos_a_sumar)
    
    # Para el fin: redondear hacia abajo al múltiplo de 10 minutos anterior
    minutos_fin = fecha_fin.minute
    segundos_fin = fecha_fin.second
    microsegundos_fin = fecha_fin.microsecond
    
    # Restar los segundos y microsegundos, y ajustar minutos si es necesario
    fecha_fin = fecha_fin.replace(second=0, microsecond=0)
    if segundos_fin > 0 or microsegundos_fin > 0:
        fecha_fin = fecha_fin - timedelta(minutes=(minutos_fin % 10))
    
    print(f"Rango de fechas común (cada 10 minutos): {fecha_inicio} a {fecha_fin}")
    
    # Generar serie de tiempos cada 10 minutos en el rango común
    tiempos_10min = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='10min')
    
    print(f"Se procesarán {len(tiempos_10min)} intervalos de 10 minutos (00:00:00, 00:10:00, 00:20:00, etc.)")
    
    # Variables para calcular velocidad y tiempo estimado
    tiempo_inicio_total = time.time()
    tiempo_primer_lote = None
    velocidad_calculada = False
    tiempo_estimado_restante = None
    
    # Procesar cada intervalo de 10 minutos
    resultados = []
    
    for i, tiempo_exacto in enumerate(tiempos_10min):
        # Primer lote de 1000 datos - Calcular velocidad y tiempo estimado
        if i == 1000 and not velocidad_calculada:
            tiempo_actual = time.time()
            tiempo_transcurrido_primer_lote = tiempo_actual - tiempo_inicio_total
            
            # Calcular velocidad (datos por segundo)
            velocidad = 1000 / tiempo_transcurrido_primer_lote if tiempo_transcurrido_primer_lote > 0 else 0
            
            # Calcular tiempo estimado restante
            datos_restantes = len(tiempos_10min) - i
            tiempo_estimado_restante = datos_restantes / velocidad if velocidad > 0 else 0
            
            # Calcular porcentaje completado
            porcentaje_completado = (i / len(tiempos_10min)) * 100
            
            # Mostrar información del primer lote
            print(f"\n--- PRIMER LOTE ANALIZADO (1000 datos) ---")
            print(f"Velocidad de procesamiento: {velocidad:.2f} datos/segundo")
            print(f"Tiempo estimado restante: {timedelta(seconds=int(tiempo_estimado_restante))}")
            print(f"Tiempo total estimado: {timedelta(seconds=int(tiempo_transcurrido_primer_lote + tiempo_estimado_restante))}")
            print("(Ahora se mostrará avance cada 10,000 datos)\n")
            
            velocidad_calculada = True
            tiempo_primer_lote = tiempo_transcurrido_primer_lote
        
        # Mostrar avance cada 10,000 datos (después del primer lote)
        elif i % 10000 == 0 and i > 0 and velocidad_calculada:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_inicio_total
            porcentaje_completado = (i / len(tiempos_10min)) * 100
            
            # Si tenemos tiempo estimado, calcular tiempo restante basado en progreso
            if tiempo_estimado_restante:
                tiempo_restante_ajustado = tiempo_estimado_restante * (1 - porcentaje_completado/100)
                print(f"Procesando intervalo {i+1}/{len(tiempos_10min)} - {porcentaje_completado:.1f}% completado")
                print(f"Tiempo transcurrido: {timedelta(seconds=int(tiempo_transcurrido))}")
                print(f"Tiempo restante estimado: {timedelta(seconds=int(tiempo_restante_ajustado))}")
            else:
                print(f"Procesando intervalo {i+1}/{len(tiempos_10min)} - {porcentaje_completado:.1f}% completado")
                print(f"Tiempo transcurrido: {timedelta(seconds=int(tiempo_transcurrido))}")
        
        # Interpolar valores en tiempo exacto
        boyeruca_bruto = interpolar_minuto_exacto(serie_boyeruca, tiempo_exacto)
        san_antonio_bruto = interpolar_minuto_exacto(serie_san_antonio, tiempo_exacto)
        
        if np.isnan(boyeruca_bruto) or np.isnan(san_antonio_bruto):
            continue
        
        # Calcular promedios móviles de 7 días
        prom_movil_boyeruca = calcular_promedio_movil_7dias(serie_boyeruca, tiempo_exacto)
        prom_movil_san_antonio = calcular_promedio_movil_7dias(serie_san_antonio, tiempo_exacto)
        
        if np.isnan(prom_movil_boyeruca) or np.isnan(prom_movil_san_antonio):
            continue
        
        # Calcular mareas relativas (bruto - promedio móvil)
        boyeruca_relativa = boyeruca_bruto - prom_movil_boyeruca
        san_antonio_relativa = san_antonio_bruto - prom_movil_san_antonio
        
        resultados.append({
            'time': tiempo_exacto,
            'boyeruca_bruto': boyeruca_bruto,
            'san_antonio_bruto': san_antonio_bruto,
            'prom_movil_boyeruca': prom_movil_boyeruca,
            'prom_movil_san_antonio': prom_movil_san_antonio,
            'boyeruca_relativa': boyeruca_relativa,
            'san_antonio_relativa': san_antonio_relativa
        })
    
    # Calcular estadísticas finales
    tiempo_total = time.time() - tiempo_inicio_total
    velocidad_promedio = len(resultados) / tiempo_total if tiempo_total > 0 else 0
    
    print(f"\n--- RESUMEN DE PROCESAMIENTO ---")
    print(f"Procesamiento completado en {timedelta(seconds=int(tiempo_total))}")
    
    # Mostrar comparación con estimación inicial si está disponible
    if tiempo_primer_lote and tiempo_estimado_restante:
        tiempo_estimado_inicial = tiempo_primer_lote + tiempo_estimado_restante
        diferencia = tiempo_total - tiempo_estimado_inicial
        print(f"Tiempo estimado inicialmente: {timedelta(seconds=int(tiempo_estimado_inicial))}")
        print(f"Diferencia con estimación: {timedelta(seconds=int(abs(diferencia)))} "
              f"({'menos' if diferencia < 0 else 'más'} de lo estimado)")
    
    print(f"Velocidad promedio: {velocidad_promedio:.2f} datos/segundo")
    print(f"Total de intervalos de 10 minutos válidos obtenidos: {len(resultados)}")
    
    df_resultado = pd.DataFrame(resultados)
    return df_resultado

def obtener_distancias_cahuil(df_distancias):
    """
    Obtiene las distancias relevantes para Cáhuil desde el DataFrame de distancias
    """
    distancias = {}
    
    # Buscar distancia Boyeruca - Cahuil
    mask1 = (df_distancias['Estacion_Origen'] == 'Boyeruca_CL') & (df_distancias['Estacion_Destino'] == 'Cahuil')
    mask2 = (df_distancias['Estacion_Origen'] == 'Cahuil') & (df_distancias['Estacion_Destino'] == 'Boyeruca_CL')
    
    if any(mask1):
        distancias['boyeruca_cahuil'] = df_distancias.loc[mask1, 'Distancia_km'].values[0]
    elif any(mask2):
        distancias['boyeruca_cahuil'] = df_distancias.loc[mask2, 'Distancia_km'].values[0]
    else:
        raise ValueError("No se encontró distancia entre Boyeruca y Cahuil")
    
    # Buscar distancia San Antonio - Cahuil
    mask1 = (df_distancias['Estacion_Origen'] == 'San_Antonio_CL') & (df_distancias['Estacion_Destino'] == 'Cahuil')
    mask2 = (df_distancias['Estacion_Origen'] == 'Cahuil') & (df_distancias['Estacion_Destino'] == 'San_Antonio_CL')
    
    if any(mask1):
        distancias['san_antonio_cahuil'] = df_distancias.loc[mask1, 'Distancia_km'].values[0]
    elif any(mask2):
        distancias['san_antonio_cahuil'] = df_distancias.loc[mask2, 'Distancia_km'].values[0]
    else:
        raise ValueError("No se encontró distancia entre San Antonio y Cáhuil")
    
    print(f"Distancias obtenidas: Boyeruca-Cahuil = {distancias['boyeruca_cahuil']} km, "
          f"San_Antonio-Cahuil = {distancias['san_antonio_cahuil']} km")
    
    return distancias

def estimar_marea_cahuil(df_procesado, distancias):
    """
    Estima la marea en Cáhuil usando ponderación inversa por distancia
    """
    boyeruca_cahuil = distancias['boyeruca_cahuil']
    san_antonio_cahuil = distancias['san_antonio_cahuil']
    
    # Calcular pesos inversos
    peso_boyeruca = 1 / boyeruca_cahuil
    peso_san_antonio = 1 / san_antonio_cahuil
    peso_total = peso_boyeruca + peso_san_antonio
    
    print(f"Pesos: Boyeruca = {peso_boyeruca:.6f}, San Antonio = {peso_san_antonio:.6f}, Total = {peso_total:.6f}")
    
    # Estimar marea en Cáhuil usando ponderación inversa
    df_procesado['cahuil_estimada'] = (
        (peso_boyeruca * df_procesado['boyeruca_relativa'] + 
         peso_san_antonio * df_procesado['san_antonio_relativa']) / peso_total
    )
    
    return df_procesado

def guardar_netcdf_resultados(df_final, archivo_salida):
    """
    Guarda los resultados en un archivo NetCDF
    """
    # Crear dataset de xarray
    ds_salida = xr.Dataset({
        'altura_boyeruca_relativa': (['time'], df_final['boyeruca_relativa'].values),
        'altura_san_antonio_relativa': (['time'], df_final['san_antonio_relativa'].values),
        'altura_cahuil_estimada': (['time'], df_final['cahuil_estimada'].values),
        'boyeruca_bruto': (['time'], df_final['boyeruca_bruto'].values),
        'san_antonio_bruto': (['time'], df_final['san_antonio_bruto'].values),
        'promedio_movil_boyeruca': (['time'], df_final['prom_movil_boyeruca'].values),
        'promedio_movil_san_antonio': (['time'], df_final['prom_movil_san_antonio'].values)
    }, coords={
        'time': df_final['time'].values
    })
    
    # Añadir atributos
    ds_salida.attrs = {
        'title': 'Resultado mareas estimadas - Alturas relativas de marea (cada 10 minutos)',
        'description': 'Alturas de marea relativas al promedio móvil de 7 días (3 días antes + 3 días después)',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'author': 'Procesamiento automático',
        'estaciones': 'Boyeruca, San Antonio, Cáhuil (estimada)',
        'frecuencia_temporal': '10 minutos (00:00:00, 00:10:00, 00:20:00, etc.)',
        'metodo': 'Ponderación inversa por distancia',
        'interpolacion': 'Lineal en ventana de ±10 minutos'
    }
    
    # Añadir atributos a cada variable
    ds_salida['altura_boyeruca_relativa'].attrs = {'units': 'm', 'long_name': 'Altura de marea relativa Boyeruca'}
    ds_salida['altura_san_antonio_relativa'].attrs = {'units': 'm', 'long_name': 'Altura de marea relativa San Antonio'}
    ds_salida['altura_cahuil_estimada'].attrs = {'units': 'm', 'long_name': 'Altura de marea estimada Cáhuil'}
    
    # Guardar a NetCDF
    ds_salida.to_netcdf(archivo_salida)
    print(f"Resultados guardados en: {archivo_salida}")

def graficar_resultados(df_final, archivo_grafico):
    """
    Genera gráficos de los resultados:
    - Superior: Datos brutos y promedios móviles de Boyeruca y San Antonio (todo el período).
    - Inferior: Mareas relativas de Boyeruca, San Antonio y Cáhuil estimada (todo el período).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # ===== GRÁFICA 1: Datos brutos y promedios móviles =====
    ax1.plot(df_final['time'], df_final['boyeruca_bruto'],
             label='Boyeruca bruto', linewidth=0.8, alpha=0.6, color='C0')
    ax1.plot(df_final['time'], df_final['san_antonio_bruto'],
             label='San Antonio bruto', linewidth=0.8, alpha=0.6, color='C1')
    ax1.plot(df_final['time'], df_final['prom_movil_boyeruca'],
             label='Prom. móvil Boyeruca', linewidth=1.5, color='C0', linestyle='--')
    ax1.plot(df_final['time'], df_final['prom_movil_san_antonio'],
             label='Prom. móvil San Antonio', linewidth=1.5, color='C1', linestyle='--')
    ax1.set_ylabel('Altura (m)')
    ax1.set_title('Datos brutos y promedios móviles de 7 días (resolución 10 min)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ===== GRÁFICA 2: Mareas relativas =====
    ax2.plot(df_final['time'], df_final['boyeruca_relativa'],
             label='Boyeruca relativa', linewidth=1.0, color='C0')
    ax2.plot(df_final['time'], df_final['san_antonio_relativa'],
             label='San Antonio relativa', linewidth=1.0, color='C1')
    ax2.plot(df_final['time'], df_final['cahuil_estimada'],
             label='Cáhuil estimada', linewidth=1.5, color='red', linestyle='-')
    ax2.set_ylabel('Altura relativa (m)')
    ax2.set_xlabel('Tiempo')
    ax2.set_title('Mareas relativas al promedio móvil de 7 días (todo el período)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {archivo_grafico}")
    plt.show()

def main():
    """
    Función principal que ejecuta todo el procesamiento
    """
    # Configuración de rutas
    RUTA_BOYERUCA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Serie_bruta_Zm_Boyeruca.nc'
    RUTA_SAN_ANTONIO = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Serie_bruta_Zm_San_Antonio.nc'
    RUTA_COORDENADAS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Coordenadas_estaciones_IOC.txt'
    ARCHIVO_SALIDA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas/Resultado_mareas_estimadas_10min.nc'
    GRAFICO_SALIDA = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas/Grafico_resultados_mareas_10min.png'
    
    try:
        # Asegurarse de que el directorio de salida exista
        os.makedirs(os.path.dirname(ARCHIVO_SALIDA), exist_ok=True)

        print("=== INICIANDO PROCESAMIENTO DE MAREAS (CADA 10 MINUTOS) ===")
        tiempo_inicio_total = time.time()
        
        # 1. Calcular distancias entre estaciones
        print("\n1. Calculando distancias entre estaciones...")
        df_distancias = calcular_distancias_estaciones(RUTA_COORDENADAS)
        if df_distancias is None:
            raise ValueError("No se pudieron calcular las distancias")
        print("Distancias calculadas exitosamente")
        print(df_distancias)
        
        # 2. Obtener distancias específicas para Cáhuil
        distancias_cahuil = obtener_distancias_cahuil(df_distancias)
        
        # 3. Cargar datos NetCDF
        print("\n2. Cargando datos NetCDF...")
        ds_boyeruca, ds_san_antonio = cargar_datos_netcdf(RUTA_BOYERUCA, RUTA_SAN_ANTONIO)
        
        # 4. Extraer series temporales
        print("\n3. Extrayendo series temporales...")
        serie_boyeruca = extraer_serie_temporal(ds_boyeruca, "Boyeruca")
        serie_san_antonio = extraer_serie_temporal(ds_san_antonio, "San_Antonio")
        
        print(f"Boyeruca: {len(serie_boyeruca)} registros, desde {serie_boyeruca.index.min()} hasta {serie_boyeruca.index.max()}")
        print(f"San Antonio: {len(serie_san_antonio)} registros, desde {serie_san_antonio.index.min()} hasta {serie_san_antonio.index.max()}")
        
        # 5. Procesar cada 10 minutos (00:00:00, 00:10:00, etc.)
        print("\n4. Procesando datos cada 10 minutos...")
        print("(Se calculará velocidad y tiempo estimado con los primeros 1000 datos)")
        print("(Luego se mostrará avance cada 10,000 datos)\n")
        df_procesado = procesar_cada_10_minutos(serie_boyeruca, serie_san_antonio)
        
        if len(df_procesado) == 0:
            raise ValueError("No se obtuvieron datos válidos después del procesamiento")
        
        # 6. Estimar marea en Cáhuil
        print("\n5. Estimando marea en Cáhuil...")
        df_final = estimar_marea_cahuil(df_procesado, distancias_cahuil)
        
        # 7. Calcular e imprimir promedios de toda la data (solo series brutas y estimada)
        print("\n--- PROMEDIOS DE MAREA DE TODA LA DATA ---")
        print(f"Promedio de marea bruta en Boyeruca: {df_final['boyeruca_bruto'].mean():.4f} m")
        print(f"Promedio de marea bruta en San Antonio: {df_final['san_antonio_bruto'].mean():.4f} m")
        print(f"Promedio de marea estimada en Cáhuil: {df_final['cahuil_estimada'].mean():.4f} m")
        
        # 8. Guardar resultados
        print("\n6. Guardando resultados en NetCDF...")
        guardar_netcdf_resultados(df_final, ARCHIVO_SALIDA)
        
        # 9. Generar gráficos
        print("\n7. Generando gráficos...")
        graficar_resultados(df_final, GRAFICO_SALIDA)
        
        tiempo_total = time.time() - tiempo_inicio_total
        
        print("\n=== PROCESAMIENTO COMPLETADO EXITOSAMENTE ===")
        print(f"Tiempo total de ejecución: {timedelta(seconds=int(tiempo_total))}")
        print(f"Archivo de salida: {ARCHIVO_SALIDA}")
        print(f"Total de intervalos de 10 minutos procesados: {len(df_final)}")
        print(f"Período: {df_final['time'].min()} a {df_final['time'].max()}")
        print(f"Frecuencia temporal: 10 minutos")
        print(f"Reducción de datos: aproximadamente 1/10 del original (más ágil)")
        
    except Exception as e:
        print(f"\n*** ERROR EN EL PROCESAMIENTO: {str(e)} ***")
        raise

if __name__ == "__main__":
    main()