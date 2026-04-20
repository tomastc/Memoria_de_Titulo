"""
Verificación de estimación de mareas en Boyeruca usando San Antonio y Concepción.
Datos cada 10 minutos, obtenidos desde archivos de texto con formato:
- Primera línea: nombre de la estación
- Segunda línea: encabezados: "Time (UTC)	prs(m)	rad(m)"
- Datos: timestamp, presión, altura rad(m)
Se utiliza la columna 'prs(m)' como altura de marea.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math

# ==================== CONFIGURACIÓN DE RUTAS ====================
CARPETA_DATOS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Codes extras'
RUTA_BOYERUCA_TXT = os.path.join(CARPETA_DATOS, 'Boyeruca_CL.txt')      # Ajustar nombre si es necesario
RUTA_SAN_ANTONIO_TXT = os.path.join(CARPETA_DATOS, 'San_Antonio_CL.txt') # Ajustar nombre
RUTA_CONCEPCION_TXT = os.path.join(CARPETA_DATOS, 'Constitucion_CL.txt')   # Ajustar nombre
RUTA_COORDENADAS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Coordenadas_estaciones_IOC.txt'
CARPETA_RESULTADOS = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/02 Mareas/Resultados_mareas'
CARPETA_VERIFICACION = os.path.join(CARPETA_RESULTADOS, 'verificacion')
# ================================================================

def cargar_serie_desde_txt(ruta_archivo, nombre_estacion):
    """
    Lee un archivo de texto con el formato:
    Línea 1: Nombre estación (se ignora)
    Línea 2: Encabezados: "Time (UTC)	prs(m)	rad(m)"
    Resto: datos tabulados con timestamp, presión y rad.
    Devuelve una Serie de pandas con índice datetime y valores de prs(m).
    """
    try:
        # Leer el archivo saltando la primera línea (nombre) y usando la segunda como encabezado
        df = pd.read_csv(ruta_archivo, sep='\t', skiprows=1, header=0,
                         names=['Time', 'prs', 'rad'],
                         parse_dates=['Time'], dayfirst=False)
        # Verificar que las columnas existen
        if 'Time' not in df.columns or 'prs' not in df.columns:
            raise ValueError(f"El archivo {ruta_archivo} no tiene las columnas esperadas (Time, prs).")
        # Convertir a serie con índice temporal usando la columna 'prs'
        serie = df.set_index('Time')['prs'].sort_index()
        # Eliminar posibles duplicados de índice (conservar el primero)
        serie = serie[~serie.index.duplicated(keep='first')]
        print(f"{nombre_estacion}: {len(serie)} registros, desde {serie.index.min()} hasta {serie.index.max()}")
        return serie
    except Exception as e:
        raise ValueError(f"Error al cargar {ruta_archivo}: {str(e)}")

def obtener_matriz_coordenadas(ruta_archivo: str):
    """Lee las coordenadas desde el archivo TXT"""
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
    """Distancia entre dos puntos geográficos (fórmula del haversine)"""
    r = 6371.0  # Radio Tierra en km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return r * c

def obtener_distancia_entre(estaciones, nombre1, nombre2):
    """Busca la distancia entre dos estaciones en la lista de coordenadas"""
    coord1 = None
    coord2 = None
    for nombre, coord in estaciones:
        if nombre == nombre1:
            coord1 = coord
        elif nombre == nombre2:
            coord2 = coord
    if coord1 is None or coord2 is None:
        raise ValueError(f"No se encontraron coordenadas para {nombre1} o {nombre2}")
    return distancia_haversine(coord1[0], coord1[1], coord2[0], coord2[1])

def interpolar_minuto_exacto(serie, tiempo_objetivo, ventana_minutos=10):
    """Interpola linealmente para obtener el valor en un tiempo exacto"""
    tiempo_antes = tiempo_objetivo - timedelta(minutes=ventana_minutos)
    tiempo_despues = tiempo_objetivo + timedelta(minutes=ventana_minutos)
    datos_cercanos = serie[(serie.index >= tiempo_antes) & (serie.index <= tiempo_despues)]
    if len(datos_cercanos) < 2:
        tiempo_antes = tiempo_objetivo - timedelta(minutes=30)
        tiempo_despues = tiempo_objetivo + timedelta(minutes=30)
        datos_cercanos = serie[(serie.index >= tiempo_antes) & (serie.index <= tiempo_despues)]
        if len(datos_cercanos) < 2:
            return np.nan
    datos_cercanos = datos_cercanos.sort_index()
    try:
        valor_interpolado = np.interp(
            tiempo_objetivo.timestamp(),
            [t.timestamp() for t in datos_cercanos.index],
            datos_cercanos.values
        )
        return valor_interpolado
    except:
        return np.nan

def calcular_promedio_movil_7dias(serie, fecha_central):
    """Promedio móvil de 7 días centrado en fecha_central (±3 días)"""
    fecha_inicio = fecha_central - timedelta(days=3)
    fecha_fin = fecha_central + timedelta(days=3)
    mascara = (serie.index >= fecha_inicio) & (serie.index <= fecha_fin)
    datos_7dias = serie[mascara]
    if len(datos_7dias) == 0:
        return np.nan
    return datos_7dias.mean()

def procesar_cada_10_minutos(serie_boyeruca, serie_san_antonio, serie_concepcion):
    """
    Procesa los datos cada 10 minutos en el rango común de las tres series.
    Devuelve DataFrame con columnas: time, boyeruca_bruto, san_antonio_bruto, concepcion_bruto,
    prom_movil_boyeruca, prom_movil_san_antonio, prom_movil_concepcion,
    boyeruca_relativa, san_antonio_relativa, concepcion_relativa.
    """
    # Rango común
    fecha_inicio = max(serie_boyeruca.index.min(), serie_san_antonio.index.min(), serie_concepcion.index.min())
    fecha_fin = min(serie_boyeruca.index.max(), serie_san_antonio.index.max(), serie_concepcion.index.max())

    # Redondear inicio al próximo múltiplo de 10 minutos hacia arriba
    minutos_inicio = fecha_inicio.minute
    segundos_inicio = fecha_inicio.second
    microsegundos_inicio = fecha_inicio.microsecond
    minutos_a_sumar = (10 - (minutos_inicio % 10)) % 10
    if minutos_a_sumar == 0 and (segundos_inicio > 0 or microsegundos_inicio > 0):
        minutos_a_sumar = 10
    fecha_inicio = fecha_inicio.replace(second=0, microsecond=0) + timedelta(minutes=minutos_a_sumar)

    # Redondear fin al múltiplo de 10 minutos anterior
    minutos_fin = fecha_fin.minute
    segundos_fin = fecha_fin.second
    microsegundos_fin = fecha_fin.microsecond
    fecha_fin = fecha_fin.replace(second=0, microsecond=0)
    if segundos_fin > 0 or microsegundos_fin > 0:
        fecha_fin = fecha_fin - timedelta(minutes=(minutos_fin % 10))

    print(f"Rango común cada 10 min: {fecha_inicio} a {fecha_fin}")

    tiempos_10min = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='10min')
    print(f"Total de intervalos a procesar: {len(tiempos_10min)}")

    resultados = []
    total = len(tiempos_10min)
    for i, t in enumerate(tiempos_10min):
        if i % 10000 == 0 and i > 0:
            print(f"Procesados {i}/{total} intervalos...")

        boy_bruto = interpolar_minuto_exacto(serie_boyeruca, t)
        san_bruto = interpolar_minuto_exacto(serie_san_antonio, t)
        conc_bruto = interpolar_minuto_exacto(serie_concepcion, t)

        if np.isnan(boy_bruto) or np.isnan(san_bruto) or np.isnan(conc_bruto):
            continue

        prom_boy = calcular_promedio_movil_7dias(serie_boyeruca, t)
        prom_san = calcular_promedio_movil_7dias(serie_san_antonio, t)
        prom_conc = calcular_promedio_movil_7dias(serie_concepcion, t)

        if np.isnan(prom_boy) or np.isnan(prom_san) or np.isnan(prom_conc):
            continue

        boy_rel = boy_bruto - prom_boy
        san_rel = san_bruto - prom_san
        conc_rel = conc_bruto - prom_conc

        resultados.append({
            'time': t,
            'boyeruca_bruto': boy_bruto,
            'san_antonio_bruto': san_bruto,
            'concepcion_bruto': conc_bruto,
            'prom_movil_boyeruca': prom_boy,
            'prom_movil_san_antonio': prom_san,
            'prom_movil_concepcion': prom_conc,
            'boyeruca_relativa': boy_rel,
            'san_antonio_relativa': san_rel,
            'concepcion_relativa': conc_rel
        })

    df = pd.DataFrame(resultados)
    print(f"Total de intervalos válidos: {len(df)}")
    return df

def estimar_boyeruca(df, dist_boy_san, dist_boy_conc):
    """
    Estima la altura relativa en Boyeruca mediante ponderación inversa por distancia
    usando San Antonio y Concepción.
    """
    peso_san = 1 / dist_boy_san
    peso_conc = 1 / dist_boy_conc
    peso_total = peso_san + peso_conc

    df['boyeruca_estimada_relativa'] = (
        (peso_san * df['san_antonio_relativa'] + peso_conc * df['concepcion_relativa']) / peso_total
    )
    return df

def calcular_errores(df):
    """Calcula métricas de error entre la relativa observada y estimada en Boyeruca"""
    df['error'] = df['boyeruca_relativa'] - df['boyeruca_estimada_relativa']
    df['error_abs'] = np.abs(df['error'])
    df['error_cuad'] = df['error']**2

    mae = df['error_abs'].mean()
    rmse = np.sqrt(df['error_cuad'].mean())
    mse = df['error_cuad'].mean()
    bias = df['error'].mean()
    std_error = df['error'].std()
    # R²
    ss_res = np.sum(df['error_cuad'])
    ss_tot = np.sum((df['boyeruca_relativa'] - df['boyeruca_relativa'].mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    correlacion = df['boyeruca_relativa'].corr(df['boyeruca_estimada_relativa'])

    max_error = df['error_abs'].max()
    idx_max = df['error_abs'].idxmax()
    tiempo_max = df.loc[idx_max, 'time']
    valor_max = df.loc[idx_max, 'error']

    metricas = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'Bias': bias,
        'Desviación Estándar Error': std_error,
        'R²': r2,
        'Correlación': correlacion,
        'Máximo Error Absoluto': max_error,
        'Tiempo Máximo Error': tiempo_max,
        'Valor Máximo Error': valor_max,
        'Número de Puntos': len(df)
    }
    return metricas, df

def graficar_resultados(df, metricas, archivo_grafico):
    """Genera gráfico de comparación (sin panel de error)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))  # Un solo subplot

    # Gráfico: series observada vs estimada
    ax.plot(df['time'], df['boyeruca_relativa'], label='Boyeruca observada (relativa)',
            color='blue', linewidth=1.5)
    ax.plot(df['time'], df['boyeruca_estimada_relativa'], label='Boyeruca estimada (relativa)',
            color='red', linewidth=1.5, linestyle='--')

    # Marcar punto de máximo error
    max_time = metricas['Tiempo Máximo Error']
    max_idx = (df['time'] - max_time).abs().idxmin()
    max_obs = df.loc[max_idx, 'boyeruca_relativa']
    max_est = df.loc[max_idx, 'boyeruca_estimada_relativa']
    ax.scatter(max_time, max_obs, color='red', marker='X', s=150, zorder=5,
               label=f'Máx error: {metricas["Máximo Error Absoluto"]:.4f} m')
    ax.scatter(max_time, max_est, color='red', marker='X', s=150, zorder=5)

    ax.set_ylabel('Altura relativa [m]')
    ax.set_xlabel('Tiempo')
    ax.set_title('Verificación: Boyeruca observada vs estimada (San Antonio + Concepción) usando prs(m)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Cuadro de métricas
    texto = (f"MAE: {metricas['MAE']:.4f} m\n"
             f"RMSE: {metricas['RMSE']:.4f} m\n"
             f"R²: {metricas['R²']:.4f}\n"
             f"Correlación: {metricas['Correlación']:.4f}\n"
             f"Bias: {metricas['Bias']:.4f} m\n"
             f"Puntos: {metricas['Número de Puntos']}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.75, texto, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {archivo_grafico}")
    plt.show()

def guardar_metricas_csv(metricas, archivo_metricas, fuente):
    """Guarda las métricas en un archivo CSV"""
    df_met = pd.DataFrame({
        'Métrica': list(metricas.keys()),
        'Valor': list(metricas.values())
    })
    df_met.to_csv(archivo_metricas, index=False)
    print(f"Métricas guardadas en: {archivo_metricas}")

def main():
    print("=== VERIFICACIÓN DE ESTIMACIÓN DE BOYERUCA (10 MIN) usando prs(m) ===")
    os.makedirs(CARPETA_VERIFICACION, exist_ok=True)

    # 1. Obtener coordenadas y distancias
    print("\n1. Calculando distancias...")
    estaciones = obtener_matriz_coordenadas(RUTA_COORDENADAS)
    if not estaciones:
        raise ValueError("No se pudieron cargar las coordenadas.")

    # Ajustar nombres según el archivo de coordenadas
    dist_boy_san = obtener_distancia_entre(estaciones, 'Boyeruca_CL', 'San_Antonio_CL')
    dist_boy_conc = obtener_distancia_entre(estaciones, 'Boyeruca_CL', 'Constitucion_CL')  # Verificar nombre
    print(f"Distancia Boyeruca - San Antonio: {dist_boy_san:.3f} km")
    print(f"Distancia Boyeruca - Concepción: {dist_boy_conc:.3f} km")

    # 2. Cargar series desde archivos de texto
    print("\n2. Cargando archivos de texto...")
    serie_boy = cargar_serie_desde_txt(RUTA_BOYERUCA_TXT, "Boyeruca")
    serie_san = cargar_serie_desde_txt(RUTA_SAN_ANTONIO_TXT, "San Antonio")
    serie_conc = cargar_serie_desde_txt(RUTA_CONCEPCION_TXT, "Constitucion")

    # 3. Procesar cada 10 minutos
    print("\n3. Procesando datos cada 10 minutos...")
    df_proc = procesar_cada_10_minutos(serie_boy, serie_san, serie_conc)

    if len(df_proc) == 0:
        raise ValueError("No se obtuvieron datos válidos en el rango común.")

    # 4. Estimar Boyeruca
    print("\n4. Estimando Boyeruca por ponderación inversa...")
    df_final = estimar_boyeruca(df_proc, dist_boy_san, dist_boy_conc)

    # 5. Calcular errores
    print("\n5. Calculando métricas de error...")
    metricas, df_final = calcular_errores(df_final)

    # Mostrar métricas en consola
    print("\n" + "="*50)
    for k, v in metricas.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("="*50)

    # 6. Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_grafico = os.path.join(CARPETA_VERIFICACION, f'verificacion_boyeruca_{timestamp}.png')
    archivo_metricas = os.path.join(CARPETA_VERIFICACION, f'metricas_error_{timestamp}.csv')
    archivo_datos = os.path.join(CARPETA_VERIFICACION, f'datos_verificacion_{timestamp}.csv')

    graficar_resultados(df_final, metricas, archivo_grafico)
    guardar_metricas_csv(metricas, archivo_metricas, "Verificación Boyeruca")
    df_final.to_csv(archivo_datos, index=False)
    print(f"Datos completos guardados en: {archivo_datos}")

    print("\n=== PROCESO COMPLETADO ===")
    print(f"Resultados en: {CARPETA_VERIFICACION}")

if __name__ == "__main__":
    main()