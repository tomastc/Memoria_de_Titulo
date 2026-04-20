import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN MODIFICABLE =====================
X_LIMIT = 470           # Línea de referencia para cruce del canal
ESCALA_DISTANCIAS = 0.6   # Equivalencia 1 pixel = 0,6 m (SIMONA COSTA)

# DÍA DE INICIO Y CANTIDAD DE DÍAS PARA GENERAR GRÁFICOS
DIA_SELECCIONADO = '2024-07-10'  # Fecha de inicio (formato: 'YYYY-MM-DD')
N_DAYS = 7                       # Número de días consecutivos a graficar
# =====================================================================

def calcular_distancia_minima_entre_vectores(puntos_vector1, puntos_vector2):
    """
    Calcula la distancia mínima entre dos vectores de puntos usando broadcasting
    para mayor eficiencia. Retorna la distancia mínima y los índices de los puntos.
    """
    if len(puntos_vector1) == 0 or len(puntos_vector2) == 0:
        return 0.0, (-1, -1), (np.array([0, 0]), np.array([0, 0]))
    
    # Usar broadcasting para calcular matriz de distancias
    puntos1_exp = puntos_vector1[:, np.newaxis, :]  # Forma: (n1, 1, 2)
    puntos2_exp = puntos_vector2[np.newaxis, :, :]  # Forma: (1, n2, 2)
    
    diferencias = puntos1_exp - puntos2_exp  # Forma: (n1, n2, 2)
    distancias_cuadrado = np.sum(diferencias**2, axis=2)  # Forma: (n1, n2)
    distancias = np.sqrt(distancias_cuadrado)
    
    min_idx = np.unravel_index(np.argmin(distancias), distancias.shape)
    distancia_minima = distancias[min_idx]
    
    punto1_idx = min_idx[0]
    punto2_idx = min_idx[1]
    
    punto1 = puntos_vector1[punto1_idx]
    punto2 = puntos_vector2[punto2_idx]
    
    return distancia_minima, (punto1_idx, punto2_idx), (punto1, punto2)

def extraer_puntos_limites(limite_superior, limite_inferior, coordenada_x, coordenada_y):
    """
    Extrae los puntos (coordenadas x, y) de los límites superior e inferior
    """
    indices_superior = np.where(limite_superior == 1)[0]
    indices_inferior = np.where(limite_inferior == 1)[0]
    
    if len(indices_superior) > 0:
        x_superior = coordenada_x[indices_superior]
        y_superior = coordenada_y[indices_superior]
        puntos_superior = np.column_stack((x_superior, y_superior))
    else:
        puntos_superior = np.array([])
    
    if len(indices_inferior) > 0:
        x_inferior = coordenada_x[indices_inferior]
        y_inferior = coordenada_y[indices_inferior]
        puntos_inferior = np.column_stack((x_inferior, y_inferior))
    else:
        puntos_inferior = np.array([])
    
    return puntos_superior, puntos_inferior, len(indices_superior), len(indices_inferior)

def calcular_ancho_desde_limites_reales(archivo_aperturas, archivo_salida, 
                                       escala_distancias=ESCALA_DISTANCIAS):
    """
    Calcula el ancho de desembocadura usando las 2 curvas límites reales
    con estado de apertura "abierto"
    """
    print(f"Leyendo archivo de aperturas: {archivo_aperturas}")
    
    ds_aperturas = xr.open_dataset(archivo_aperturas)
    
    variables_requeridas = ['estado_apertura', 'limite_superior_canal', 'limite_inferior_canal',
                           'coordenada_x', 'coordenada_y']
    
    for var in variables_requeridas:
        if var not in ds_aperturas:
            print(f"ERROR: Variable requerida '{var}' no encontrada en el archivo")
            return None
    
    tiempos = ds_aperturas.time.values
    estados = ds_aperturas.estado_apertura.values
    n_tiempos = len(tiempos)
    
    coordenada_x = ds_aperturas.coordenada_x.values
    coordenada_y = ds_aperturas.coordenada_y.values
    
    print(f"\nINFORMACIÓN INICIAL:")
    print(f"Total de tiempos a procesar: {n_tiempos}")
    print(f"Tiempos con estado_apertura=1 (ABIERTO): {np.sum(estados == 1)}")
    print(f"Tiempos con estado_apertura=0 (CERRADO): {np.sum(estados == 0)}")
    
    anchos = np.zeros(n_tiempos, dtype=np.float32)
    puntos_superior = np.zeros(n_tiempos, dtype=np.int32)
    puntos_inferior = np.zeros(n_tiempos, dtype=np.int32)
    
    x_punto_sup_min = np.zeros(n_tiempos, dtype=np.float32)
    y_punto_sup_min = np.zeros(n_tiempos, dtype=np.float32)
    x_punto_inf_min = np.zeros(n_tiempos, dtype=np.float32)
    y_punto_inf_min = np.zeros(n_tiempos, dtype=np.float32)
    
    contadores = {
        'total_tiempos': n_tiempos,
        'estado_abierto': 0,
        'estado_cerrado': 0,
        'abierto_con_limites': 0,
        'abierto_sin_limites': 0,
        'anchos_calculados': 0,
        'anchos_cero': 0
    }
    
    print("\nProcesando instantes...")
    print("-" * 80)
    
    for i in range(n_tiempos):
        if i % 100 == 0 and i > 0:
            porcentaje = i / n_tiempos * 100
            print(f"  Procesando instante {i}/{n_tiempos} ({porcentaje:.1f}%)")
        
        estado = estados[i]
        
        if estado == 1:
            contadores['estado_abierto'] += 1
            
            limite_sup = ds_aperturas.limite_superior_canal.values[i, :]
            limite_inf = ds_aperturas.limite_inferior_canal.values[i, :]
            
            if np.sum(limite_sup) > 0 and np.sum(limite_inf) > 0:
                contadores['abierto_con_limites'] += 1
                
                puntos_sup, puntos_inf, n_sup, n_inf = extraer_puntos_limites(
                    limite_sup, limite_inf, coordenada_x, coordenada_y
                )
                
                puntos_superior[i] = n_sup
                puntos_inferior[i] = n_inf
                
                if n_sup > 0 and n_inf > 0:
                    distancia_min, (idx_sup, idx_inf), (punto_sup, punto_inf) = \
                        calcular_distancia_minima_entre_vectores(puntos_sup, puntos_inf)
                    
                    ancho_metros = distancia_min * escala_distancias
                    anchos[i] = ancho_metros
                    
                    x_punto_sup_min[i] = punto_sup[0]
                    y_punto_sup_min[i] = punto_sup[1]
                    x_punto_inf_min[i] = punto_inf[0]
                    y_punto_inf_min[i] = punto_inf[1]
                    
                    if ancho_metros > 0:
                        contadores['anchos_calculados'] += 1
                    else:
                        contadores['anchos_cero'] += 1
                else:
                    contadores['abierto_sin_limites'] += 1
            else:
                contadores['abierto_sin_limites'] += 1
                puntos_superior[i] = 0
                puntos_inferior[i] = 0
        else:
            contadores['estado_cerrado'] += 1
            anchos[i] = 0.0
            puntos_superior[i] = 0
            puntos_inferior[i] = 0
    
    print("\n" + "="*80)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*80)
    print(f"Total de tiempos procesados: {contadores['total_tiempos']}")
    print(f"Tiempos ABIERTOS (estado=1): {contadores['estado_abierto']}")
    print(f"  - Con límites válidos: {contadores['abierto_con_limites']}")
    print(f"  - Sin límites válidos: {contadores['abierto_sin_limites']}")
    print(f"Tiempos CERRADOS (estado=0): {contadores['estado_cerrado']}")
    print(f"Anchos calculados (> 0 m): {contadores['anchos_calculados']}")
    print(f"Anchos iguales a 0 m: {contadores['anchos_cero']}")
    
    print("\nCreando archivo NetCDF con resultados...")
    
    ds_salida = ds_aperturas.copy()
    
    ds_salida['ancho_desembocadura'] = (('time',), anchos)
    ds_salida['puntos_limite_superior'] = (('time',), puntos_superior)
    ds_salida['puntos_limite_inferior'] = (('time',), puntos_inferior)
    ds_salida['x_punto_sup_min'] = (('time',), x_punto_sup_min)
    ds_salida['y_punto_sup_min'] = (('time',), y_punto_sup_min)
    ds_salida['x_punto_inf_min'] = (('time',), x_punto_inf_min)
    ds_salida['y_punto_inf_min'] = (('time',), y_punto_inf_min)
    
    ds_salida.attrs.update({
        'title': 'Ancho de desembocadura del canal basado en límites reales',
        'history': f'Creado el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'source_file': archivo_aperturas,
        'description': 'Ancho mínimo de desembocadura calculado usando las 2 curvas límites reales del canal',
        'methodology': 'Distancia euclidiana mínima entre puntos de los límites superior e inferior del canal',
        'calculation_details': 'Matriz de distancias entre todos los puntos de vector_limite_sup y vector_limite_inf',
        'minimum_distance': 'Se encuentra el mínimo de la matriz vector_sup X vector_inf',
        'coordinate_system': 'Origen en esquina superior izquierda (Y aumenta hacia abajo)',
        'escala_distancias_used': escala_distancias,
        'units': f'ancho_desembocadura en metros (escala={escala_distancias})',
        'note': 'Para tiempos sin apertura (estado_apertura=0), el ancho es 0',
        'x_limit_reference': X_LIMIT,
        'processing_summary': f"{contadores['anchos_calculados']} anchos calculados de {contadores['estado_abierto']} instantes abiertos",
        'diagnostico_total_abiertos': int(contadores['estado_abierto']),
        'diagnostico_abiertos_con_limites': int(contadores['abierto_con_limites']),
        'diagnostico_anchos_positivos': int(contadores['anchos_calculados']),
        'diagnostico_anchos_cero': int(contadores['anchos_cero'])
    })
    
    print(f"Guardando archivo: {archivo_salida}")
    ds_salida.to_netcdf(archivo_salida)
    
    ds_aperturas.close()
    
    return ds_salida

def aplicar_limpieza_outliers(ds, umbral=40):
    """
    Aplica limpieza de valores extremos en los anchos de desembocadura.
    Reglas:
      1. Si un instante abierto tiene ancho > 40 y ambos adyacentes son cerrados,
         se cambia a cerrado (estado=0, ancho=0).
      2. Para los instantes abiertos restantes, si el valor difiere en más de 'umbral'
         metros del promedio de los tres valores (t-1, t, t+1), se reemplaza por
         interpolación lineal entre los adyacentes (considerando 0 si el adyacente es cerrado).
    Se registran todos los cambios.
    """
    print("\n" + "="*80)
    print("APLICANDO LIMPIEZA DE OUTLIERS")
    print("="*80)
    
    n = len(ds.time)
    estado = ds.estado_apertura.values.copy()
    ancho = ds.ancho_desembocadura.values.copy()
    tiempos = pd.to_datetime(ds.time.values)
    
    cambios = []  # lista de diccionarios con detalles de cada cambio
    
    # --- PRIMERA PASADA: regla especial (ambos adyacentes cerrados y ancho > 50) ---
    print("  Paso 1: Detectando casos especiales (adyacentes cerrados y ancho > 50)...")
    indices_especiales = []
    for i in range(1, n-1):
        if estado[i] == 1 and ancho[i] > umbral and estado[i-1] == 0 and estado[i+1] == 0:
            indices_especiales.append(i)
    
    for i in indices_especiales:
        viejo_estado = estado[i]
        viejo_ancho = ancho[i]
        estado[i] = 0
        ancho[i] = 0.0
        cambios.append({
            'indice': i,
            'fecha': tiempos[i].strftime('%Y-%m-%d %H:%M'),
            'estado_original': viejo_estado,
            'ancho_original': viejo_ancho,
            'estado_nuevo': 0,
            'ancho_nuevo': 0.0,
            'motivo': 'Ambos adyacentes cerrados y ancho>40 -> cerrado'
        })
    
    print(f"    Se aplicaron {len(indices_especiales)} cambios por regla especial.")
    
    # --- SEGUNDA PASADA: detección de outliers por desviación del promedio ---
    print("  Paso 2: Detectando outliers por desviación del promedio...")
    cont_outliers = 0
    for i in range(1, n-1):
        if estado[i] == 0:
            continue  # solo procesamos instantes que siguen abiertos
        
        # Valores de los adyacentes (considerando cerrados como 0)
        val_prev = ancho[i-1] if estado[i-1] == 1 else 0.0
        val_next = ancho[i+1] if estado[i+1] == 1 else 0.0
        val_curr = ancho[i]
        
        promedio = (val_prev + val_curr + val_next) / 3.0
        
        if abs(val_curr - promedio) > umbral:
            # Interpolar linealmente entre adyacentes
            nuevo_ancho = (val_prev + val_next) / 2.0
            if nuevo_ancho < 0:
                nuevo_ancho = 0.0
            nuevo_estado = 1  # sigue abierto
            
            cambios.append({
                'indice': i,
                'fecha': tiempos[i].strftime('%Y-%m-%d %H:%M'),
                'estado_original': estado[i],
                'ancho_original': val_curr,
                'estado_nuevo': nuevo_estado,
                'ancho_nuevo': nuevo_ancho,
                'motivo': f'Outlier >{umbral}m del promedio, interpolado'
            })
            
            ancho[i] = nuevo_ancho
            # estado ya es 1, no cambia
            cont_outliers += 1
    
    print(f"    Se aplicaron {cont_outliers} cambios por outlier.")
    print(f"  Total de cambios realizados: {len(cambios)}")
    
    # Crear nuevo dataset con los valores modificados
    ds_mod = ds.copy()
    ds_mod['estado_apertura'].values = estado
    ds_mod['ancho_desembocadura'].values = ancho
    
    # Añadir metadatos sobre la limpieza
    ds_mod.attrs['limpieza_outliers_aplicada'] = f'Sí, umbral={umbral}m'
    ds_mod.attrs['total_cambios_limpieza'] = len(cambios)
    
    return ds_mod, cambios

def generar_reporte_estadistico(ds_anchos, tiempo_procesamiento, tiempo_total_ejecucion, 
                               archivo_aperturas, archivo_anchos, carpeta_resultados, 
                               x_limit, escala_distancias, registro_cambios=None):
    """
    Genera un archivo de texto con un resumen completo del procesamiento de anchos,
    incluyendo un registro detallado de los cambios realizados por la limpieza de outliers.
    """
    archivo_reporte = os.path.join(carpeta_resultados, "reporte_estadistico_anchos_limites.txt")
    
    tiempos = ds_anchos.time.values
    estados = ds_anchos.estado_apertura.values
    anchos = ds_anchos.ancho_desembocadura.values
    
    total_tiempos = len(tiempos)
    tiempos_abiertos = np.sum(estados == 1)
    tiempos_cerrados = np.sum(estados == 0)
    
    anchos_validos = anchos[(estados == 1) & (anchos > 0)]
    anchos_cero = anchos[(estados == 1) & (anchos == 0)]
    
    porcentaje_abiertos = 100 * tiempos_abiertos / total_tiempos if total_tiempos > 0 else 0
    porcentaje_cerrados = 100 * tiempos_cerrados / total_tiempos if total_tiempos > 0 else 0
    
    df = pd.DataFrame({
        'fecha': pd.to_datetime(tiempos),
        'estado': estados,
        'ancho': anchos
    })
    
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['año_mes'] = df['fecha'].dt.to_period('M')
    
    stats_mensuales = []
    for periodo in sorted(df['año_mes'].unique()):
        df_mes = df[df['año_mes'] == periodo]
        total_mes = len(df_mes)
        abiertos_mes = np.sum(df_mes['estado'] == 1)
        cerrados_mes = np.sum(df_mes['estado'] == 0)
        
        df_anchos_validos = df_mes[(df_mes['estado'] == 1) & (df_mes['ancho'] > 0)]
        anchos_validos_mes = len(df_anchos_validos)
        
        if anchos_validos_mes > 0:
            ancho_prom_mes = np.mean(df_anchos_validos['ancho'])
            ancho_max_mes = np.max(df_anchos_validos['ancho'])
            ancho_min_mes = np.min(df_anchos_validos['ancho'])
            ancho_std_mes = np.std(df_anchos_validos['ancho'])
        else:
            ancho_prom_mes = 0.0
            ancho_max_mes = 0.0
            ancho_min_mes = 0.0
            ancho_std_mes = 0.0
        
        stats_mensuales.append({
            'año_mes': str(periodo),
            'total': total_mes,
            'abiertos': abiertos_mes,
            '%_abiertos': 100 * abiertos_mes / total_mes if total_mes > 0 else 0,
            'cerrados': cerrados_mes,
            '%_cerrados': 100 * cerrados_mes / total_mes if total_mes > 0 else 0,
            'anchos_validos': anchos_validos_mes,
            'ancho_promedio': ancho_prom_mes,
            'ancho_maximo': ancho_max_mes,
            'ancho_minimo': ancho_min_mes,
            'ancho_std': ancho_std_mes
        })
    
    stats_anuales = []
    for año in sorted(df['año'].unique()):
        df_año = df[df['año'] == año]
        total_año = len(df_año)
        abiertos_año = np.sum(df_año['estado'] == 1)
        cerrados_año = np.sum(df_año['estado'] == 0)
        
        df_anchos_validos_año = df_año[(df_año['estado'] == 1) & (df_año['ancho'] > 0)]
        anchos_validos_año = len(df_anchos_validos_año)
        
        if anchos_validos_año > 0:
            ancho_prom_año = np.mean(df_anchos_validos_año['ancho'])
            ancho_max_año = np.max(df_anchos_validos_año['ancho'])
            ancho_min_año = np.min(df_anchos_validos_año['ancho'])
            ancho_std_año = np.std(df_anchos_validos_año['ancho'])
        else:
            ancho_prom_año = 0.0
            ancho_max_año = 0.0
            ancho_min_año = 0.0
            ancho_std_año = 0.0
        
        stats_anuales.append({
            'año': año,
            'total': total_año,
            'abiertos': abiertos_año,
            '%_abiertos': 100 * abiertos_año / total_año if total_año > 0 else 0,
            'cerrados': cerrados_año,
            '%_cerrados': 100 * cerrados_año / total_año if total_año > 0 else 0,
            'anchos_validos': anchos_validos_año,
            'ancho_promedio': ancho_prom_año,
            'ancho_maximo': ancho_max_año,
            'ancho_minimo': ancho_min_año,
            'ancho_std': ancho_std_año
        })
    
    reporte = []
    reporte.append("="*80)
    reporte.append("REPORTE ESTADÍSTICO COMPLETO DE ANCHOS DE DESEMBOCADURA (BASADO EN LÍMITES)")
    reporte.append("="*80)
    reporte.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"Parámetros utilizados: X_LIMIT={x_limit} (referencia), ESCALA_DISTANCIAS={escala_distancias}")
    if registro_cambios is not None:
        reporte.append(f"Limpieza de outliers aplicada: Sí (umbral=50 m, total cambios={len(registro_cambios)})")
    else:
        reporte.append("Limpieza de outliers aplicada: No")
    reporte.append("")
    
    reporte.append("1. RESUMEN GENERAL")
    reporte.append("-"*80)
    reporte.append(f"  • Total de instantes analizados: {total_tiempos}")
    reporte.append(f"  • Instantes ABIERTOS (estado=1): {tiempos_abiertos} ({porcentaje_abiertos:.1f}%)")
    reporte.append(f"  • Instantes CERRADOS (estado=0): {tiempos_cerrados} ({porcentaje_cerrados:.1f}%)")
    reporte.append(f"  • Anchos válidos calculados (estado=1 y ancho>0): {len(anchos_validos)}")
    reporte.append(f"  • Anchos iguales a 0 (estado=1): {len(anchos_cero)}")
    reporte.append(f"  • Método: Distancia mínima entre las 2 curvas límites reales del canal")
    reporte.append(f"  • Verificación: Solo se procesan instantes con estado_apertura=1 y 2 límites reales")
    reporte.append("")
    
    reporte.append("2. ESTADÍSTICAS DE ANCHOS (solo válidos: estado=1 y ancho>0)")
    reporte.append("-"*80)
    if len(anchos_validos) > 0:
        reporte.append(f"  • Ancho promedio: {np.mean(anchos_validos):.2f} m")
        reporte.append(f"  • Desviación estándar: {np.std(anchos_validos):.2f} m")
        reporte.append(f"  • Ancho mínimo: {np.min(anchos_validos):.2f} m")
        reporte.append(f"  • Ancho máximo: {np.max(anchos_validos):.2f} m")
        reporte.append(f"  • Mediana: {np.median(anchos_validos):.2f} m")
        reporte.append(f"  • Percentil 25: {np.percentile(anchos_validos, 25):.2f} m")
        reporte.append(f"  • Percentil 75: {np.percentile(anchos_validos, 75):.2f} m")
        if np.mean(anchos_validos) > 0:
            cv = (np.std(anchos_validos) / np.mean(anchos_validos)) * 100
            reporte.append(f"  • Coeficiente de variación: {cv:.1f}%")
        
        reporte.append(f"\n  • DISTRIBUCIÓN DE ANCHOS POR RANGOS (metros):")
        bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
        bin_labels = ["0-1", "1-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200-500", "500-1000"]
        
        for i in range(len(bins)-1):
            count = np.sum((anchos_validos >= bins[i]) & (anchos_validos < bins[i+1]))
            if count > 0:
                porcentaje = 100 * count / len(anchos_validos)
                reporte.append(f"      {bin_labels[i]} m: {count} ({porcentaje:.1f}%)")
    else:
        reporte.append("  • No hay anchos válidos calculados")
    reporte.append("")
    
    reporte.append("3. ANÁLISIS MENSUAL")
    reporte.append("-"*80)
    reporte.append("Mes       | Total | Abiertos | % Abiertos | Cerrados | % Cerrados | Anchos >0 | Ancho Prom (m) | Ancho Max (m)")
    reporte.append("-"*120)
    
    for stats in stats_mensuales:
        reporte.append(f"{stats['año_mes']:10} | {stats['total']:5d} | {stats['abiertos']:8d} | {stats['%_abiertos']:10.1f}% | {stats['cerrados']:7d} | {stats['%_cerrados']:10.1f}% | {stats['anchos_validos']:9d} | {stats['ancho_promedio']:13.2f} | {stats['ancho_maximo']:12.2f}")
    
    reporte.append("")
    
    reporte.append("4. ANÁLISIS ANUAL")
    reporte.append("-"*80)
    reporte.append("Año  | Total | Abiertos | % Abiertos | Cerrados | % Cerrados | Anchos >0 | Ancho Prom (m) | Ancho Max (m)")
    reporte.append("-"*115)
    
    for stats in stats_anuales:
        reporte.append(f"{stats['año']:4d} | {stats['total']:5d} | {stats['abiertos']:8d} | {stats['%_abiertos']:10.1f}% | {stats['cerrados']:7d} | {stats['%_cerrados']:10.1f}% | {stats['anchos_validos']:9d} | {stats['ancho_promedio']:13.2f} | {stats['ancho_maximo']:12.2f}")
    
    reporte.append("")
    
    reporte.append("5. ESTADÍSTICAS TEMPORALES")
    reporte.append("-"*80)
    
    if len(tiempos) > 0:
        primera_fecha = pd.to_datetime(tiempos[0]).strftime('%Y-%m-%d %H:%M')
        ultima_fecha = pd.to_datetime(tiempos[-1]).strftime('%Y-%m-%d %H:%M')
        reporte.append(f"  • Primera fecha analizada: {primera_fecha}")
        reporte.append(f"  • Última fecha analizada: {ultima_fecha}")
        
        fecha_inicio = pd.to_datetime(tiempos[0])
        fecha_fin = pd.to_datetime(tiempos[-1])
        duracion_total = fecha_fin - fecha_inicio
        reporte.append(f"  • Período total analizado: {duracion_total.days} días")
    
    cambios_estado = np.diff(estados)
    aperturas = np.sum(cambios_estado == 1)
    cierres = np.sum(cambios_estado == -1)
    
    reporte.append(f"  • Número de aperturas detectadas: {aperturas}")
    reporte.append(f"  • Número de cierres detectados: {cierres}")
    reporte.append("")
    
    reporte.append("6. DIAGNÓSTICO DEL PROCESAMIENTO")
    reporte.append("-"*80)
    if 'diagnostico_total_abiertos' in ds_anchos.attrs:
        reporte.append(f"  • Total instantes con estado_apertura=1: {ds_anchos.attrs['diagnostico_total_abiertos']}")
        reporte.append(f"  • Instantes con límites válidos: {ds_anchos.attrs['diagnostico_abiertos_con_limites']}")
        reporte.append(f"  • Anchos positivos calculados (> 0): {ds_anchos.attrs['diagnostico_anchos_positivos']}")
        reporte.append(f"  • Anchos iguales a 0: {ds_anchos.attrs['diagnostico_anchos_cero']}")
        if ds_anchos.attrs['diagnostico_abiertos_con_limites'] > 0:
            porcentaje_exito = 100 * ds_anchos.attrs['diagnostico_anchos_positivos'] / ds_anchos.attrs['diagnostico_abiertos_con_limites']
            reporte.append(f"  • Porcentaje de éxito (anchos>0 / abiertos con límites): {porcentaje_exito:.1f}%")
    reporte.append("")
    
    reporte.append("7. TIEMPO Y EFICIENCIA")
    reporte.append("-"*80)
    reporte.append(f"  • Tiempo de procesamiento (cálculo de anchos): {tiempo_procesamiento:.2f}s ({tiempo_procesamiento/60:.1f} min)")
    reporte.append(f"  • Tiempo total de ejecución del código: {tiempo_total_ejecucion:.2f}s ({tiempo_total_ejecucion/60:.1f} min)")
    velocidad_promedio = tiempo_procesamiento/total_tiempos if total_tiempos > 0 else 0
    reporte.append(f"  • Velocidad promedio: {velocidad_promedio:.4f}s/instante")
    reporte.append(f"  • Archivo de entrada: {archivo_aperturas}")
    reporte.append(f"  • Archivo de salida: {archivo_anchos}")
    reporte.append(f"  • Carpeta de resultados: {carpeta_resultados}")
    reporte.append("")
    
    reporte.append("8. MÉTODO DE CÁLCULO")
    reporte.append("-"*80)
    reporte.append("  • Cálculo basado en: 2 curvas límites reales (superior e inferior)")
    reporte.append("  • Matriz de distancias: Todos los puntos del límite superior vs todos los puntos del límite inferior")
    reporte.append("  • Métrica: Distancia euclidiana mínima de la matriz completa")
    reporte.append("  • Filtro: Solo instantes con estado_apertura=1")
    reporte.append("  • Escala: 1 píxel = 0.6 metros")
    reporte.append("  • Línea de referencia: X_LIMIT = 470 (solo para visualización)")
    reporte.append("")
    
    reporte.append("9. DETALLE COMPLETO DE TODOS LOS INSTANTES PROCESADOS")
    reporte.append("-"*80)
    reporte.append("Nº  | Fecha/Hora              | Estado   | Ancho (m)  | Puntos Superior | Puntos Inferior")
    reporte.append("-"*95)
    
    for i in range(len(tiempos)):
        fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M')
        ancho_val = anchos[i]
        estado_str = "ABIERTO" if estados[i] == 1 else "CERRADO"
        
        if 'puntos_limite_superior' in ds_anchos and 'puntos_limite_inferior' in ds_anchos:
            puntos_sup = ds_anchos.puntos_limite_superior.values[i]
            puntos_inf = ds_anchos.puntos_limite_inferior.values[i]
        else:
            puntos_sup = 0
            puntos_inf = 0
            
        reporte.append(f"{i+1:4d} | {fecha_str}         | {estado_str:8} | {ancho_val:9.2f} | {puntos_sup:14} | {puntos_inf:14}")
    
    reporte.append("")
    reporte.append(f"Total de instantes listados: {len(tiempos)}")
    reporte.append("")
    
    # ===== NUEVA SECCIÓN: REGISTRO DE CAMBIOS POR LIMPIEZA =====
    if registro_cambios is not None and len(registro_cambios) > 0:
        reporte.append("10. REGISTRO DE CAMBIOS REALIZADOS POR LIMPIEZA DE OUTLIERS")
        reporte.append("-"*80)
        reporte.append("Nº  | Fecha/Hora              | Estado Orig | Ancho Orig (m) | Estado Nuevo | Ancho Nuevo (m) | Motivo")
        reporte.append("-"*110)
        
        for idx, cambio in enumerate(registro_cambios):
            reporte.append(f"{idx+1:4d} | {cambio['fecha']}         | {cambio['estado_original']:10d} | {cambio['ancho_original']:14.2f} | {cambio['estado_nuevo']:11d} | {cambio['ancho_nuevo']:14.2f} | {cambio['motivo']}")
        
        reporte.append("")
        reporte.append(f"Total de instantes modificados: {len(registro_cambios)}")
        reporte.append("")
    else:
        reporte.append("10. REGISTRO DE CAMBIOS POR LIMPIEZA")
        reporte.append("-"*80)
        reporte.append("  No se realizaron modificaciones (ningún outlier detectado).")
        reporte.append("")
    
    # Escribir archivo
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("\n".join(reporte))
    
    print(f"Reporte estadístico generado en: {archivo_reporte}")
    
    # Archivo CSV con todos los datos
    archivo_csv = os.path.join(carpeta_resultados, "datos_completos_anchos.csv")
    df_completo = pd.DataFrame({
        'fecha': pd.to_datetime(tiempos),
        'estado_apertura': estados,
        'ancho_desembocadura_m': anchos,
        'puntos_limite_superior': ds_anchos.puntos_limite_superior.values if 'puntos_limite_superior' in ds_anchos else [0]*len(tiempos),
        'puntos_limite_inferior': ds_anchos.puntos_limite_inferior.values if 'puntos_limite_inferior' in ds_anchos else [0]*len(tiempos),
        'x_punto_sup_min': ds_anchos.x_punto_sup_min.values if 'x_punto_sup_min' in ds_anchos else [0]*len(tiempos),
        'y_punto_sup_min': ds_anchos.y_punto_sup_min.values if 'y_punto_sup_min' in ds_anchos else [0]*len(tiempos),
        'x_punto_inf_min': ds_anchos.x_punto_inf_min.values if 'x_punto_inf_min' in ds_anchos else [0]*len(tiempos),
        'y_punto_inf_min': ds_anchos.y_punto_inf_min.values if 'y_punto_inf_min' in ds_anchos else [0]*len(tiempos)
    })
    
    df_completo.to_csv(archivo_csv, index=False, encoding='utf-8')
    print(f"Datos completos exportados a CSV: {archivo_csv}")
    
    return archivo_reporte

def generar_grafico_unico_dia(ds_anchos, dia_seleccionado, archivo_imagen, carpeta_resultados):
    """
    Genera UNA SOLA ventana de visualización con todos los instantes de un día específico.
    Se incluye el perímetro completo agua-arena en gris plateado para contexto.
    """
    print(f"\nGenerando gráfico único para el día: {dia_seleccionado}")
    
    try:
        dia_dt = pd.to_datetime(dia_seleccionado)
    except:
        print(f"ERROR: Formato de fecha incorrecto. Usa 'YYYY-MM-DD'")
        return False
    
    tiempos = ds_anchos.time.values
    tiempos_dt = pd.to_datetime(tiempos)
    
    mascara_dia = (tiempos_dt.date == dia_dt.date())
    
    if not np.any(mascara_dia):
        print(f"ERROR: No hay datos para el día {dia_seleccionado}")
        return False
    
    indices_dia = np.where(mascara_dia)[0]
    
    estados = ds_anchos.estado_apertura.values[mascara_dia]
    instantes_abiertos_dia = np.sum(estados == 1)
    instantes_cerrados_dia = np.sum(estados == 0)
    
    print(f"  Instantes encontrados en {dia_seleccionado}: {len(indices_dia)}")
    print(f"  - ABIERTOS: {instantes_abiertos_dia}")
    print(f"  - CERRADOS: {instantes_cerrados_dia}")
    
    if instantes_abiertos_dia == 0:
        print(f"  ADVERTENCIA: No hay instantes ABIERTOS en {dia_seleccionado}")
        return False
    
    n_instantes = len(indices_dia)
    n_cols = 4
    n_rows = int(np.ceil(n_instantes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    coordenada_x = ds_anchos.coordenada_x.values
    coordenada_y = ds_anchos.coordenada_y.values
    
    tiene_perimetro = 'perimetro_arena_agua' in ds_anchos
    
    for idx, (ax, instante_idx) in enumerate(zip(axes.flat, indices_dia)):
        if idx >= n_instantes:
            ax.axis('off')
            continue
        
        tiempo = tiempos[instante_idx]
        tiempo_str = pd.to_datetime(tiempo).strftime('%H:%M')
        estado = ds_anchos.estado_apertura.values[instante_idx]
        ancho = ds_anchos.ancho_desembocadura.values[instante_idx]
        
        if tiene_perimetro:
            perimetro_mask = ds_anchos.perimetro_arena_agua.values[instante_idx, :] == 1
            if np.any(perimetro_mask):
                x_perim = coordenada_x[perimetro_mask]
                y_perim = coordenada_y[perimetro_mask]
                ax.scatter(x_perim, y_perim, c='gray', s=3, alpha=0.6, 
                           label='Perímetro', zorder=1)
        
        limite_sup = ds_anchos.limite_superior_canal.values[instante_idx, :]
        limite_inf = ds_anchos.limite_inferior_canal.values[instante_idx, :]
        
        puntos_sup, puntos_inf, n_sup, n_inf = extraer_puntos_limites(
            limite_sup, limite_inf, coordenada_x, coordenada_y
        )
        
        if len(puntos_sup) > 0:
            ax.scatter(puntos_sup[:, 0], puntos_sup[:, 1], c='blue', s=5, 
                      alpha=0.9, label=f'Sup ({n_sup})', zorder=2)
        
        if len(puntos_inf) > 0:
            ax.scatter(puntos_inf[:, 0], puntos_inf[:, 1], c='red', s=5, 
                      alpha=0.9, label=f'Inf ({n_inf})', zorder=2)
        
        if estado == 1 and ancho > 0:
            if len(puntos_sup) > 0 and len(puntos_inf) > 0:
                distancia_min, (idx_sup, idx_inf), (punto_sup, punto_inf) = \
                    calcular_distancia_minima_entre_vectores(puntos_sup, puntos_inf)
                
                ax.plot([punto_sup[0], punto_inf[0]], 
                       [punto_sup[1], punto_inf[1]], 
                       'y-', linewidth=2, label=f'{ancho:.1f}m', zorder=3)
                
                ax.scatter(punto_sup[0], punto_sup[1], c='yellow', s=30, marker='*',
                          edgecolors='black', linewidths=1, zorder=4)
                ax.scatter(punto_inf[0], punto_inf[1], c='yellow', s=30, marker='*',
                          edgecolors='black', linewidths=1, zorder=4)
            
            estado_texto = "ABIERTO"
            color_fondo = 'lightgreen'
        else:
            estado_texto = "CERRADO"
            color_fondo = 'lightcoral'
        
        ax.set_title(f'{tiempo_str}\n{estado_texto}: {ancho:.1f}m', fontsize=10, fontweight='bold')
        ax.set_facecolor(color_fondo)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
        
        ax.set_aspect('equal', adjustable='datalim')
        
        todos_puntos = []
        if tiene_perimetro and np.any(perimetro_mask):
            todos_puntos.append(np.column_stack((x_perim, y_perim)))
        if len(puntos_sup) > 0:
            todos_puntos.append(puntos_sup)
        if len(puntos_inf) > 0:
            todos_puntos.append(puntos_inf)
        
        if todos_puntos:
            todos_puntos = np.vstack(todos_puntos)
            x_min, x_max = np.min(todos_puntos[:, 0]), np.max(todos_puntos[:, 0])
            y_min, y_max = np.min(todos_puntos[:, 1]), np.max(todos_puntos[:, 1])
            
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax.invert_yaxis()
        
        if idx >= n_instantes - n_cols:
            ax.set_xlabel('X', fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel('Y', fontsize=8)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
                     fontsize=6, markerscale=0.5)
    
    plt.suptitle(f'ANCHO DE DESEMBOCADURA - {dia_seleccionado}\n'
                f'Total instantes: {n_instantes} (Abiertos: {instantes_abiertos_dia}, Cerrados: {instantes_cerrados_dia})', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print(f"Guardando gráfico único: {archivo_imagen}")
    plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return True

def generar_grafico_temporal_todos_anchos(ds_anchos, archivo_imagen, carpeta_resultados):
    """
    Genera gráfico temporal de TODOS los anchos calculados
    """
    print(f"\nGenerando gráfico temporal de TODOS los anchos...")
    
    tiempos = ds_anchos.time.values
    anchos = ds_anchos.ancho_desembocadura.values
    estados = ds_anchos.estado_apertura.values
    
    tiempos_dt = pd.to_datetime(tiempos)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    idx_abiertos_positivos = np.where((estados == 1) & (anchos > 0))[0]
    idx_abiertos_cero = np.where((estados == 1) & (anchos == 0))[0]
    idx_cerrados = np.where(estados == 0)[0]
    
    if len(idx_abiertos_positivos) > 0:
        ax.scatter(tiempos_dt[idx_abiertos_positivos], anchos[idx_abiertos_positivos], 
                  c='green', s=30, alpha=0.7, marker='o', label='Abierto con ancho > 0')
    
    if len(idx_abiertos_cero) > 0:
        ax.scatter(tiempos_dt[idx_abiertos_cero], anchos[idx_abiertos_cero], 
                  c='orange', s=30, alpha=0.7, marker='o', label='Abierto con ancho = 0')
    
    if len(idx_cerrados) > 0:
        ax.scatter(tiempos_dt[idx_cerrados], anchos[idx_cerrados], 
                  c='red', s=30, alpha=0.3, marker='o', label='Cerrado')
    
    ax.set_xlabel('Tiempo', fontsize=12)
    ax.set_ylabel('Ancho de desembocadura (m)', fontsize=12)
    ax.set_title('EVOLUCIÓN TEMPORAL DEL ANCHO DE DESEMBOCADURA\n', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=45, ha='center')
    
    total = len(estados)
    abierto_positivo = len(idx_abiertos_positivos)
    abierto_cero = len(idx_abiertos_cero)
    cerrado = len(idx_cerrados)
    
    pct_abierto_positivo = 100 * abierto_positivo / total if total > 0 else 0
    pct_abierto_cero = 100 * abierto_cero / total if total > 0 else 0
    pct_cerrado = 100 * cerrado / total if total > 0 else 0
    
    stats_text = (f'Total datos: {total}\n'
                  f'Abierto con ancho>0: {abierto_positivo} ({pct_abierto_positivo:.1f}%)\n'
                  f'Abierto con ancho=0: {abierto_cero} ({pct_abierto_cero:.1f}%)\n'
                  f'Cerrados: {cerrado} ({pct_cerrado:.1f}%)')
    
    anchos_validos = anchos[(estados == 1) & (anchos > 0)]
    if len(anchos_validos) > 0:
        stats_text += (f'\n\nEstadísticas de anchos (>0):\n'
                       f'Promedio: {np.mean(anchos_validos):.2f} m\n'
                       f'Desv. estándar: {np.std(anchos_validos):.2f} m\n'
                       f'Máximo: {np.max(anchos_validos):.2f} m\n'
                       f'Mínimo: {np.min(anchos_validos):.2f} m')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    print(f"Guardando gráfico temporal: {archivo_imagen}")
    plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return True

def main():
    """Función principal"""
    inicio_tiempo_total = datetime.now()
    
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    
    import glob
    archivo_aperturas_pattern = os.path.join(carpeta_base, 'Resultados_Aperturas_ROI1-FINAL', 'Data_Aperturas_COMPLETO_*.nc')
    archivos_encontrados = glob.glob(archivo_aperturas_pattern)
    
    if not archivos_encontrados:
        archivo_aperturas = os.path.join(carpeta_base, 'Resultados_Aperturas_ROI1-FINAL', 'Data_Aperturas_COMPLETO.nc')
        if not os.path.exists(archivo_aperturas):
            print(f"ERROR: No se encontró el archivo de aperturas en: {archivo_aperturas_pattern}")
            return
    else:
        archivo_aperturas = max(archivos_encontrados, key=os.path.getctime)
    
    carpeta_resultados = os.path.join(carpeta_base, f'Resultados_Ancho_Canal_{DIA_SELECCIONADO.replace("-", "")}_ROI1-FINAL')
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    fecha_actual = datetime.now().strftime('%Y%m%d_%H%M%S')
    archivo_anchos = os.path.join(carpeta_resultados, f'Data_Ancho_Desde_Limites_{DIA_SELECCIONADO.replace("-", "")}.nc')
    
    print("="*80)
    print("CÁLCULO COMPLETO DE ANCHO DE DESEMBOCADURA BASADO EN LÍMITES")
    print("="*80)
    print(f"Archivo de entrada: {archivo_aperturas}")
    print(f"Día de inicio para visualización: {DIA_SELECCIONADO}, N días: {N_DAYS}")
    
    inicio_procesamiento = datetime.now()
    ds_anchos = calcular_ancho_desde_limites_reales(
        archivo_aperturas, archivo_anchos, escala_distancias=ESCALA_DISTANCIAS
    )
    
    if ds_anchos is None:
        print("Error en el cálculo de anchos")
        return
    
    fin_procesamiento = datetime.now()
    tiempo_procesamiento = (fin_procesamiento - inicio_procesamiento).total_seconds()
    
    # --- APLICAR LIMPIEZA DE OUTLIERS ---
    ds_limpio, registro_cambios = aplicar_limpieza_outliers(ds_anchos, umbral=40)
    
    # Guardar el dataset limpio en un nuevo archivo (añadir "_limpio" al nombre)
    archivo_anchos_limpio = archivo_anchos.replace('.nc', '_limpio.nc')
    print(f"\nGuardando dataset con limpieza en: {archivo_anchos_limpio}")
    ds_limpio.to_netcdf(archivo_anchos_limpio)
    
    # Generar gráficos con los datos limpios
    print("\n" + "="*80)
    print(f"GENERANDO GRÁFICOS PARA CADA DÍA DESDE {DIA_SELECCIONADO} HASTA +{N_DAYS-1} DÍAS")
    print("="*80)
    
    carpeta_imagenes_diarias = os.path.join(carpeta_resultados, "Resultados anchos medidos x día")
    os.makedirs(carpeta_imagenes_diarias, exist_ok=True)
    
    fecha_inicio = datetime.strptime(DIA_SELECCIONADO, '%Y-%m-%d')
    
    for i in range(N_DAYS):
        fecha_actual_dia = fecha_inicio + timedelta(days=i)
        fecha_str = fecha_actual_dia.strftime('%Y-%m-%d')
        
        archivo_grafico_dia = os.path.join(carpeta_imagenes_diarias, 
                                          f'ancho_desembocadura_{fecha_actual_dia.strftime("%Y%m%d")}.png')
        
        generar_grafico_unico_dia(ds_limpio, fecha_str, archivo_grafico_dia, carpeta_imagenes_diarias)
    
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO TEMPORAL DE TODOS LOS ANCHOS")
    print("="*80)
    
    archivo_grafico_temporal = os.path.join(carpeta_resultados, 
                                          f'evolucion_temporal_todos_anchos_{fecha_actual}.png')
    
    generar_grafico_temporal_todos_anchos(ds_limpio, archivo_grafico_temporal, carpeta_resultados)
    
    fin_tiempo_total = datetime.now()
    tiempo_total_ejecucion = (fin_tiempo_total - inicio_tiempo_total).total_seconds()
    
    print("\n" + "="*80)
    print("GENERANDO REPORTE ESTADÍSTICO")
    print("="*80)
    
    archivo_reporte = generar_reporte_estadistico(
        ds_anchos=ds_limpio,
        tiempo_procesamiento=tiempo_procesamiento,
        tiempo_total_ejecucion=tiempo_total_ejecucion,
        archivo_aperturas=archivo_aperturas,
        archivo_anchos=archivo_anchos_limpio,  # reporta el archivo limpio
        carpeta_resultados=carpeta_resultados,
        x_limit=X_LIMIT,
        escala_distancias=ESCALA_DISTANCIAS,
        registro_cambios=registro_cambios
    )
    
    print("\n" + "="*80)
    print("ESTADÍSTICAS FINALES - COMPLETAS")
    print("="*80)
    
    tiempos = ds_limpio.time.values
    estados = ds_limpio.estado_apertura.values
    anchos = ds_limpio.ancho_desembocadura.values
    
    total_tiempos = len(tiempos)
    tiempos_abiertos = np.sum(estados == 1)
    tiempos_cerrados = np.sum(estados == 0)
    
    anchos_validos = anchos[(estados == 1) & (anchos > 0)]
    anchos_cero = anchos[(estados == 1) & (anchos == 0)]
    
    print(f"RESUMEN EJECUCIÓN:")
    print(f"  • Total de instantes: {total_tiempos}")
    print(f"  • Instantes ABIERTOS: {tiempos_abiertos} ({tiempos_abiertos/total_tiempos*100:.1f}%)")
    print(f"  • Instantes CERRADOS: {tiempos_cerrados} ({tiempos_cerrados/total_tiempos*100:.1f}%)")
    
    if len(anchos_validos) > 0:
        print(f"\nESTADÍSTICAS DE ANCHOS (solo válidos > 0):")
        print(f"  • Cantidad de anchos calculados: {len(anchos_validos)}")
        print(f"  • Ancho promedio: {np.mean(anchos_validos):.2f} m")
        print(f"  • Ancho mínimo: {np.min(anchos_validos):.2f} m")
        print(f"  • Ancho máximo: {np.max(anchos_validos):.2f} m")
        print(f"  • Desviación estándar: {np.std(anchos_validos):.2f} m")
        print(f"  • Mediana: {np.median(anchos_validos):.2f} m")
    else:
        print(f"\nNo se calcularon anchos válidos (> 0)")
    
    if len(anchos_cero) > 0:
        print(f"  • Instantes abiertos con ancho = 0: {len(anchos_cero)}")
    
    print(f"\nMÉTODO UTILIZADO:")
    print(f"  • Cálculo basado en: 2 curvas límites reales (superior e inferior)")
    
    if 'puntos_limite_superior' in ds_limpio and 'puntos_limite_inferior' in ds_limpio:
        mascara_abiertos = (estados == 1)
        if np.any(mascara_abiertos):
            puntos_sup_array = ds_limpio.puntos_limite_superior.values[mascara_abiertos]
            puntos_inf_array = ds_limpio.puntos_limite_inferior.values[mascara_abiertos]
            
            avg_puntos_sup = np.mean(puntos_sup_array)
            avg_puntos_inf = np.mean(puntos_inf_array)
            min_puntos_sup = np.min(puntos_sup_array)
            min_puntos_inf = np.min(puntos_inf_array)
            max_puntos_sup = np.max(puntos_sup_array)
            max_puntos_inf = np.max(puntos_inf_array)
            
            print(f"  • Estadísticas de puntos por límite (solo instantes ABIERTOS):")
            print(f"    - Límite superior: Promedio={avg_puntos_sup:.1f}, Min={min_puntos_sup}, Max={max_puntos_sup}")
            print(f"    - Límite inferior: Promedio={avg_puntos_inf:.1f}, Min={min_puntos_inf}, Max={max_puntos_inf}")
            print(f"    - Matriz de distancias típica: {avg_puntos_sup:.0f} x {avg_puntos_inf:.0f} ≈ {avg_puntos_sup*avg_puntos_inf:.0f} cálculos")
    
    print(f"  • Métrica: Distancia euclidiana mínima de la matriz completa")
    print(f"  • Escala: 1 píxel = {ESCALA_DISTANCIAS} metro")
    
    print(f"\nTIEMPOS DE EJECUCIÓN:")
    print(f"  • Tiempo de procesamiento: {tiempo_procesamiento:.2f} segundos")
    print(f"  • Tiempo total: {tiempo_total_ejecucion:.2f} segundos ({tiempo_total_ejecucion/60:.1f} minutos)")
    print(f"  • Velocidad: {total_tiempos/tiempo_procesamiento:.1f} instantes/segundo")
    
    print(f"\nARCHIVOS GENERADOS:")
    print(f"  • NetCDF con anchos (original): {archivo_anchos}")
    print(f"  • NetCDF con anchos (limpio): {archivo_anchos_limpio}")
    print(f"  • Reporte estadístico: {archivo_reporte}")
    print(f"  • Gráficos diarios: {carpeta_imagenes_diarias} (se generaron {N_DAYS} imágenes)")
    print(f"  • Gráfico temporal de todos los anchos: {archivo_grafico_temporal}")
    print(f"  • Carpeta de resultados principal: {carpeta_resultados}")
    
    print("\n" + "="*80)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    ds_limpio.close()
    ds_anchos.close()

if __name__ == "__main__":
    main()