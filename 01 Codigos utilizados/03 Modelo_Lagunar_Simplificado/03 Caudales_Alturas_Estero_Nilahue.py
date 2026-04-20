import pandas as pd
import numpy as np
from datetime import datetime
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import re

def procesar_datos_caudal(archivo_excel, hoja=0):
    """
    Procesa archivo Excel con datos de caudal en formato no convencional.
    Los datos comienzan en la fila 13 con tres grupos de columnas por fila.
    Estructura: 
    - Grupo 1: Día(B), Hora(C), Altura(D-E combinada), Caudal(F)
    - Grupo 2: Día(I), Hora(J), Altura(K), Caudal(L)
    - Grupo 3: Día(Q), Hora(R), Altura(S), Caudal(T)
    
    Args:
        archivo_excel (str): Ruta al archivo Excel
        hoja (int/str): Hoja a procesar
    
    Returns:
        pandas.DataFrame: Serie temporal con datos de altura y caudal ordenados
    """
    
    # Verificar que el archivo existe
    if not os.path.exists(archivo_excel):
        raise FileNotFoundError(f"No se encuentra el archivo: {archivo_excel}")
    
    # Leer archivo Excel
    try:
        df_raw = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)
    except ImportError:
        raise ImportError("Para leer archivos Excel, instale: pip install openpyxl")
    
    print(f"Archivo cargado correctamente: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
    
    # Detectar mes y año de los datos - ahora pasamos el nombre del archivo como respaldo
    mes_anio = _detectar_periodo(df_raw, archivo_excel)
    print(f"Periodo de datos: {mes_anio.strftime('%m/%Y')}")
    
    # Procesar datos desde fila 13 (índice 12)
    datos_procesados = []
    
    for fila_idx in range(12, len(df_raw)):
        fila = df_raw.iloc[fila_idx]
        
        # Procesar los tres grupos de datos en cada fila
        # Grupo 1: Día(B-1), Hora(C-2), Altura(D-3), Caudal(F-5)
        _procesar_grupo_datos(fila, 1, 2, 3, 5, datos_procesados, mes_anio)
        
        # Grupo 2: Día(I-8), Hora(J-9), Altura(K-10), Caudal(L-11)
        _procesar_grupo_datos(fila, 8, 9, 10, 11, datos_procesados, mes_anio)
        
        # Grupo 3: Día(Q-16), Hora(R-17), Altura(S-18), Caudal(T-19)
        _procesar_grupo_datos(fila, 16, 17, 18, 19, datos_procesados, mes_anio)
    
    if not datos_procesados:
        raise ValueError("No se encontraron datos válidos para procesar")
    
    # Crear DataFrame final
    df_final = pd.DataFrame(datos_procesados)
    df_final = df_final.sort_values('fecha_hora').reset_index(drop=True)
    df_final.set_index('fecha_hora', inplace=True)
    
    # Seleccionar solo las columnas requeridas: altura y caudal
    df_final = df_final[['altura_m', 'caudal_m3s']]
    
    return df_final

def _detectar_periodo(df_raw, archivo_excel=None):
    """Detecta el mes y año de los datos a partir de la celda D-E11 (fila 10, columna 3).
       Si falla, intenta extraer el mes del nombre del archivo."""
    try:
        # La celda D-E11 es la fila 10, columna 3 (índices base 0)
        celda_periodo = df_raw.iloc[10, 3]
        
        if pd.isna(celda_periodo):
            # Si la celda D11 está vacía, podría estar en E11
            celda_periodo = df_raw.iloc[10, 4]
        
        if not pd.isna(celda_periodo):
            # Convertir a string y procesar
            texto_periodo = str(celda_periodo).strip()
            print(f"Texto de periodo detectado: {texto_periodo}")
            
            # Buscar patrones de fecha en el texto
            if '/' in texto_periodo:
                try:
                    # Formato: "0107/2024 - 3107/2024" o similar
                    partes = texto_periodo.split('/')
                    if len(partes) >= 2:
                        # El primer segmento contiene el día y mes
                        dia_mes = partes[0].strip()
                        anio = partes[1].strip().split()[0]  # Tomar solo el año, ignorar lo que sigue
                        
                        # El mes son los últimos 2 dígitos del segmento día-mes
                        mes = dia_mes[-2:]
                        
                        # Validar y convertir
                        mes_int = int(mes)
                        anio_int = int(anio)
                        
                        if 1 <= mes_int <= 12 and anio_int > 2000:
                            return datetime(anio_int, mes_int, 1)
                except (ValueError, IndexError):
                    pass
            
            # Si el formato anterior falla, buscar "MES:" en el texto
            if 'MES:' in texto_periodo.upper():
                try:
                    texto_mes = texto_periodo.split('MES:')[-1].strip()
                    mes = int(texto_mes.split('/')[0])
                    anio = int(texto_mes.split('/')[1])
                    return datetime(anio, mes, 1)
                except (ValueError, IndexError):
                    pass
                    
            # Si el formato anterior falla, buscar "PERIODO:" en el texto
            if 'PERIODO:' in texto_periodo.upper():
                try:
                    texto_periodo = texto_periodo.split('PERIODO:')[-1].strip()
                    partes = texto_periodo.split('/')
                    if len(partes) >= 2:
                        dia_mes = partes[0].strip()
                        anio = partes[1].strip().split()[0]
                        mes = dia_mes[-2:]
                        return datetime(int(anio), int(mes), 1)
                except (ValueError, IndexError):
                    pass
    
    except (IndexError, ValueError) as e:
        print(f"Error al leer la celda de periodo: {e}")
    
    # Si no se pudo detectar, buscar en otras celdas comunes
    for i in range(min(15, len(df_raw))):
        for j in range(min(5, len(df_raw.columns))):
            celda = df_raw.iloc[i, j]
            if isinstance(celda, str):
                if 'MES:' in celda.upper():
                    try:
                        texto_mes = celda.split('MES:')[-1].strip()
                        mes = int(texto_mes.split('/')[0])
                        anio = int(texto_mes.split('/')[1])
                        return datetime(anio, mes, 1)
                    except (ValueError, IndexError):
                        continue
    
    # --- FALLBACK: extraer mes del nombre del archivo ---
    if archivo_excel:
        nombre = os.path.basename(archivo_excel)
        # Busca el nombre del mes en español (Enero, Febrero, ...)
        meses = {
            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
            'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
        }
        for mes_str, mes_num in meses.items():
            if mes_str in nombre:
                # Intentar obtener año (por defecto 2024 si no se encuentra)
                anio = 2024
                # Buscar un año de 4 dígitos en el nombre
                match = re.search(r'\b(20\d{2})\b', nombre)
                if match:
                    anio = int(match.group(1))
                print(f"  -> Mes detectado desde nombre de archivo: {mes_str}/{anio}")
                return datetime(anio, mes_num, 1)
    
    # Valor por defecto si no se detecta
    print("No se pudo detectar el periodo, usando Julio 2024 por defecto")
    return datetime(2024, 7, 1)

def _procesar_grupo_datos(fila, col_dia, col_hora, col_altura, col_caudal, datos_procesados, mes_anio):
    """
    Procesa un grupo individual de datos de caudal
    
    Args:
        fila: Fila del DataFrame
        col_dia: Índice de la columna del día
        col_hora: Índice de la columna de la hora
        col_altura: Índice de la columna de la altura
        col_caudal: Índice de la columna del caudal
        datos_procesados: Lista donde agregar los datos procesados
        mes_anio: Fecha base para construir el timestamp
    """
    try:
        # Verificar que hay suficientes columnas en este grupo
        if max(col_dia, col_hora, col_altura, col_caudal) >= len(fila):
            return
        
        # Extraer datos
        dia = fila[col_dia]
        hora = fila[col_hora]
        altura = fila[col_altura]
        caudal = fila[col_caudal]
        
        # Validar que todos los datos necesarios estén presentes
        if (pd.isna(dia) or pd.isna(hora) or pd.isna(altura) or pd.isna(caudal)):
            return
        
        # Convertir hora a componentes
        horas, minutos = _convertir_hora(hora)
        if horas is None:
            return
        
        # Crear timestamp
        fecha_hora = datetime(
            year=mes_anio.year,
            month=mes_anio.month,
            day=int(dia),
            hour=horas,
            minute=minutos
        )
        
        # Agregar a los datos procesados
        datos_procesados.append({
            'fecha_hora': fecha_hora,
            'altura_m': float(altura),
            'caudal_m3s': float(caudal)
        })
        
    except (ValueError, TypeError, IndexError) as e:
        # Error silencioso en procesamiento de grupo individual
        pass

def _convertir_hora(hora):
    """Convierte diferentes formatos de hora a horas y minutos"""
    try:
        if isinstance(hora, (int, float)):
            # Hora en formato decimal (ej: 12.5 = 12:30)
            horas = int(hora)
            minutos = int(round((hora - horas) * 60))
            return horas, min(59, minutos)
        elif isinstance(hora, str):
            # Hora en formato string
            hora_limpia = hora.strip()
            if '.' in hora_limpia:
                # Formato decimal como string "12.30"
                partes = hora_limpia.split('.')
                horas = int(partes[0])
                minutos = int(float(f"0.{partes[1]}") * 60) if len(partes) > 1 else 0
                return horas, min(59, minutos)
            elif ':' in hora_limpia:
                # Formato HH:MM
                partes = hora_limpia.split(':')
                horas = int(partes[0])
                minutos = int(partes[1]) if len(partes) > 1 else 0
                return horas, minutos
            else:
                # Intentar convertir a número
                return _convertir_hora(float(hora_limpia))
        else:
            return None, None
    except:
        return None, None

def guardar_netcdf(df, archivo_salida):
    """
    Guarda el DataFrame en formato NetCDF
    
    Args:
        df: DataFrame con datos de caudal y altura
        archivo_salida: Ruta del archivo NetCDF de salida
    """
    # Convertir DataFrame a xarray Dataset
    ds = xr.Dataset({
        'altura': (['time'], df['altura_m'].values),
        'caudal': (['time'], df['caudal_m3s'].values)
    }, coords={
        'time': df.index
    })
    
    # Agregar atributos descriptivos
    ds.altura.attrs = {'units': 'm', 'long_name': 'Altura de agua'}
    ds.caudal.attrs = {'units': 'm³/s', 'long_name': 'Caudal instantáneo'}
    ds.time.attrs = {'long_name': 'Tiempo'}
    
    # Guardar en NetCDF
    ds.to_netcdf(archivo_salida)
    print(f"Archivo NetCDF guardado: {archivo_salida}")

def crear_graficos(df, archivo_grafico):
    """
    Crea gráficos de la serie temporal de caudal y altura
    
    Args:
        df: DataFrame con datos de caudal y altura
        archivo_grafico: Ruta del archivo de imagen de salida
    """
    # Configurar estilo de los gráficos
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico de caudal
    ax1.plot(df.index, df['caudal_m3s'], color='blue', linewidth=1.0, label='Caudal')
    ax1.set_ylabel('Caudal (m³/s)', fontsize=12)
    ax1.set_title('Serie Temporal de Caudal - Estero Milahue (Todos los meses)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Gráfico de altura
    ax2.plot(df.index, df['altura_m'], color='green', linewidth=1.0, label='Altura')
    ax2.set_ylabel('Altura (m)', fontsize=12)
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_title('Serie Temporal de Altura - Estero Milahue (Todos los meses)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Rotar etiquetas para mejor lectura
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {archivo_grafico}")

def main():
    """Función principal del script - Procesa TODOS los archivos Excel de la carpeta"""
    
    # ------------------------------------------------------------
    # CONFIGURACIÓN DE RUTAS (MODIFICA SEGÚN TUS DIRECTORIOS)
    # ------------------------------------------------------------
    DIRECTORIO_ENTRADA = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Data_Caudales'
    DIRECTORIO_SALIDA  = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\03 Resultados_caudal_altura_estero'
    # ------------------------------------------------------------
    
    try:
        print("=== PROCESAMIENTO MÚLTIPLE DE ARCHIVOS DE CAUDAL ===\n")
        
        # Crear directorio de salida si no existe
        os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
        
        # Buscar todos los archivos Excel en el directorio de entrada
        patron_busqueda = os.path.join(DIRECTORIO_ENTRADA, '*.xls*')
        archivos_excel = glob.glob(patron_busqueda)
        
        if not archivos_excel:
            print(f"No se encontraron archivos Excel en: {DIRECTORIO_ENTRADA}")
            return
        
        print(f"Se encontraron {len(archivos_excel)} archivos para procesar:\n")
        for f in archivos_excel:
            print(f"  - {os.path.basename(f)}")
        print("\n")
        
        # Lista para almacenar DataFrames de cada mes
        lista_dfs = []
        archivos_fallidos = []
        
        # Procesar cada archivo
        for i, archivo in enumerate(archivos_excel, 1):
            nombre = os.path.basename(archivo)
            print(f"[{i}/{len(archivos_excel)}] Procesando: {nombre}")
            
            try:
                df_mes = procesar_datos_caudal(archivo)
                print(f"  -> Registros obtenidos: {len(df_mes)}")
                lista_dfs.append(df_mes)
            except Exception as e:
                print(f"  -> ERROR: {e}")
                archivos_fallidos.append((nombre, str(e)))
            
            print("-" * 60)
        
        # Verificar que se haya procesado al menos un archivo
        if not lista_dfs:
            print("No se pudo procesar ningún archivo. Abortando.")
            return
        
        # Combinar todos los DataFrames en uno solo
        print("\nCombinando todos los datos...")
        df_completo = pd.concat(lista_dfs)
        df_completo = df_completo.sort_index()  # ordenar por fecha
        print(f"Total de registros combinados: {len(df_completo)}")
        print(f"Periodo completo: {df_completo.index.min().strftime('%d/%m/%Y')} a {df_completo.index.max().strftime('%d/%m/%Y')}")
        
        # Guardar resultados en NetCDF (único archivo)
        archivo_netcdf = os.path.join(DIRECTORIO_SALIDA, 'caudales_alturas_completo.nc')
        guardar_netcdf(df_completo, archivo_netcdf)
        
        # Crear y guardar gráficos (serie completa)
        archivo_grafico = os.path.join(DIRECTORIO_SALIDA, 'serie_temporal_completa.png')
        crear_graficos(df_completo, archivo_grafico)
        
        # Mostrar resumen final
        print("\n=== PROCESAMIENTO COMPLETADO ===")
        print(f"Archivos procesados exitosamente: {len(lista_dfs)}/{len(archivos_excel)}")
        if archivos_fallidos:
            print("\nArchivos con errores:")
            for nombre, error in archivos_fallidos:
                print(f"  - {nombre}: {error}")
        
        print(f"\nResultados guardados en:")
        print(f"  - NetCDF: {archivo_netcdf}")
        print(f"  - Gráfico: {archivo_grafico}")
        
        # Estadísticas generales
        print("\n--- Estadísticas globales ---")
        print(f"Total de registros: {len(df_completo)}")
        print(f"Caudal - Mínimo: {df_completo['caudal_m3s'].min():.2f} m³/s, Máximo: {df_completo['caudal_m3s'].max():.2f} m³/s, Promedio: {df_completo['caudal_m3s'].mean():.2f} m³/s")
        print(f"Altura  - Mínimo: {df_completo['altura_m'].min():.2f} m, Máximo: {df_completo['altura_m'].max():.2f} m, Promedio: {df_completo['altura_m'].mean():.2f} m")
        
    except ImportError as e:
        print(f"Error de dependencias: {e}")
        print("Para resolverlo, ejecute: pip install xarray netcdf4 matplotlib openpyxl")
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()