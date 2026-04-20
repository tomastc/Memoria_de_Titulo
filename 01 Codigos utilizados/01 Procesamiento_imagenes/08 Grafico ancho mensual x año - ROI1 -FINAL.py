import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def graficar_comparacion_anual_2023_2024(df, carpeta_resultados):
    """
    Calcula el promedio mensual de los años 2023 y 2024 (filtrando valores válidos) 
    y genera una gráfica comparativa.
    """
    print("\n" + "="*80)
    print("GENERANDO GRÁFICO COMPARATIVO DE ANCHOS (2023 vs 2024)")
    print("="*80)
    
    # Extraer año y mes
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    
    # Filtrar solo instantes válidos (abierto y ancho > 0) para no distorsionar el promedio
    df_validos = df[(df['estado'] == 1) & (df['ancho'] > 0)].copy()
    
    if len(df_validos) == 0:
        print("ADVERTENCIA: No hay datos válidos (>0) para calcular promedios.")
        return False

    # Separar por años
    df_2023 = df_validos[df_validos['año'] == 2023]
    df_2024 = df_validos[df_validos['año'] == 2024]
    
    # Agrupar por mes y promediar. reindex garantiza que estén los 12 meses
    promedios_2023 = df_2023.groupby('mes')['ancho'].mean().reindex(range(1, 13))
    promedios_2024 = df_2024.groupby('mes')['ancho'].mean().reindex(range(1, 13))
    
    # Generar la gráfica
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    meses_num = range(1, 13)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Serie 2023
    ax.plot(meses_num, promedios_2023, marker='o', markersize=8, linewidth=2.5, 
            color='tab:blue', label='Promedio 2023')
    
    # Serie 2024
    ax.plot(meses_num, promedios_2024, marker='s', markersize=8, linewidth=2.5, 
            color='tab:orange', label='Promedio 2024')
    
    # Formato del gráfico
    ax.set_title("Variación mensual de anchos por año", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Mes", fontsize=12)
    ax.set_ylabel("Ancho promedio de desembocadura (m)", fontsize=12)
    
    ax.set_xticks(meses_num)
    ax.set_xticklabels(meses_nombres, fontsize=11)
    
    # Límites del eje Y (iniciar en 0)
    y_max = max(promedios_2023.max(), promedios_2024.max())
    if pd.notna(y_max):
        ax.set_ylim(bottom=0, top=y_max * 1.15)
        
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    # Guardar la imagen en la misma carpeta del archivo procesado
    nombre_archivo = "Resultados_Ancho_Canal_Comparacion_2023_2024.png"
    ruta_completa = os.path.join(carpeta_resultados, nombre_archivo)
    
    plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Gráfico generado exitosamente.")
    print(f"Guardado en: {ruta_completa}")
    print("="*80)
    
    return True

def main():
    # Configurar rutas
    carpeta_base = 'C:/Users/Tomas/Desktop/Escritorio/memoria/03 Codigos/01 Lee_imagen'
    
    # Buscar el archivo más reciente de anchos
    patron_archivos = os.path.join(carpeta_base, 'Resultados_Ancho_Canal_*', 'Data_Ancho_Desde_Limites_*_limpio.nc')
    archivos_encontrados = glob.glob(patron_archivos)
    
    if not archivos_encontrados:
        print("ERROR: No se encontraron archivos de anchos.")
        print(f"Buscando en: {patron_archivos}")
        return
    
    # Seleccionar el archivo más reciente
    archivo_anchos = max(archivos_encontrados, key=os.path.getctime)
    print(f"Archivo de anchos encontrado: {archivo_anchos}")
    
    # Obtener carpeta de resultados del archivo seleccionado
    carpeta_resultados = os.path.dirname(archivo_anchos)
    print(f"Carpeta de resultados: {carpeta_resultados}")
    
    # Abrir dataset para extraer DataFrame
    try:
        ds = xr.open_dataset(archivo_anchos)
        tiempos = ds.time.values
        estados = ds.estado_apertura.values
        anchos = ds.ancho_desembocadura.values
        
        df = pd.DataFrame({
            'fecha': pd.to_datetime(tiempos),
            'estado': estados,
            'ancho': anchos
        }).sort_values('fecha')
        
        ds.close()
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return
        
    # Ejecutar la función de graficado con los datos cargados
    graficar_comparacion_anual_2023_2024(df, carpeta_resultados)

if __name__ == "__main__":
    main()