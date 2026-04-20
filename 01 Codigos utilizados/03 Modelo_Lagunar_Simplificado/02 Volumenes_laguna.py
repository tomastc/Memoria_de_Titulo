import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def crear_directorio_resultados():
    """Crea directorio para resultados de volúmenes"""
    directorio = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\02 Resultados_volumenes'
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    return directorio

def cargar_mallado_netcdf(ruta_netcdf):
    """Carga el archivo netCDF con el mallado de kriging"""
    print("Cargando archivo netCDF...")
    ds = xr.open_dataset(ruta_netcdf)
    elevacion = ds['elevacion'].values
    x = ds['x'].values
    y = ds['y'].values
    
    print(f"Mallado cargado: {elevacion.shape}")
    print(f"Rango elevaciones: {np.nanmin(elevacion):.3f} a {np.nanmax(elevacion):.3f} m")
    
    return elevacion, x, y, ds

def calcular_volumen_para_altura(elevacion, altura, dx, dy):
    """Calcula eficientemente el volumen para una altura dada"""
    # Máscara de puntos válidos (no NaN)
    mascara_validos = ~np.isnan(elevacion)
    
    # Calcular diferencia de altura solo para puntos válidos
    diferencia = altura - elevacion[mascara_validos]
    
    # Volumen solo donde la diferencia es positiva (agua)
    volumen_celdas = np.where(diferencia > 0, diferencia * dx * dy, 0)
    
    return np.sum(volumen_celdas)

def generar_curva_volumen(elevacion, x, y, paso_altura=0.005):
    """Genera la curva de volumen vs altura de forma eficiente"""
    print("Calculando curva de volumen...")
    tiempo_inicio = time.time()
    
    # Calcular dimensiones de celda
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    area_celda = dx * dy
    
    print(f"Área por celda: {area_celda:.2f} m²")
    print(f"Paso de altura: {paso_altura} m")
    
    # Determinar rango de alturas
    altura_min = np.nanmin(elevacion)
    altura_max = np.nanmax(elevacion)
    
    # Asegurar que la altura máxima incluya toda la superficie
    altura_max += paso_altura * 2
    
    # Crear array de alturas
    alturas = np.arange(altura_min, altura_max, paso_altura)
    n_puntos = len(alturas)
    
    print(f"Rango de cálculo: {altura_min:.3f} a {altura_max:.3f} m")
    print(f"Número de puntos en curva: {n_puntos}")
    
    # Precalcular máscara de puntos válidos una sola vez
    mascara_validos = ~np.isnan(elevacion)
    elevacion_valida = elevacion[mascara_validos]
    n_celdas_validas = len(elevacion_valida)
    
    print(f"Celdas válidas para cálculo: {n_celdas_validas}")
    
    # Vectorizar cálculo de volúmenes
    volumenes = np.zeros(n_puntos)
    
    for i, altura in enumerate(alturas):
        if i % 100 == 0 and i > 0:  # Mostrar progreso cada 100 puntos
            print(f"  Procesado {i}/{n_puntos} puntos...")
        
        diferencia = altura - elevacion_valida
        volumen_celdas = np.where(diferencia > 0, diferencia * area_celda, 0)
        volumenes[i] = np.sum(volumen_celdas)
    
    tiempo_calculo = time.time() - tiempo_inicio
    print(f"✓ Cálculo completado en {tiempo_calculo:.2f} segundos")
    
    return alturas, volumenes, area_celda

def encontrar_punto_maximo(alturas, volumenes):
    """Encuentra el volumen máximo y su altura correspondiente"""
    idx_max = np.argmax(volumenes)
    altura_max = alturas[idx_max]
    volumen_max = volumenes[idx_max]
    return altura_max, volumen_max, idx_max

def guardar_curva_volumen_netcdf(alturas, volumenes, directorio):
    """Guarda la curva de volumen en archivo netCDF"""
    # Crear Dataset
    ds = xr.Dataset({
        'volumen': (['altura'], volumenes),
        'altura': (['altura'], alturas)
    })
    
    # Añadir atributos
    ds.volumen.attrs = {'units': 'm³', 'long_name': 'Volumen de agua'}
    ds.altura.attrs = {'units': 'm', 'long_name': 'Altura de agua'}
    ds.attrs['title'] = 'Curva de Volumen vs Altura'
    ds.attrs['source'] = 'Cálculo a partir de mallado Kriging'
    ds.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    ruta = os.path.join(directorio, "curva_volumen_altura.nc")
    ds.to_netcdf(ruta)
    print(f"✓ Curva de volumen guardada: {ruta}")
    
    return ruta

def graficar_curva_volumen(alturas, volumenes, directorio):
    """Genera gráfico profesional de la curva de volumen"""
    # Encontrar punto máximo
    altura_max, volumen_max, idx_max = encontrar_punto_maximo(alturas, volumenes)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graficar curva principal
    line = ax.plot(volumenes, alturas, 'b-', linewidth=2.5, 
                   label='Curva Volumen-Altura', alpha=0.8)
    
    # Resaltar punto máximo
    ax.plot(volumen_max, altura_max, 'ro', markersize=10, 
            label=f'Máximo: {volumen_max/1e6:.2f} Mm³\nAltura: {altura_max:.3f} m')
    

    
    # Configurar ejes y etiquetas
    ax.set_xlabel('Volumen de Agua (m³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altura de Agua (m)', fontsize=12, fontweight='bold')
    ax.set_title('Curva de Volumen vs Altura - Análisis Batimétrico', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Formatear ejes
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Leyenda
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    
    # Añadir texto informativo
    texto_info = f'''Parámetros:
• Volumen máximo: {volumen_max/1e6:.3f} Mm³
• Altura máxima: {altura_max:.3f} m
• Rango alturas: {alturas[0]:.3f} - {alturas[-1]:.3f} m'''
    
    ax.text(0.02, 0.98, texto_info, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace', fontsize=9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar imagen
    ruta_imagen = os.path.join(directorio, "curva_volumen.png")
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Gráfico guardado: {ruta_imagen}")
    
    plt.show()

def main():
    """Función principal"""
    # CONFIGURACIÓN - PARÁMETRO AJUSTABLE
    PASO_ALTURA = 0.1  # metros - Se puede cambiar manualmente
    
    # Ruta al archivo netCDF generado previamente
    ruta_netcdf = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\01 Resultados_mallado_kriging\superficie_kriging.nc'
    
    try:
        tiempo_inicio_total = time.time()
        
        # Crear directorio de resultados
        directorio = crear_directorio_resultados()
        
        print("=" * 60)
        print("CÁLCULO DE CURVA VOLUMEN-ALTURA")
        print("=" * 60)
        print(f"Paso de altura: {PASO_ALTURA} m")
        
        # Cargar mallado
        elevacion, x, y, ds_original = cargar_mallado_netcdf(ruta_netcdf)
        
        # Generar curva de volumen
        alturas, volumenes, area_celda = generar_curva_volumen(
            elevacion, x, y, PASO_ALTURA
        )
        
        # Encontrar y mostrar punto máximo
        altura_max, volumen_max, idx_max = encontrar_punto_maximo(alturas, volumenes)
        
        print(f"\n--- RESULTADOS PRINCIPALES ---")
        print(f"Volumen máximo: {volumen_max:,.0f} m³ ({volumen_max/1e6:.3f} Mm³)")
        print(f"Altura de volumen máximo: {altura_max:.3f} m")
        print(f"Índice en curva: {idx_max}/{len(alturas)}")
        
        # Guardar resultados
        ruta_netcdf_curva = guardar_curva_volumen_netcdf(alturas, volumenes, directorio)
        
        # Generar gráfico
        graficar_curva_volumen(alturas, volumenes, directorio)
        
        # Tiempo total
        tiempo_total = time.time() - tiempo_inicio_total
        
        print(f"\n" + "=" * 60)
        print("RESUMEN FINAL")
        print("=" * 60)
        print(f"Tiempo total de procesamiento: {tiempo_total:.2f} segundos")
        print(f"Paso de altura utilizado: {PASO_ALTURA} m")
        print(f"Puntos en curva: {len(alturas):,}")
        print(f"Volumen máximo calculado: {volumen_max/1e6:.3f} millones de m³")
        print(f"Archivo de curva: {ruta_netcdf_curva}")
        print(f"Directorio de resultados: {directorio}")
        print("=" * 60)
        
        # Cerrar dataset original
        ds_original.close()
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_netcdf}")
        print("Asegúrate de que el archivo netCDF existe en la ruta especificada")
    except Exception as e:
        print(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()