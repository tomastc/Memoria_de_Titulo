import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from pykrige.ok import OrdinaryKriging
import xarray as xr
from scipy.spatial import ConvexHull
import time

def crear_directorio_resultados():
    """Crea directorio para resultados"""
    directorio = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\01 Resultados_mallado_kriging'
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    return directorio

def leer_archivo(nombre_archivo):
    """Lee y procesa el archivo CSV"""
    df = pd.read_csv(nombre_archivo, engine='python')
    
    # Identificar columnas automáticamente
    mapeo_columnas = {'x': ['x', 'este', 'easting'], 
                     'y': ['y', 'norte', 'northing'],
                     'z': ['z', 'elevacion', 'elevation', 'altura', 'profundidad', 'depth']}
    
    columnas_encontradas = {}
    for tipo, nombres in mapeo_columnas.items():
        for nombre in nombres:
            col = next((c for c in df.columns if nombre.lower() in c.lower()), None)
            if col:
                columnas_encontradas[tipo] = col
                break
        if tipo not in columnas_encontradas:
            columnas_numericas = df.select_dtypes(include=[np.number]).columns
            columnas_encontradas[tipo] = columnas_numericas[len(columnas_encontradas)]

    x_col, y_col, z_col = columnas_encontradas['x'], columnas_encontradas['y'], columnas_encontradas['z']
    
    # Limpiar datos
    df_clean = df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Calcular dimensiones
    x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
    y_min, y_max = df_clean[y_col].min(), df_clean[y_col].max()
    z_min, z_max = df_clean[z_col].min(), df_clean[z_col].max()
    
    x_dim = x_max - x_min
    y_dim = y_max - y_min
    z_dim = z_max - z_min
    
    print(f"Datos procesados: {len(df_clean)} puntos")
    print(f"Rango X: {x_min:.1f} a {x_max:.1f} (ΔX = {x_dim:.1f} m)")
    print(f"Rango Y: {y_min:.1f} a {y_max:.1f} (ΔY = {y_dim:.1f} m)")
    print(f"Rango Z: {z_min:.1f} a {z_max:.1f} (ΔZ = {z_dim:.1f} m)")
    
    return df_clean, x_col, y_col, z_col

def crear_malla_en_area_datos(df, x_col, y_col, espaciado=25):
    """Crea malla solo en el área donde existen datos usando convex hull"""
    x = df[x_col].values
    y = df[y_col].values
    
    # Calcular convex hull para definir el área de datos
    puntos = np.column_stack((x, y))
    hull = ConvexHull(puntos)
    
    # Obtener los límites del convex hull
    hull_puntos = puntos[hull.vertices]
    x_min, x_max = hull_puntos[:, 0].min(), hull_puntos[:, 0].max()
    y_min, y_max = hull_puntos[:, 1].min(), hull_puntos[:, 1].max()
    
    # Calcular dimensiones del área de datos
    x_dim = x_max - x_min
    y_dim = y_max - y_min
    
    print(f"Área de datos - X: {x_min:.1f} a {x_max:.1f} (ΔX = {x_dim:.1f} m)")
    print(f"Área de datos - Y: {y_min:.1f} a {y_max:.1f} (ΔY = {y_dim:.1f} m)")
    
    # Crear malla regular dentro del área de datos
    grid_x = np.arange(x_min, x_max, espaciado)
    grid_y = np.arange(y_min, y_max, espaciado)
    
    print(f"Malla en área de datos: {len(grid_x)} x {len(grid_y)} = {len(grid_x)*len(grid_y):,} puntos")
    
    return grid_x, grid_y, hull_puntos

def aplicar_kriging_simple(df, x_col, y_col, z_col, espaciado=25):
    """Aplica kriging solo en el área con datos"""
    print("Aplicando kriging en área de datos...")
    
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    
    # Crear malla solo en área de datos
    grid_x, grid_y, hull_puntos = crear_malla_en_area_datos(df, x_col, y_col, espaciado)
    
    # Aplicar kriging
    OK = OrdinaryKriging(x, y, z, variogram_model='linear', verbose=False)
    z_grid, ss = OK.execute('grid', grid_x, grid_y)
    
    # Asegurar dimensiones correctas
    if z_grid.shape != (len(grid_y), len(grid_x)):
        z_grid = z_grid.T
        
    return grid_x, grid_y, z_grid, hull_puntos

def aplicar_mascara_datos(grid_x, grid_y, z_grid, hull_puntos):
    """Aplica máscara para mostrar solo datos dentro del convex hull"""
    from matplotlib.path import Path
    
    # Crear path del convex hull
    hull_path = Path(hull_puntos)
    
    # Crear malla de coordenadas
    XX, YY = np.meshgrid(grid_x, grid_y)
    puntos_malla = np.column_stack((XX.ravel(), YY.ravel()))
    
    # Crear máscara
    mascara = hull_path.contains_points(puntos_malla)
    mascara = mascara.reshape(XX.shape)
    
    # Aplicar máscara
    z_grid_masked = z_grid.copy()
    z_grid_masked[~mascara] = np.nan
    
    # Calcular estadísticas después del enmascarado
    z_valid = z_grid_masked[~np.isnan(z_grid_masked)]
    if len(z_valid) > 0:
        z_min_masked = np.min(z_valid)
        z_max_masked = np.max(z_valid)
        z_dim_masked = z_max_masked - z_min_masked
        print(f"Después de enmascarar - Z: {z_min_masked:.1f} a {z_max_masked:.1f} (ΔZ = {z_dim_masked:.1f} m)")
    
    return z_grid_masked

def guardar_netcdf_simple(grid_x, grid_y, z_grid, directorio):
    """Guarda el mallado en netCDF"""
    # Asegurar dimensiones
    if z_grid.shape != (len(grid_y), len(grid_x)):
        z_grid = z_grid.T
    
    # Crear DataArray
    da = xr.DataArray(
        z_grid,
        coords={'y': grid_y, 'x': grid_x},
        dims=['y', 'x'],
        name='elevacion',
        attrs={'units': 'metros', 'description': 'Superficie kriging en área de datos'}
    )
    
    ruta = os.path.join(directorio, "superficie_kriging.nc")
    da.to_netcdf(ruta)
    print(f"Archivo netCDF guardado: {ruta}")
    return ruta

def graficar_superficie_3d_simple(grid_x, grid_y, z_grid, directorio):
    """Genera gráfico 3D simple y efectivo"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear malla para plotting
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Graficar superficie
    surf = ax.plot_surface(X, Y, z_grid, cmap='terrain', 
                          alpha=0.9, antialiased=True, linewidth=0.2)
    
    # Barra de colores
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Elevación (m)', fontsize=12)
    
    # Configuración
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevación (m)', fontsize=12, labelpad=10)
    ax.set_title('Superficie Batimétrica - Kriging (Solo área con datos)', fontsize=14, pad=20)
    
    # Vista óptima
    ax.view_init(elev=35, azim=45)
    ax.grid(True, alpha=0.3)
    
    # Guardar imagen
    ruta_imagen = os.path.join(directorio, "superficie_3d.png")
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    print(f"Imagen 3D guardada: {ruta_imagen}")
    
    plt.show()

def main():
    """Función principal simple y efectiva"""
    nombre_archivo = r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Data_TopoBatimetria\Puntos_Topo_Batimetria.csv'
    espaciado = 20  # metros
    
    try:
        # Iniciar medición de tiempo
        tiempo_inicio = time.time()
        
        # Crear directorio
        directorio = crear_directorio_resultados()
        
        print("=" * 50)
        print("PROCESAMIENTO DE DATOS BATIMÉTRICOS")
        print("=" * 50)
        
        # Leer datos
        print("\n[1/4] Leyendo archivo...")
        df, x_col, y_col, z_col = leer_archivo(nombre_archivo)
        
        # Aplicar kriging en área de datos
        print("\n[2/4] Aplicando kriging...")
        grid_x, grid_y, z_grid, hull_puntos = aplicar_kriging_simple(df, x_col, y_col, z_col, espaciado)
        
        # Aplicar máscara para mostrar solo datos dentro del área
        print("\n[3/4] Aplicando máscara de datos...")
        z_grid_masked = aplicar_mascara_datos(grid_x, grid_y, z_grid, hull_puntos)
        
        # Guardar netCDF
        print("\n[4/4] Guardando resultados...")
        ruta_netcdf = guardar_netcdf_simple(grid_x, grid_y, z_grid_masked, directorio)
        
        # Graficar
        print("\nGenerando visualización...")
        graficar_superficie_3d_simple(grid_x, grid_y, z_grid_masked, directorio)
        
        # Calcular tiempo total
        tiempo_total = time.time() - tiempo_inicio
        
        print("\n" + "=" * 50)
        print("RESUMEN FINAL")
        print("=" * 50)
        print(f"Tiempo total de procesamiento: {tiempo_total:.2f} segundos")
        print(f"Puntos de entrada: {len(df):,}")
        print(f"Puntos de malla: {len(grid_x) * len(grid_y):,}")
        print(f"Espaciado de malla: {espaciado} m")
        print(f"Archivo netCDF: {ruta_netcdf}")
        print(f"Directorio de resultados: {directorio}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()