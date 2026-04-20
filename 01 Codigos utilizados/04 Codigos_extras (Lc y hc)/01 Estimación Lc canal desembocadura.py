import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tifffile as tiff

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CARPETA_IMAGENES = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\06 Codigos extras (Lc y hc)\imagenes de prueba"
ESCALA_MM_POR_PIXEL = 600  # 1 píxel = 600 mm (equivalente a 0.6 m/pixel)
ARCHIVO_SALIDA = r"C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\06 Codigos extras (Lc y hc)\Resultados Largo Canal Lc\Resultados Largo Canal Lc.txt"
COLUMNAS = 5

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def cargar_imagenes_desde_carpeta(carpeta):
    extensiones = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    archivos = []
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(carpeta, ext)))

    imagenes = []
    nombres = []
    for archivo in archivos:
        try:
            img = tiff.imread(archivo)
        except:
            img = plt.imread(archivo)
        imagenes.append(img)
        nombres.append(os.path.basename(archivo))
        print(f"Cargada: {os.path.basename(archivo)} - Dimensiones: {img.shape}")
    return imagenes, nombres

def calcular_distancia(p1, p2, escala_mm_por_pixel):
    """
    Calcula la distancia en píxeles y la convierte a metros.
    Devuelve (dist_px, dist_m)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist_px = np.linalg.norm(p2 - p1)
    dist_m = dist_px * escala_mm_por_pixel / 1000.0  # convertir mm a m
    return dist_px, dist_m

def angulo_entre_puntos(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

# =============================================================================
# CLASE PRINCIPAL
# =============================================================================
class MedidorDistancias:
    def __init__(self, imagenes, nombres):
        self.imagenes = imagenes
        self.nombres = nombres
        self.n_imagenes = len(imagenes)
        self.distancias = [None] * self.n_imagenes          # almacenadas en metros
        self.lineas = [None] * self.n_imagenes
        self.textos = [None] * self.n_imagenes
        self.puntos_actuales = []            # coordenadas de la selección actual
        self.puntos_actuales_artistas = []   # objetos gráficos de los puntos actuales
        self.indice_actual = 0

        # Crear figura y cuadrícula
        n_filas = int(np.ceil(self.n_imagenes / COLUMNAS))
        self.fig, self.axes = plt.subplots(n_filas, COLUMNAS, figsize=(4*COLUMNAS, 4*n_filas))
        self.axes = np.array(self.axes).flatten()

        for ax in self.axes[self.n_imagenes:]:
            ax.axis('off')

        for i, ax in enumerate(self.axes[:self.n_imagenes]):
            ax.imshow(self.imagenes[i])
            ax.set_title(self.nombres[i], fontsize=10)
            ax.axis('off')

        self.actualizar_resaltado()

        # Crear botones
        plt.subplots_adjust(bottom=0.2)  # Más espacio para tres botones
        ax_btn_borrar = plt.axes([0.1, 0.05, 0.15, 0.05])
        ax_btn_medir = plt.axes([0.3, 0.05, 0.2, 0.05])
        ax_btn_promedio = plt.axes([0.55, 0.05, 0.2, 0.05])

        self.btn_borrar = Button(ax_btn_borrar, 'Borrar último punto')
        self.btn_medir = Button(ax_btn_medir, 'Ingresar Medida')
        self.btn_promedio = Button(ax_btn_promedio, 'Calcular Largo Promedio')

        self.btn_borrar.on_clicked(self.borrar_ultimo_punto)
        self.btn_medir.on_clicked(self.ingresar_medida)
        self.btn_promedio.on_clicked(self.calcular_y_guardar)

        # Conectar eventos
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    # -------------------------------------------------------------------------
    # Métodos de actualización de la interfaz
    # -------------------------------------------------------------------------
    def actualizar_resaltado(self):
        for i, ax in enumerate(self.axes[:self.n_imagenes]):
            for spine in ax.spines.values():
                spine.set_color('red' if i == self.indice_actual else 'black')
                spine.set_linewidth(3 if i == self.indice_actual else 1)
        self.fig.canvas.draw_idle()

    def limpiar_anotaciones(self, indice):
        if self.lineas[indice] is not None:
            self.lineas[indice].remove()
            self.lineas[indice] = None
        if self.textos[indice] is not None:
            self.textos[indice].remove()
            self.textos[indice] = None

    def limpiar_puntos_actuales(self):
        for art in self.puntos_actuales_artistas:
            art.remove()
        self.puntos_actuales_artistas.clear()

    def actualizar_visualizacion_actual(self):
        """Redibuja la selección actual (puntos, línea y texto) en la imagen activa."""
        ax = self.axes[self.indice_actual]

        # Limpiar elementos previos de la selección actual
        self.limpiar_puntos_actuales()

        if len(self.puntos_actuales) == 1:
            # Dibujar un solo punto
            p = self.puntos_actuales[0]
            punto, = ax.plot(p[0], p[1], 'yo', markersize=8, markeredgecolor='black')
            self.puntos_actuales_artistas.append(punto)

        elif len(self.puntos_actuales) == 2:
            # Dibujar línea, puntos y texto
            p1, p2 = self.puntos_actuales
            dist_px, dist_m = calcular_distancia(p1, p2, ESCALA_MM_POR_PIXEL)

            # Eliminar cualquier anotación guardada de esta imagen (la nueva medida reemplaza a la anterior)
            self.limpiar_anotaciones(self.indice_actual)
            self.distancias[self.indice_actual] = None  # La nueva aún no está guardada

            # Dibujar línea
            linea, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
            self.puntos_actuales_artistas.append(linea)

            # Dibujar puntos
            p1_plot, = ax.plot(p1[0], p1[1], 'go', markersize=6)
            p2_plot, = ax.plot(p2[0], p2[1], 'bo', markersize=6)
            self.puntos_actuales_artistas.extend([p1_plot, p2_plot])

            # Texto en el punto medio (en metros con 2 decimales)
            xm = (p1[0] + p2[0]) / 2
            ym = (p1[1] + p2[1]) / 2
            ang = angulo_entre_puntos(p1, p2)
            texto = ax.text(xm, ym, f'{dist_m:.2f} m',
                            fontsize=10, color='yellow',
                            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                            rotation=ang, rotation_mode='anchor',
                            ha='center', va='center')
            self.puntos_actuales_artistas.append(texto)

        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Eventos
    # -------------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is None:
            return

        # Identificar qué subplot se clickeó
        for i, ax in enumerate(self.axes[:self.n_imagenes]):
            if ax == event.inaxes:
                if i != self.indice_actual:
                    # Ignorar clics en otras imágenes
                    return
                break
        else:
            return

        # Si la imagen actual ya tenía una medida guardada y empezamos una nueva selección,
        # eliminamos la medida guardada.
        if self.distancias[self.indice_actual] is not None and len(self.puntos_actuales) == 0:
            self.limpiar_anotaciones(self.indice_actual)
            self.distancias[self.indice_actual] = None

        # Si ya hay dos puntos, reiniciamos la selección
        if len(self.puntos_actuales) == 2:
            self.puntos_actuales = []
            self.limpiar_puntos_actuales()

        # Agregar el nuevo punto
        self.puntos_actuales.append((event.xdata, event.ydata))
        self.actualizar_visualizacion_actual()

    def borrar_ultimo_punto(self, event):
        """Elimina el último punto ingresado en la imagen actual."""
        if len(self.puntos_actuales) == 0:
            print("No hay puntos para borrar.")
            return

        self.puntos_actuales.pop()
        self.actualizar_visualizacion_actual()
        print("Último punto borrado.")

    def ingresar_medida(self, event):
        if self.indice_actual >= self.n_imagenes:
            print("Ya se procesaron todas las imágenes.")
            return
        if len(self.puntos_actuales) != 2:
            print("Debe seleccionar dos puntos en la imagen actual antes de ingresar la medida.")
            return

        # Calcular y guardar (distancia en metros)
        p1, p2 = self.puntos_actuales
        _, dist_m = calcular_distancia(p1, p2, ESCALA_MM_POR_PIXEL)
        self.distancias[self.indice_actual] = dist_m

        # Dibujar la medida de forma permanente
        ax = self.axes[self.indice_actual]
        self.limpiar_anotaciones(self.indice_actual)
        linea, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
        ax.plot(p1[0], p1[1], 'go', markersize=6)
        ax.plot(p2[0], p2[1], 'bo', markersize=6)
        xm = (p1[0] + p2[0]) / 2
        ym = (p1[1] + p2[1]) / 2
        ang = angulo_entre_puntos(p1, p2)
        texto = ax.text(xm, ym, f'{dist_m:.2f} m',
                        fontsize=10, color='yellow',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                        rotation=ang, rotation_mode='anchor',
                        ha='center', va='center')
        self.lineas[self.indice_actual] = linea
        self.textos[self.indice_actual] = texto

        # Limpiar selección actual
        self.puntos_actuales = []
        self.limpiar_puntos_actuales()
        self.fig.canvas.draw_idle()

        print(f"Medida guardada para {self.nombres[self.indice_actual]}: {dist_m:.2f} m")

        # Avanzar a la siguiente imagen
        self.indice_actual += 1
        if self.indice_actual < self.n_imagenes:
            self.actualizar_resaltado()
        else:
            print("Todas las imágenes han sido procesadas.")
            self.btn_medir.active = False

    def calcular_y_guardar(self, event):
        distancias_validas = [d for d in self.distancias if d is not None]
        if len(distancias_validas) == 0:
            print("No hay medidas registradas.")
            return

        promedio = np.mean(distancias_validas)
        desviacion = np.std(distancias_validas, ddof=1)

        print("\n" + "="*50)
        print("RESULTADOS (en metros)")
        print("="*50)
        for nombre, dist in zip(self.nombres, self.distancias):
            if dist is not None:
                print(f"{nombre}: {dist:.2f} m")
            else:
                print(f"{nombre}: No medida")
        print(f"\nPromedio: {promedio:.2f} m")
        print(f"Desviación estándar: {desviacion:.2f} m")
        print("="*50)

        # Crear el directorio si no existe
        directorio = os.path.dirname(ARCHIVO_SALIDA)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio)
            print(f"Directorio creado: {directorio}")

        # Guardar en archivo
        with open(ARCHIVO_SALIDA, 'w') as f:
            f.write("Resultados de medición de largo de canal (en metros)\n")
            f.write("===================================================\n\n")
            for nombre, dist in zip(self.nombres, self.distancias):
                if dist is not None:
                    f.write(f"{nombre}: {dist:.2f} m\n")
                else:
                    f.write(f"{nombre}: No medida\n")
            f.write(f"\nPromedio: {promedio:.2f} m\n")
            f.write(f"Desviación estándar: {desviacion:.2f} m\n")
        print(f"Resultados guardados en '{ARCHIVO_SALIDA}'")

    def on_key(self, event):
        if event.key == 'q':
            plt.close(self.fig)
            print("Programa terminado por el usuario.")

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    if not os.path.isdir(CARPETA_IMAGENES):
        print(f"ERROR: No se encuentra la carpeta '{CARPETA_IMAGENES}'.")
        exit(1)

    imagenes, nombres = cargar_imagenes_desde_carpeta(CARPETA_IMAGENES)
    if len(imagenes) == 0:
        print("No se encontraron imágenes en la carpeta.")
        exit(1)

    print(f"\nSe cargaron {len(imagenes)} imágenes.")
    app = MedidorDistancias(imagenes, nombres)
    plt.show()
    print("Programa terminado.")