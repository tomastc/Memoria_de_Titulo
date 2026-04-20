===========================================================
       README - Análisis de Evolución Perímetros Agua-Arena
===========================================================

Este documento explica cómo usar los códigos 02, 03 y 04, que continúan
el proceso iniciado con el código 01 (segmentación de agua y arena).

Si aún no ha ejecutado el código 01, hágalo primero. Ese código genera
un archivo NetCDF con las máscaras de agua y arena para cada imagen.

===========================================================
CÓDIGO 02: CREAR GRILLA DE MUESTREO DENTRO DEL ÁREA DE INTERÉS
===========================================================

¿Qué hace?
----------
Toma el archivo con las máscaras (resultado del código 01) y crea una
grilla de puntos espaciados regularmente dentro del polígono ROI.
Luego, para cada imagen (cada fecha), anota en cada punto de la grilla
si hay agua o arena. El resultado es una tabla (archivo NetCDF) que
contiene, para cada fecha y cada punto de la grilla, un valor que indica
"agua" (1) o "no agua" (0) y otro para "arena".

También genera una imagen de ejemplo que muestra las máscaras de agua
(azul), arena (amarillo), la grilla (líneas grises) y el contorno del ROI.

Entradas
-----------------------
- El archivo NetCDF generado por el código 01 (ej: "resultados_mascaras_ROI1.nc").
- El polígono ROI (los mismos puntos que usó en el código 01).
- El espaciado de la grilla (en píxeles). Valor recomendado: 1.

Salidas
---------------------
- Un archivo NetCDF (ej: "grilla_mascaras.nc") con:
    * Coordenadas X e Y de cada punto de la grilla.
    * Para cada fecha, el valor de agua (1 o 0) y arena (1 o 0) en cada punto.
    * Una copia de los nombres de archivo originales.
- Una imagen PNG que muestra, para una fecha que usted elija, las máscaras
  de agua/arena, la grilla y el contorno ROI.
- Un reporte de texto ("reporte_procesamiento_Grilla.txt") con estadísticas:
    * Cantidad de imágenes procesadas.
    * Cantidad de puntos de grilla.
    * Tiempo de procesamiento, tamaños de archivo, etc.

Cómo usarlo
-----------
1. Abra el código 02.
2. Modifique las variables de CONFIGURACIÓN al inicio del archivo:
   - ESPACIADO_GRILLA: número entero (1 = un punto por píxel, 2 = un punto cada 2 píxeles).
   - ROI_POINTS: los mismos puntos que usó en el código 01.
   - ARCHIVO_NETCDF_ENTRADA: ruta completa al archivo .nc generado por el código 01.
   - CARPETA_SALIDA: carpeta donde se guardarán los resultados.
   - FECHA_VISUALIZACION: fecha y hora de una imagen para generar la vista de ejemplo
     (formato "AAAA-MM-DD HH:MM").
3. Ejecute el código completo.
4. Revise la carpeta de salida: encontrará el archivo NetCDF, la imagen de ejemplo
   y el reporte de texto.

===========================================================
CÓDIGO 03: CALCULAR EL PERÍMETRO REAL ENTRE AGUA Y ARENA
===========================================================

¿Qué hace?
----------
A partir de la grilla generada por el código 02, este código identifica
los puntos que están justo en el borde de la arena que toca el agua.
Es decir, el "perímetro" real de la playa (de un solo punto de ancho).
Además, elimina los puntos que caen en el borde del ROI para no contar
falsos bordes.

Luego guarda solo esos puntos de perímetro en un nuevo archivo NetCDF,
y genera una visualización con tres fechas que usted elija, mostrando
el agua (azul), la arena (amarillo) y el perímetro (puntos rojos).

Entradas
-----------------------
- El archivo NetCDF de grilla generado por el código 02 (ej: "grilla_mascaras.nc").
- Las fechas que desea visualizar (tres fechas, formato "AAAA-MM-DD HH:MM").

Salidas
---------------------
- Un archivo NetCDF (ej: "Data_perimetros_arena_agua.nc") que contiene:
    * Las coordenadas de los puntos que forman el perímetro.
    * Para cada fecha, qué puntos de perímetro están activos.
    * Indicador de si la imagen tenía máscara de agua válida.
- Una imagen PNG con tres gráficos (uno por fecha) mostrando:
    * Fondo: agua (azul) y arena (amarillo).
    * Perímetro: puntos rojos.
- Un reporte de texto ("reporte_procesamiento_Perimetros.txt") con:
    * Cantidad total de puntos de perímetro.
    * Relación entre puntos de perímetro y puntos de arena.
    * Detalle por cada fecha (cuántos puntos de perímetro y arena tuvo).

Cómo usarlo
-----------
1. Abra el código 03.
2. Verifique las rutas al inicio del programa principal (abajo del todo):
   - archivo_grilla_entrada: ruta al archivo .nc del código 02.
   - carpeta_salida: dónde guardar los resultados.
   - dias_a_mostrar: lista con tres fechas para visualizar.
3. Ejecute el código.
4. Revise la carpeta de salida: archivo NetCDF de perímetros, imagen con los
   tres gráficos, y reporte de texto.

===========================================================
CÓDIGO 04: ANALIZAR LA EVOLUCIÓN TEMPORAL DEL PERÍMETRO
===========================================================

¿Qué hace?
----------
Toma el archivo de perímetros (código 03) y, para un período de tiempo
que usted elija (fecha de inicio y número de días), genera dos productos:

1. Una gráfica "superpuesta": todos los puntos de perímetro de todas
   las fechas del período se dibujan juntos. Cada día tiene un color
   diferente (si son varios días). Si es un solo día, cada instante
   (hora) tiene un color diferente. Así se ve cómo se mueve la línea
   de costa en el tiempo.

2. Un video (MP4 o GIF) que muestra la evolución paso a paso:
   - Los perímetros de días anteriores se muestran en el color de su día.
   - El perímetro del instante actual se muestra en ROJO.
   - El título indica la fecha y el progreso.

Entradas
-----------------------
- El archivo NetCDF de perímetros (código 03).
- El archivo NetCDF de grilla (código 02) – se usa solo para coordenadas.
- Una fecha de inicio (formato "AAAA-MM-DD HH:MM").
- El número de días a analizar (entero).

Salidas (qué produce)
---------------------
- Una imagen PNG con la gráfica superpuesta.
- Un video MP4 (o GIF si no tiene FFmpeg) con la evolución.
- (Opcional) Mensajes en pantalla con el progreso.

Cómo usarlo
-----------
1. Abra el código 04.
2. Modifique las variables al inicio del bloque "if __name__ == '__main__':":
   - FECHA_INICIO: la fecha y hora desde donde empezar (ej: "2024-07-10 00:00").
   - NUMERO_DIAS: cuántos días hacia adelante analizar (ej: 5).
3. Verifique que las rutas a los archivos de entrada sean correctas:
   - archivo_perimetros (resultado del código 03)
   - archivo_grilla_original (resultado del código 02)
4. Ejecute el código.
5. En la carpeta de salida aparecerán la imagen superpuesta y el video.

Nota: Para generar el video MP4 se necesita tener instalado FFmpeg.
Si no lo tiene, el código intentará crear un GIF (más pesado).
Si falla, solo se generará la gráfica superpuesta.

===========================================================
¿NECESITA AYUDA?
===========================================================

- Revise que las rutas de archivos sean correctas y que las carpetas existan.
- Asegúrese de tener instaladas las librerías: xarray, numpy, pandas, matplotlib,
  scipy, y (para el código 04) ffmpeg si desea MP4.
- Si algo falla, lea los mensajes de error en la consola; suelen ser claros.
- Los reportes de texto (.txt) contienen información útil para verificar
  que todo funcionó correctamente.

--- Fin del README ---