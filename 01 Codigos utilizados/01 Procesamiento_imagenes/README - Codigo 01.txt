===========================================================
       README - Segmentación de Imágenes
===========================================================

¿Qué hace este programa?
------------------------
Este programa analiza fotografías geo-rectificadas de la desembocadura 
y separa automáticamente el agua de la arena. El resultado son 
máscaras (mapas en blanco y negro) que indican dónde hay agua y 
dónde hay arena en cada imagen.

Además, el programa conserva la fecha y hora original de cada foto 
(sacada del nombre del archivo) y genera un único archivo de salida 
con todos los resultados.


Entradas
------------------------------------
1. Una carpeta que contenga imágenes en formato TIFF (.tiff). 
   El nombre de cada archivo debe incluir la fecha y hora así:
   AAAAMMDD-HHMM (ejemplo: 20250321-1430.tiff)

2. Un polígono que defina la "zona de interés" (ROI). 
   Solo se analizará el interior de ese polígono.

3. Dos puntos de referencia dentro de la imagen:
   - Un punto que claramente sea agua (mar)
   - Un punto que claramente sea arena (playa)


Salidas
---------------------------------
Al finalizar, el programa crea una carpeta con:

• Un archivo NetCDF (.nc) que contiene:
   - Las imágenes originales con anotaciones (ROI y puntos marcados)
   - Las máscaras de agua (blanco=agua, negro=no agua)
   - Las máscaras de arena (blanco=arena, negro=no arena)
   - La fecha y hora corregida de cada imagen
   - Un resumen con estadísticas básicas

• Un reporte de texto (reporte_completo_procesamiento.txt) con:
   - Cuántas imágenes se procesaron
   - Cuántas tenían agua válida
   - Cuántas fechas fueron corregidas
   - Detalle de cada imagen (fecha original, fecha final, etc.)

• Un archivo CSV (resumen_imagenes.csv) con la misma información 
  en formato de tabla.

¿Cómo se usa?
-------------
Paso 1: Preparar los datos
   - Coloque todas sus imágenes TIFF en una misma carpeta.
   - Asegúrese de que los nombres contengan la fecha/hora en formato 
     AAAAMMDD-HHMM (ej: 20250101-1200.tiff).

Paso 2: Definir el ROI y los puntos de referencia
   - Abra una imagen de ejemplo con cualquier visor.
   - Anote las coordenadas (x, y) de los vértices del polígono ROI.
   - Anote las coordenadas (x, y) del punto que representa agua.
   - Anote las coordenadas (x, y) del punto que representa arena.

Paso 3: Configurar el programa
   - Edite las últimas líneas del código (abajo del todo) y cambie:
        carpeta_raiz   = "ruta/a/su/carpeta/de/imagenes"
        carpeta_salida = "ruta/donde/guardar/resultados"
        roi1_points    = [(x1,y1), (x2,y2), ...]  # los vértices del polígono
        punto_mar1     = (x_agua, y_agua)
        punto_arena1   = (x_arena, y_arena)

Paso 4: Ejecutar
   - Ejecute el programa completo (en Python).
   - Espere a que termine. Verá mensajes de avance en pantalla.

Paso 5: Revisar los resultados
   - Abra el archivo NetCDF con Python (con xarray) para ver las máscaras 
     y las imágenes anotadas.
   - Lea el reporte de texto para entender cómo se trató cada imagen.

Notas importantes
-----------------
• El programa solo guarda las imágenes ANOTADAS (con el polígono y los 
  puntos dibujados). No guarda las originales sin anotaciones.
• Si una imagen está muy oscura (noche) o cubierta por neblina, el programa 
  detectará que no se puede segmentar y lo indicará en el reporte.
• Las fechas se conservan originales siempre que sea posible. Solo se 
  corrigen si el nombre tiene una fecha imposible o mal escrita (ej: mes 13)
  o si hay dos imágenes con la misma fecha (entonces se ajustan mínimamente).
• El archivo NetCDF está comprimido para que ocupe menos espacio.


Análisis de resultados
----------------------
Revise el reporte de texto generado; allí se explica cada corrección 
aplicada. Si algo no funciona, verifique que las rutas de las carpetas 
sean correctas y que las imágenes sean realmente TIFF.

--- Fin del README ---