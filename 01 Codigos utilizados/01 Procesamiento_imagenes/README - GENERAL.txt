================================================================
     README GENERAL - ANÁLISIS COMPLETO DE LÍNEA DE COSTA
                (Códigos 01 al 06)
================================================================

Este documento explica el proceso completo para analizar fotografías
de la desembocadura, desde la separación de agua y arena 
hasta la medición del ancho de la desembocadura de su canal.

El proceso consta de 6 pasos (cada uno es un código independiente). 
Al final, usted obtendrá:

- El ancho de la desembocadura en cada momento (en metros).
- Gráficos que muestran cómo cambia los bordes del canal día a día.
- Un video de la evolución del perímetro.
- Reportes detallados con estadísticas mensuales y anuales.

================================================================
¿QUÉ NECESITA AL INICIO?
================================================================

- Una carpeta con imágenes en formato TIFF (.tiff).
- Que el nombre de cada imagen contenga la fecha y hora así:
  AAAAMMDD-HHMM (ejemplo: 20250321-1430.tiff).
- Saber las coordenadas (en píxeles) de:
    * Un polígono que defina la zona de interés (ROI).
    * Un punto que sea claramente AGUA.
    * Un punto que sea claramente ARENA.

================================================================
RESUMEN DE ENTRADAS Y SALIDAS DEL PROCESO COMPLETO
================================================================

ENTRADAS:
-----------------------------------
1. Carpeta con imágenes TIFF (nombres con fecha/hora).
2. Coordenadas del polígono ROI (ej: [(439,100), (558,100), (516,600), (180,600)]).
3. Coordenadas de un punto de agua (ej: (280,500)).
4. Coordenadas de un punto de arena (ej: (470,500)).
5. (Opcional) Parámetros: espaciado de grilla, línea de referencia X_LIMIT,
   escala de metros por píxel, fechas para visualizaciones.

SALIDAS:
----------------------------
- Archivos NetCDF intermedios (máscaras, grilla, perímetros, estados, anchos).
- Imágenes PNG:
    * Ejemplo de grilla con máscaras (código 02).
    * Tres perímetros de ejemplo (código 03).
    * Gráficos diarios del ancho (código 06).
    * Gráfico temporal de todos los anchos (código 06).
- Video MP4/GIF de evolución del perímetro (código 04, opcional).
- Reportes de texto:
    * Del código 05: análisis completo de aperturas (todos los instantes).
    * Del código 06: estadísticas de anchos (promedios mensuales, etc.).
- Archivo CSV con todos los datos de anchos (fecha, estado, ancho).

================================================================
¿CÓMO EJECUTAR TODO EN ORDEN?
================================================================

Paso 1: Ejecute el Código 01
   - Modifique las rutas de carpeta_raiz y carpeta_salida.
   - Defina roi1_points, punto_mar1, punto_arena1.
   - Ejecute. Obtendrá "resultados_mascaras_ROI1.nc".

Paso 2: Ejecute el Código 02
   - Configure ARCHIVO_NETCDF_ENTRADA apuntando al .nc del paso 1.
   - Ajuste ESPACIADO_GRILLA (recomendado: 1).
   - Ejecute. Obtendrá "grilla_mascaras.nc".

Paso 3: Ejecute el Código 03
   - Configure archivo_grilla_entrada apuntando al .nc del paso 2.
   - Elija tres fechas en "dias_a_mostrar".
   - Ejecute. Obtendrá "Data_perimetros_arena_agua.nc".

Paso 4 (opcional): Ejecute el Código 04
   - Configure FECHA_INICIO y NUMERO_DIAS.
   - Ejecute. Obtendrá video y gráfica superpuesta.

Paso 5: Ejecute el Código 05
   - Verifique que archivo_perimetros apunte al .nc del paso 3.
   - Ejecute. Obtendrá "Data_Aperturas_COMPLETO_*.nc" y un reporte.

Paso 6: Ejecute el Código 06
   - El código buscará automáticamente el último archivo de aperturas.
   - Configure DIA_SELECCIONADO y N_DAYS para los gráficos diarios.
   - Ajuste ESCALA_DISTANCIAS (metros por píxel) según sus imágenes.
   - Ejecute. Obtendrá todos los resultados finales.

================================================================
CONSEJOS Y SOLUCIÓN DE PROBLEMAS
================================================================

- Los reportes de texto del código 05 contienen una lista COMPLETA de
  todos los instantes. Busque las palabras "¡ATENCIÓN!" o "¡CONTRADICCIÓN!"
  para identificar fechas donde el algoritmo tuvo dudas.

- Si los anchos calculados parecen incorrectos, verifique:
    * La segmentación mediante k-means en código 01: n° de iteraciones y 
      variables del post-procesamiento morfológico.
    * La escala ESCALA_DISTANCIAS en el código 06.
    * La línea X_LIMIT (debe cruzar el canal en todas las imágenes).
    * La tolerancia TOLERANCIA_ROI (si es muy pequeña, puede no encontrar
      puntos cerca del borde).

- Para el video del código 04, necesitará tener instalado FFmpeg en su
  computadora. Si no lo tiene, el código intentará crear un GIF (más
  pesado). Si falla, solo obtendrá la gráfica superpuesta.

================================================================
¿NECESITA AYUDA?
================================================================

- Revise que todos los archivos NetCDF se hayan generado correctamente
  antes de pasar al siguiente paso.
- Los mensajes en la consola indican el progreso. Si aparece un error,
  lea el mensaje completo: suele indicar qué archivo falta.
- Los reportes de texto son su mejor herramienta para entender los
  resultados y detectar posibles problemas.


--- Fin del README general ---