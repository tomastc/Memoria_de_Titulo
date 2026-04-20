===========================================================
       README - SIMULACIÓN NUMÉRICA DE LAGUNA COSTERA
       (Modelo Lagunar Simplificado)
===========================================================

Este conjunto de 5 códigos (todos en Python) permite construir un modelo
numérico simplificado de una laguna costera. El proceso incluye:

1. Generar una superficie batimétrica continua a partir de puntos de
   topografía/batimetría (mediante kriging).
2. Calcular la curva de volumen vs altura de la laguna.
3. Procesar datos de caudal y altura del estero afluente.
4. Simular la dinámica de la lagora usando un modelo de ancho de boca
   CONSTANTE (simulando solo en los intervalos donde hay datos reales
   de ancho, para comparación justa).
5. Simular la dinámica usando el ANCHO REAL (variable en el tiempo)
   proveniente de mediciones o del análisis de imágenes.

Los códigos están diseñados para ser ejecutados en orden (1 → 2 → 3 → 4/5),
aunque los modelos (4 y 5) pueden ejecutarse independientemente después de
tener los datos de entrada.

===========================================================
RESUMEN DE CADA CÓDIGO
===========================================================

CÓDIGO 01: Mallado batimétrico (kriging)
-----------------------------------------
- Qué hace: Toma un archivo CSV con puntos X, Y, Z (coordenadas y elevación)
  y genera una superficie batimétrica regular mediante interpolación por
  kriging. Solo interpola dentro del área donde hay datos (convex hull).
- Entrada: Archivo CSV con columnas de coordenadas X, Y y elevación Z.
  (El código detecta automáticamente los nombres de columna).
- Salida:
    * Archivo NetCDF "superficie_kriging.nc" con la malla de elevación.
    * Imagen 3D "superficie_3d.png" de la batimetría.
- Uso: Modifique la ruta del archivo CSV al inicio del código y ejecute.
  Ajuste el espaciado de la malla (variable "espaciado", por defecto 20 m).

CÓDIGO 02: Curva volumen-altura
--------------------------------
- Qué hace: Lee la superficie batimétrica generada por el código 01 y
  calcula, para cada altura (con un paso configurable), el volumen de agua
  que almacenaría la laguna (sumando celdas donde la altura supera la
  elevación del terreno). Genera la curva de volumen vs altura.
- Entrada: Archivo NetCDF "superficie_kriging.nc" (del código 01).
- Salida:
    * Archivo NetCDF "curva_volumen_altura.nc" con la curva.
    * Gráfico "curva_volumen.png".
- Uso: Ejecute después del código 01. Puede cambiar el paso de altura
  (variable "PASO_ALTURA", por defecto 0.1 m).

CÓDIGO 03: Procesamiento de caudal y altura del estero
-------------------------------------------------------
- Qué hace: Lee archivos Excel con datos de caudal y altura del estero
  (formato específico con tres grupos de columnas por fila). Procesa
  todos los archivos de una carpeta, combina los datos y genera una
  serie temporal única. Guarda los resultados en NetCDF y genera un
  gráfico de la serie completa.
- Entrada: Carpeta con archivos Excel (.xls, .xlsx) que contienen los
  datos de caudal y altura (formato específico explicado dentro del código).
- Salida:
    * Archivo NetCDF "caudales_alturas_completo.nc" con series de caudal
      (m³/s) y altura (m).
    * Imagen "serie_temporal_completa.png" con ambas series.
- Uso: Configure las rutas de entrada y salida en la función main().
  El código procesa automáticamente todos los archivos Excel de la carpeta.

CÓDIGO 04: Modelo lagunar con ancho CONSTANTE (simulación por intervalos)
-------------------------------------------------------------------------
- Qué hace: Implementa el modelo simplificado de laguna costera usando las
  ecuaciones gobernantes (continuidad y momentum). El ancho de la boca se
  mantiene constante (configurable). La simulación se ejecuta SOLO en los
  intervalos de tiempo donde existen datos reales de ancho (cargados desde
  un archivo NetCDF opcional), para permitir una comparación justa con el
  modelo de ancho variable. Si no se proporciona archivo de anchos, simula
  continuamente.
- Entradas requeridas:
    * Archivo NetCDF de caudal del estero (del código 03).
    * Archivo NetCDF de marea (reconstruida, por ejemplo desde el código
      de mareas MATLAB, con variable "marea").
    * Archivo NetCDF de batimetría (del código 01).
    * (Opcional) Archivo NetCDF de anchos reales (para determinar intervalos).
- Parámetros configurables (al inicio de main()):
    * PASO_TEMPORAL_MINUTOS (por defecto 0.25 min = 15 seg).
    * FECHA_INICIO y DURACION_DIAS.
    * PARAMETROS_MODELO: bc (ancho constante), hc (profundidad canal),
      Lc (longitud canal), n (Manning), ZL0 (nivel inicial laguna).
- Salidas:
    * Gráfico "Resultados_modelo_lagunar_constante_intervalos_FINAL_2024-07.png"
      con 6 paneles: niveles (marea y laguna), ancho constante, caudal Qc,
      área transversal Ac, caudal Qr, volumen.
    * NetCDF "resultados_simulacion_constante_intervalos_FINAL_2024-07.nc"
      con todas las variables simuladas.
    * Reporte TXT con estadísticas detalladas y tabla de resultados.
- Uso: Ejecute después de tener los archivos de entrada. Modifique rutas
  y parámetros según su caso.

CÓDIGO 05: Modelo lagunar con ancho VARIABLE (datos reales)
------------------------------------------------------------
- Qué hace: Igual que el código 04, pero el ancho de la boca se obtiene
  desde un archivo NetCDF de anchos reales (por ejemplo, del análisis de
  imágenes de satélite). Se aplican reglas de interpolación específicas
  (si los datos están muy separados se usa el más cercano; si ambos
  puntos están abiertos se interpola linealmente; si hay cambios de estado
  se maneja adecuadamente). La simulación se ejecuta solo en los intervalos
  donde existen datos de ancho (sin extender).
- Entradas: Las mismas que el código 04, más el archivo NetCDF de anchos
  reales (obligatorio para este modelo). El archivo debe contener las
  variables "time", "ancho_desembocadura" y "estado_apertura".
- Salidas:
    * Gráfico "Resultados_modelo_lagunar_FINAL_2024-07.png" con 6 paneles
      (el panel de ancho muestra el valor simulado y los puntos reales).
    * NetCDF "resultados_simulacion_variable_FINAL_2024-07.nc".
    * Reporte TXT detallado.
- Uso: Similar al código 04, pero con ARCHIVO_ANCHOS_NC apuntando a los
  datos reales. Ajuste los parámetros temporales y físicos según
  necesidad.

===========================================================
FLUJO DE TRABAJO RECOMENDADO
===========================================================

1. Prepare sus datos de topografía/batimetría (archivo CSV con X,Y,Z).
   Ejecute el código 01 para generar la superficie batimétrica.

2. Ejecute el código 02 para obtener la curva volumen-altura (necesaria
   para el modelo, aunque no se usa directamente en los códigos 04/05
   porque ellos calculan volumen desde la batimetría original).

3. Coloque todos los archivos Excel de caudal del estero en una carpeta.
   Ejecute el código 03 para generar el NetCDF de caudal y altura.

4. (Opcional) Si dispone de un archivo NetCDF con anchos reales
   (generado por el código 06 del análisis de imágenes), utilícelo.

5. Ejecute el código 04 (ancho constante) si desea una simulación de
   referencia con ancho fijo. Ejecute el código 05 (ancho variable)
   para la simulación realista.

6. Revise los gráficos y los reportes TXT para analizar los resultados.

===========================================================
NOTAS IMPORTANTES
===========================================================

- Todos los códigos están escritos en Python 3. Requieren las librerías:
  numpy, matplotlib, pandas, xarray, scipy, pykrige, openpyxl (para
  leer Excel). Instálelas con pip si es necesario.

- Los archivos NetCDF de marea deben contener una variable "marea"
  y coordenada "time" en segundos desde 1970-01-01. Si tiene otro
  formato, ajuste el código de carga de marea.

- El modelo lagunar resuelve las ecuaciones con Runge-Kutta de 4to orden.
  El paso temporal es pequeño (0.25 minutos) para estabilidad.

- Los códigos 04 y 05 generan reportes TXT muy detallados (pueden
  pesar varios MB si la simulación es larga). Se recomienda revisarlos
  con un editor de texto.

- Si no tiene datos reales de ancho, el código 04 puede simular
  continuamente con ancho constante (sin necesidad de archivo de
  anchos). El código 05 requiere el archivo de anchos para funcionar.

===========================================================
SOLUCIÓN DE PROBLEMAS COMUNES
===========================================================

- "No se encuentra el archivo de batimetría": verifique la ruta en el
  código 02 y en los modelos. Use rutas absolutas.

- Error en la carga de la marea: asegúrese de que el NetCDF tenga la
  variable "marea". Si la variable se llama diferente, modifique el
  código 04/05 en la función cargar_datos_marea().

- La simulación es muy lenta: reduzca la duración (DURACION_DIAS) o
  aumente el paso temporal (PASO_TEMPORAL_MINUTOS). Sin embargo, pasos
  mayores a 1 minuto pueden causar inestabilidad.

- El gráfico no muestra datos: verifique que el rango de fechas de la
  simulación esté dentro del rango de los datos de caudal, marea y
  ancho (si se usa). El código hace una verificación al inicio.

===========================================================
¿NECESITA AYUDA?
===========================================================

Revise los mensajes impresos en la consola; indican el progreso y
posibles errores. Los reportes TXT contienen estadísticas que ayudan
a validar la simulación. Si el modelo no converge, pruebe reducir el
paso temporal o ajustar los parámetros físicos (Manning, profundidad
del canal, etc.).

--- Fin del README ---