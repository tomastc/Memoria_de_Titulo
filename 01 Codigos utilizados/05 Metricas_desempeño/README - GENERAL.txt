===========================================================
       README - Evaluación de Desempeño del Modelo Lagunar
       (Comparación entre nivel medido y simulado)
===========================================================

Este documento explica dos programas que permiten comparar los niveles
de agua medidos en la laguna (datos reales) con los niveles simulados
por el modelo numérico. El proceso consta de dos pasos:

1. Convertir los datos medidos a la misma referencia vertical que usa
   el modelo (Nivel Medio del Mar, NMM).
2. Calcular métricas de error (MAE, RMSE, sesgo) y generar gráficos
   comparativos para cada simulación.

===========================================================
CÓDIGO 01: Cambio de referencia vertical de los datos medidos
===========================================================

¿Qué hace?
----------
Los datos de nivel de laguna medidos en terreno vienen referidos al
Nivel de Referencia del Sensor (NRS). El modelo lagunar utiliza el
Nivel Medio del Mar (NMM). Este programa resta un valor constante
(NRS) a las mediciones para obtener el nivel referido a NMM.

Además, combina los datos de dos años (2023 y 2024) en un solo archivo,
filtra por un rango de fechas definido por el usuario, y genera
gráficos para visualizar la conversión.

Entradas
-----------------------
- Dos archivos CSV con los datos medidos:
    * 2023-Cahuil-Barrancas_Nivel_Salinidad.csv
    * 2024-Cahuil-LaBalsa_Nivel_Salinidad.csv
  Cada archivo debe tener una columna 'Date UTC' (fecha y hora) y una
  columna de nivel (con nombres específicos, configurados en el código).
- El valor de NRS (constante) que se resta. Por defecto: 0.9257 m. DESDE ANÁLISIS ARMÓNICO
- Rango de fechas a procesar (inicio y fin).
- (Opcional) Una ventana de tiempo adicional para un segundo gráfico.

Salidas
---------------------
- Un archivo NetCDF: "Nivel_laguna_ref_NMM.nc" que contiene:
    * tiempo (en segundos desde 1970-01-01)
    * nivel_nrs (original, en m)
    * nivel_nmm (convertido, en m)
- Dos gráficos en formato PNG:
    * comparacion_niveles_completo.png (todo el período)
    * comparacion_niveles_ventana_30dias.png (ventana configurable)

Cómo usarlo
-----------
1. Abra el código 01 en un editor (Spyder, VS Code, etc.).
2. Modifique las rutas al principio del archivo:
   - ARCHIVO_CSV_2023 y ARCHIVO_CSV_2024.
   - CARPETA_SALIDA (donde se guardarán los resultados).
3. Ajuste los parámetros si es necesario:
   - NRS (valor fijo, obtenido del análisis armónico).
   - INICIO y FIN (formato "dd-mm-aa HH:MM").
   - INICIO_VENTANA y DIAS_VENTANA (para el segundo gráfico).
4. Ejecute el código.
5. Revise la carpeta de salida: encontrará el archivo NetCDF y los
   gráficos. Este NetCDF será la entrada para el código 02.

Nota: Si sus archivos CSV tienen otros nombres de columnas, ajuste
las variables COLUMNA_NIVEL_2023 y COLUMNA_NIVEL_2024.

===========================================================
CÓDIGO 02: Cálculo de métricas de error y gráficos comparativos
===========================================================

¿Qué hace?
----------
Toma el nivel medido (ya convertido a NMM) y lo compara con los niveles
simulados por diferentes versiones del modelo lagunar. Para cada
simulación, calcula tres métricas de error:

- MAE (Error Absoluto Medio): promedio de las diferencias absolutas.
- RMSE (Raíz del Error Cuadrático Medio): sensible a errores grandes.
- Sesgo (Bias): error sistemático (positivo = modelo sobreestima).

También genera gráficos que muestran, en un mismo eje:
   * Nivel medido (azul)
   * Nivel simulado (rojo)
   * Marea reconstruida (verde punteada, para contexto)
   * (Opcional) Ancho real de la desembocadura (puntos amarillos/rojos
     en un eje secundario).

Los gráficos se generan para todo el período común y también en
intervalos de 4 días para facilitar el análisis detallado.

Además, puede calcular las métricas sobre dos mallas temporales:
   * Sobre los propios tiempos de la simulación (interpolando mediciones).
   * Sobre los tiempos del archivo de referencia (que corresponden a los
     intervalos donde hay datos de ancho real), para una comparación más
     justa con el modelo de ancho variable.

Entradas
-----------------------
- El archivo NetCDF generado por el código 01 (nivel medido en NMM).
- Una carpeta que contenga los archivos NetCDF de las simulaciones
  (por ejemplo, los generados por los códigos 04 y 05 del modelo lagunar).
- El archivo NetCDF de marea reconstruida (marea_reconstruida_utide_2023-2024.nc).
- (Opcional) El archivo NetCDF de ancho real (Data_Ancho_Desde_Limites_limpio.nc)
  para agregar los puntos de ancho en los gráficos.
- (Opcional) Un archivo de simulación de referencia cuyos tiempos se
  usarán como malla para calcular métricas adicionales (por defecto,
  "resultados_simulacion_variable_FINAL_2024-07.nc").

Salidas
---------------------
- Para cada archivo de simulación, una subcarpeta con:
    * Un gráfico completo (nombre_archivo_completo.png)
    * Varios gráficos de intervalos de 4 días (intervalo_*.png)
- Un archivo de texto global "metricas_error_todas_simulaciones.txt"
  que resume, para cada simulación:
    * Período común analizado.
    * Métricas sobre la malla propia (n_puntos, MAE, RMSE, sesgo).
    * Métricas sobre la malla de referencia (si se proporcionó).

Cómo usarlo
-----------
1. Asegúrese de haber ejecutado el código 01 para generar el NetCDF
   de nivel medido.
2. Coloque todos los archivos NetCDF de simulaciones en una misma carpeta
   (por ejemplo, "Resultados Laguna Simulados").
3. Abra el código 02 y modifique las rutas al inicio:
   - archivo_medido: ruta al NetCDF generado por el código 01.
   - carpeta_simulaciones: carpeta donde están los archivos .nc de simulación.
   - archivo_marea: ruta al NetCDF de marea reconstruida.
   - RUTA_ANCHOS_NETCDF: ruta al archivo de ancho real (opcional, pero
     recomendado para los gráficos).
   - archivo_referencia_tiempos_nombre: nombre del archivo de simulación
     cuyos tiempos se usarán como referencia (opcional).
   - carpeta_salida_base: carpeta donde se guardarán todos los resultados.
4. Revise el mapeo de nombres de simulación (mapeo_simulaciones) para que
   las etiquetas en los gráficos sean claras.
5. Ejecute el código. El proceso puede tomar varios minutos si hay muchas
   simulaciones y largos períodos de tiempo.
6. Revise la carpeta de salida: encontrará una subcarpeta por simulación
   con los gráficos, y el archivo de texto con las métricas.

Notas importantes
-----------------
- Las métricas se calculan después de interpolar linealmente las mediciones
  a los tiempos de la simulación (o de la referencia). Esto asegura que se
  comparen en los mismos instantes.
- Si una simulación es de ancho constante, el programa automáticamente no
  incluirá los puntos de ancho real en los gráficos (para no saturar).
- El archivo de texto final incluye una nota especial para la simulación
  continua (Simulación 1a) indicando que debe considerarse la métrica sobre
  la malla de referencia (intervalos de ancho) para una comparación justa.

===========================================================
FLUJO DE TRABAJO COMPLETO
===========================================================

1. Mida los parámetros del canal (Lc con el código de medición de imágenes,
   hc con el código de análisis de marea y ancho).
2. Ejecute las simulaciones del modelo lagunar (códigos 04 y 05).
3. Convierta los datos medidos de nivel de laguna a NMM (código 01).
4. Compare las simulaciones con los datos reales (código 02) y revise
   las métricas para determinar qué versión del modelo se ajusta mejor.

===========================================================
¿NECESITA AYUDA?
===========================================================

- Si el código 01 no encuentra las columnas en los CSV, verifique los
  nombres exactos y ajústelos en las variables COLUMNA_NIVEL_2023 y
  COLUMNA_NIVEL_2024.
- Si el código 02 no encuentra alguna simulación, asegúrese de que los
  archivos .nc tengan las variables esperadas ('nivel_laguna' o similares).
- Las métricas se expresan en metros. Valores de MAE inferiores a 0.1 m
  indican un muy buen ajuste; entre 0.1 y 0.3 m es aceptable; mayores
  sugieren revisar parámetros o datos de entrada.
- Revise los gráficos de intervalos de 4 días para identificar en qué
  períodos el modelo se comporta mejor o peor.

--- Fin del README ---