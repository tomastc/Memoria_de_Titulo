===========================================================
       README - PROCESAMIENTO DE MAREAS
       (Códigos Python y MATLAB)
===========================================================

Este conjunto de tres códigos procesa datos de mareas desde archivos
texto hasta un análisis armónico y reconstrucción de la marea en
un punto de interés (Cáhuil). Los dos primeros códigos están en Python,
el tercero en MATLAB.

===========================================================
DESCRIPCIÓN GENERAL DEL PROCESO
===========================================================

1. El código 01 (Python) lee archivos de marea de dos estaciones
   (San Antonio y Boyeruca) que están organizados en subcarpetas por
   rango de fechas. Combina todos los archivos, limpia los datos y
   guarda las series temporales en formato NetCDF. También genera
   gráficos de las series completas.

2. El código 02 (Python) utiliza los NetCDF generados, más un archivo
   con coordenadas geográficas de las estaciones, para:
   - Calcular distancias entre estaciones.
   - Sincronizar las series temporales a intervalos regulares de 10
     minutos (00:00, 00:10, 00:20, etc.).
   - Calcular promedios móviles de 7 días.
   - Obtener las mareas "relativas" (bruto - promedio móvil).
   - Estimar la marea en Cáhuil mediante ponderación inversa por
     distancia de las dos estaciones.
   - Guardar los resultados (brutos, promedios móviles y relativos)
     en un nuevo archivo NetCDF y generar gráficos.

3. El código 03 (MATLAB) toma la serie estimada de Cáhuil (del código 02)
   y realiza un análisis armónico de mareas usando la librería UTide.
   Calcula los coeficientes de las constituyentes, los guarda en un
   archivo CSV, y permite reconstruir la marea para un período futuro
   (configurable: fecha de inicio y número de días). Genera gráficos
   de verificación del ajuste y de la reconstrucción comparada con
   observaciones.

===========================================================
ENTRADAS
===========================================================

CÓDIGO 01 (Python)
------------------
- Una carpeta principal (Data_mareas) que contenga subcarpetas.
  Cada subcarpeta debe tener nombre con el formato "AAAA-MM-DD hasta AAAA-MM-DD"
  (ej: "2023-01-01 hasta 2023-03-31").
- Dentro de cada subcarpeta, dos archivos:
    * San_Antonio.txt
    * Boyeruca.txt
  Cada archivo es de texto con columnas separadas por tabulaciones:
    datetime, prs, rad
  donde "prs" y "rad" son alturas de marea (metros) de dos sensores.
- El código genera automáticamente los archivos de salida en la misma
  carpeta raíz.

CÓDIGO 02 (Python)
------------------
- Los archivos NetCDF generados por el código 01:
    Serie_bruta_Zm_San_Antonio.nc
    Serie_bruta_Zm_Boyeruca.nc
- Un archivo de texto con coordenadas de estaciones (Coordenadas_estaciones_IOC.txt)
  con formato: nombre; latitud; longitud (por ejemplo: "Cahuil;-34.479914;-72.021111").
- Parámetros internos (ya configurados, pero puede modificar las rutas).

CÓDIGO 03 (MATLAB)
------------------
- El archivo NetCDF generado por el código 02 que contiene la serie
  estimada de Cáhuil (por defecto: "Resultado_mareas_estimadas_10min.nc").
- La librería UTide (no incluida, debe descargarla por separado
  y agregarla a la ruta de MATLAB.
- Parámetros configurables dentro del código: fecha de inicio de la
  reconstrucción y duración en días.

===========================================================
SALIDAS 
===========================================================

CÓDIGO 01
---------
- Serie_bruta_Zm_San_Antonio.nc (NetCDF con la serie de San Antonio)
- Serie_bruta_Zm_Boyeruca.nc (NetCDF con la serie de Boyeruca)
- series_temporales_completas.png (gráfico de ambas series)

CÓDIGO 02
---------
- Resultado_mareas_estimadas_10min.nc (NetCDF con las series:
    * boyeruca_bruto, san_antonio_bruto
    * promedio_movil_boyeruca, promedio_movil_san_antonio
    * boyeruca_relativa, san_antonio_relativa
    * cahuil_estimada)
- Grafico_resultados_mareas_10min.png (gráfico con dos paneles:
    arriba: brutos y promedios móviles
    abajo: mareas relativas y estimación en Cáhuil)

CÓDIGO 03
---------
- coeficientes_utide_para_python.csv (archivo con los coeficientes
  armónicos: nombre de constituyente, frecuencia (ciclos/día),
  amplitud (m), fase (grados) y nivel medio)
- marea_reconstruida_utide_2023-2024.nc (NetCDF con la marea reconstruida
  para el período futuro elegido, paso de 5 minutos)
- Carpeta "figuras_2023-2024" (se crea automáticamente) que contiene:
    * verificacion_ajuste.png (comparación entre datos observados
      y ajuste armónico en el período de calibración)
    * marea_reconstruida_comparada.png (comparación entre la
      reconstrución futura y los datos observados en ese mismo período,
      si existen)

===========================================================
CÓMO USARLOS (paso a paso)
===========================================================

Paso 1: Preparar los datos brutos
----------------------------------
- Cree una carpeta llamada "Data_mareas".
- Dentro, cree subcarpetas con nombres como "2023-01-01 hasta 2023-03-31".
- En cada subcarpeta, coloque los archivos "San_Antonio.txt" y "Boyeruca.txt"
  con el formato indicado.

Paso 2: Ejecutar el código 01 (Python)
---------------------------------------
- Abra el código 01 en un editor (Spyder, VS Code, etc.).
- Verifique que las rutas de las variables "data_folder", "output_san_antonio"
  y "output_boyeruca" apunten a las ubicaciones correctas.
- Ejecute el código. Se generarán los archivos NetCDF y el gráfico.

Paso 3: Ejecutar el código 02 (Python)
---------------------------------------
- Asegúrese de tener los archivos NetCDF del paso anterior en las rutas
  especificadas en el código 02 (variables RUTA_BOYERUCA, RUTA_SAN_ANTONIO).
- Verifique que el archivo "Coordenadas_estaciones_IOC.txt" exista en la
  ruta indicada (RUTA_COORDENADAS).
- Ejecute el código 02. El proceso puede tardar varios minutos (dependiendo
  de la cantidad de datos). Mostrará avance cada 10,000 intervalos.
- Al finalizar, obtendrá el NetCDF de resultados y el gráfico.

Paso 4: Instalar UTide en MATLAB
--------------------------------
- Descargue UTide.
- Descomprima la carpeta y agréguela a la ruta de MATLAB.

Paso 5: Ejecutar el código 03 (MATLAB)
---------------------------------------
- Abra el código 03 en MATLAB.
- Modifique las rutas de archivo si es necesario (archivo_nc,
  archivo_salida_txt, etc.).
- Configure la fecha de inicio (fecha_inicio_str) y la duración en días
  (num_dias) para la reconstrucción futura. Por defecto está puesto
  "2023-01-01 00:00:00" y 730 días (2 años).
- Ejecute el código. MATLAB realizará el análisis armónico, guardará los
  coeficientes en CSV, reconstruirá la marea futura, y generará las figuras
  en la carpeta "figuras_2023-2024".

===========================================================
NOTAS IMPORTANTES
===========================================================

- Los códigos 01 y 02 requieren Python con las librerías: pandas, xarray,
  matplotlib, numpy, scipy (para el cálculo de distancias). Instálelas
  con pip si es necesario.
- El código 03 requiere MATLAB con la toolbox de manejo de NetCDF (generalmente
  incluida) y la librería UTide (gratuita). No requiere toolboxes adicionales
  porque se usa OLS y White para evitar dependencias.
- El paso de tiempo de los resultados del código 02 es de 10 minutos. El
  código 03 reconstruye con paso de 5 minutos para mayor resolución.
- Los promedios móviles de 7 días se calculan como el promedio de 3 días
  antes, el día central y 3 días después, con el objetivo de aislar la
  marea astronómica de la meteorológica.

===========================================================
SOLUCIÓN DE PROBLEMAS FRECUENTES
===========================================================

- "No se encontraron archivos de marea": verifique que las subcarpetas
  tengan exactamente los nombres "San_Antonio.txt" y "Boyeruca.txt".
- Error en la conversión de tiempo en MATLAB: asegúrese de que el atributo
  "units" de la variable "time" en el NetCDF tenga el formato esperado
  ("minutes since YYYY-MM-DD HH:MM:SS").
- UTide no se encuentra: agregue la carpeta de UTide a la ruta de MATLAB
  usando "addpath(genpath('ruta/a/UTide'))".
- Si la reconstrucción futura no muestra datos observados para comparar,
  es normal si el período elegido está fuera del rango de los datos
  originales. En ese caso, el gráfico mostrará solo la reconstrucción.

===========================================================
¿NECESITA AYUDA?
===========================================================

Revise los mensajes que aparecen en la consola de Python o en la
ventana de comandos de MATLAB; suelen indicar claramente el error.
Asegúrese de seguir el orden de ejecución (primero código 01, luego 02,
luego 03). Si algo falla, verifique que las rutas de archivos sean
absolutas y que las carpetas existan.

--- Fin del README ---