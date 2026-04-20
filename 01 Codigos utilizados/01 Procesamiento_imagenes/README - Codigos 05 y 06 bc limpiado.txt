===========================================================
       README - Análisis de Apertura y Ancho del Canal
===========================================================

Este documento explica los códigos 05 y 06, que continúan el análisis
después de haber obtenido los perímetros (códigos 02 y 03).

===========================================================
CÓDIGO 05: DETERMINAR SI EL CANAL ESTÁ ABIERTO O CERRADO
===========================================================

¿Qué hace?
----------
A partir de los puntos del perímetro (borde entre agua y arena), este
código identifica si el canal tiene una desembocadura activa. Para eso:

1. Encuentra los puntos del perímetro que están cerca del borde de la
   zona de interés (ROI).
2. Agrupa esos puntos en zonas (mediante un algoritmo de agrupamiento).
3. Desde cada zona, sigue el perímetro en ambas direcciones.
4. Verifica si dos seguimientos (desde zonas opuestas) se encuentran,
   si vienen de zonas distintas, y si ambos cruzan una
   línea vertical de referencia (X = 470 píxeles).
5. Si se cumplen las tres condiciones, se considera que ese par de
   seguimientos forma un "límite real" del canal.
6. Si hay al menos dos límites reales en un mismo instante, el canal
   se clasifica como ABIERTO. En caso contrario, CERRADO.

El código guarda, para cada instante (cada imagen):
   - El estado (1 = abierto, 0 = cerrado).
   - La cantidad de posibles límites encontrados.
   - La cantidad de límites reales (los que cumplen las 3 condiciones).
   - Las coordenadas de los dos límites más largos (superior e inferior).

Además, genera un reporte de texto detallado con:
   - Estadísticas mensuales y anuales.
   - Listado COMPLETO de cada instante (fecha, estado, número de límites).
   - Casos especiales (contradicciones, instantes sin perímetro, etc.).

Entradas
-----------------------
- El archivo NetCDF generado por el código 03 (perímetros), normalmente
  llamado "Data_perimetros_arena_agua.nc".
- La definición de la ROI (los mismos puntos que usó desde el principio).
- Parámetros configurables (al inicio del código):
   * X_LIMIT = 470           (línea vertical que debe cruzar el canal)
   * UMBRAL_DISTANCIA = 10   (máxima distancia para conectar puntos)
   * TOLERANCIA_ROI = 5      (qué tan cerca de la ROI se considera "cerca")

Salidas
---------------------
- Un archivo NetCDF (ej: "Data_Aperturas_COMPLETO_20250321_123456.nc") con:
   * estado_apertura (para cada tiempo, 1=abierto, 0=cerrado)
   * cantidad_posibles_limites
   * cantidad_limites_reales
   * limite_superior_canal (máscara con los puntos del límite superior)
   * limite_inferior_canal (máscara con los puntos del límite inferior)
   * coordenadas de todos los puntos del perímetro (originales)
- Un reporte de texto (ej: "reporte_completo_aperturas_20250321_123456.txt")
  con análisis completo, incluyendo TODOS los instantes y estadísticas
  mensuales/anuales.

Cómo usarlo
-----------
1. Asegúrese de haber ejecutado antes los códigos 01, 02 y 03.
2. Abra el código 05.
3. Revise los parámetros al inicio (X_LIMIT, UMBRAL_DISTANCIA, TOLERANCIA_ROI).
   Los valores por defecto suelen funcionar bien.
4. Verifique que las rutas de archivos en la función main() sean correctas:
   - archivo_perimetros: apunte al .nc generado por el código 03.
   - carpeta_resultados: donde guardar los resultados.
5. Ejecute el código completo.
6. Espere a que termine (puede tomar varios minutos dependiendo de cuántas
   imágenes tenga). Verá mensajes de progreso cada 50 instantes.
7. Revise la carpeta de salida: encontrará el archivo NetCDF y el reporte
   de texto. Abra el reporte para ver el detalle de cada instante.

Nota: Si el reporte muestra "¡ATENCIÓN!" o "¡CONTRADICCIÓN!", revise esos
instantes manualmente para entender por qué el algoritmo tuvo dudas.

===========================================================
CÓDIGO 06: CALCULAR EL ANCHO DE LA DESEMBOCADURA
===========================================================

¿Qué hace?
----------
Una vez que el código 05 identificó los límites superior e inferior del
canal (solo cuando está abierto), este código calcula la distancia mínima
entre esos dos límites. Esa distancia es el ancho de la desembocadura.

Además, aplica una limpieza automática para eliminar valores anómalos
(outliers):
   - Si un instante abierto tiene un ancho mayor a 40 metros y los dos
     instantes vecinos (anterior y siguiente) están cerrados, se cambia
     a cerrado (ancho = 0).
   - Para los instantes que siguen abiertos, si el ancho se desvía más
     de 40 metros del promedio de sí mismo y sus vecinos, se reemplaza
     por el promedio de los vecinos (interpolación).

El código genera:
   - Gráficos individuales para cada día que usted elija (muestra todos
     los instantes de ese día, con los límites y la línea del ancho).
   - Un gráfico temporal con la evolución de todos los anchos a lo largo
     del tiempo.
   - Un reporte estadístico completo (promedios, máximos, mínimos por mes,
     etc.).
   - Un archivo CSV con todos los datos (fecha, estado, ancho, etc.).

Entradas
-----------------------
- El archivo NetCDF generado por el código 05 (aperturas).
- Parámetros configurables:
   * X_LIMIT = 470           (misma línea de referencia, para consistencia)
   * ESCALA_DISTANCIAS = 0.6 (cuántos metros equivale un píxel)
   * DIA_SELECCIONADO = '2024-07-10' (fecha desde donde empezar a graficar)
   * N_DAYS = 7              (número de días consecutivos a graficar)

Salidas
---------------------
- Un archivo NetCDF con los mismos datos del código 05 más una nueva
  variable: "ancho_desembocadura" (en metros).
- Una versión "limpia" del mismo archivo (con "_limpio.nc") después de
  aplicar la corrección de outliers.
- Una carpeta "Resultados anchos medidos x día" con imágenes PNG para
  cada día del período seleccionado. Cada imagen muestra todos los
  instantes de ese día, con:
   * Fondo de color verde claro si el canal está abierto, rojo claro si
     está cerrado.
   * Puntos del perímetro (gris), límite superior (azul), límite inferior
     (rojo).
   * Una línea amarilla que marca la distancia mínima (ancho) y su valor.
- Un gráfico temporal (evolucion_temporal_todos_anchos.png) con todos los
  anchos a lo largo de toda la serie de tiempo.
- Un reporte de texto con estadísticas detalladas (promedios mensuales,
  distribución de anchos, etc.).
- Un archivo CSV con los datos completos (fecha, estado, ancho, etc.).

Cómo usarlo
-----------
1. Ejecute primero el código 05 para obtener el archivo de aperturas.
2. Abra el código 06.
3. Modifique los parámetros al inicio (especialmente DIA_SELECCIONADO y
   N_DAYS para elegir qué días visualizar).
4. Verifique las rutas:
   - archivo_aperturas: debe apuntar al archivo .nc generado por el código 05.
   - carpeta_resultados: donde guardar los resultados.
5. Ejecute el código.
6. Revise la carpeta de salida: encontrará los NetCDF, las imágenes diarias,
   el gráfico temporal, el reporte y el CSV.

Nota: La escala de distancia (ESCALA_DISTANCIAS) debe ser la correcta para
sus imágenes. Si sus fotos tienen una resolución de 0.6 metros por píxel,
déjelo así. Si es diferente, ajústelo.


===========================================================
¿NECESITA AYUDA?
===========================================================

- Revise que las rutas de archivos sean correctas y que las carpetas existan.
- Si el código 05 tarda mucho, es normal: procesa todos los instantes uno
  por uno. Puede reducir el tamaño de la ROI o aumentar el espaciado de
  grilla en el código 02 para acelerar.
- Los reportes de texto contienen información muy útil para diagnosticar
  problemas. Busque palabras como "ERROR", "ATENCIÓN" o "CONTRADICCIÓN".
- Si los gráficos del código 06 muestran valores extraños, verifique la
  escala de distancias y los umbrales de limpieza.

--- Fin del README ---