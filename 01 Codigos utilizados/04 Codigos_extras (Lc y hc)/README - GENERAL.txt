===========================================================
       README - Medición de Parámetros CONSTANTES del Canal
       (Largo Lc y Altura hc)
===========================================================

Estos dos programas le ayudan a medir dos parámetros del canal: 
el largo (Lc) y la altura o profundidad (hc). Ambos son
necesarios para el modelo de laguna costera.

===========================================================
CÓDIGO 01: Medir el largo del canal (Lc) desde imágenes
===========================================================

¿Qué hace?
----------
Este programa le permite abrir varias imágenes (mismas que del análisis de anchos)
y medir con el mouse la distancia entre dos puntos. Es ideal para medir
el largo de un canal (desde el mar hasta donde comienza la laguna).

Usted selecciona dos puntos en cada imagen (por ejemplo, inicio y fin del
canal) y el programa calcula la distancia en metros, considerando que cada
píxel equivale a 0.6 metros (600 mm). Puede medir varias imágenes y al
final obtiene el promedio y la desviación estándar.

Entradas
-----------------------
- Una carpeta con imágenes en formato .tif, .tiff, .png, .jpg o .jpeg.
- Saber la escala de la imagen (cuántos metros representa un píxel).
  El programa usa 0.6 m/píxel por defecto, pero puede cambiarlo.

Salidas
---------------------
- En pantalla: una ventana interactiva donde puede:
    * Hacer clic para marcar puntos (dos puntos por imagen).
    * Presionar el botón "Ingresar Medida" para guardar la distancia.
    * Presionar "Borrar último punto" si se equivocó.
    * Al final, presionar "Calcular Largo Promedio" para ver el resultado.
- Un archivo de texto (Resultados Largo Canal Lc.txt) con:
    * La distancia medida en cada imagen.
    * El promedio y la desviación estándar.

Cómo usarlo
-----------
1. Coloque todas las imágenes que desea medir en una carpeta.
2. Abra el código 01 en un editor (Spyder, VS Code, etc.).
3. Modifique las rutas al principio del archivo:
   - CARPETA_IMAGENES: ruta a su carpeta con imágenes.
   - ARCHIVO_SALIDA: ruta y nombre del archivo de resultados.
   - ESCALA_MM_POR_PIXEL: por defecto 600 (0.6 m/píxel). Si sus imágenes
     tienen otra resolución, cámbielo (ej: 1000 si 1 m/píxel).
4. Ejecute el código. Se abrirá una ventana con todas las imágenes en una
   cuadrícula. La imagen activa tiene el borde rojo.
5. Para cada imagen:
   - Haga clic izquierdo para marcar el primer punto.
   - Haga clic izquierdo nuevamente para marcar el segundo punto.
   - Verá una línea amarilla y la distancia en metros.
   - Presione el botón "Ingresar Medida" para guardar esa distancia.
   - El programa pasará automáticamente a la siguiente imagen.
6. Si se equivoca, presione "Borrar último punto" y vuelva a marcar.
7. Cuando haya medido todas las imágenes, presione "Calcular Largo Promedio".
   Aparecerán los resultados en la consola y se guardará el archivo de texto.
8. Cierre la ventana o presione la tecla 'q' para salir.

Nota: Si alguna imagen no tiene una medida clara, puede saltarla (no presione
"Ingresar Medida" y pase a la siguiente usando el botón o haciendo clic en
otra imagen; pero el programa avanzará automáticamente solo después de medir).

===========================================================
CÓDIGO 02: Calcular la profundidad del canal (hc) a partir de anchos y marea
===========================================================

¿Qué hace?
----------
Este programa calcula la altura (profundidad) del canal en cada mes usando
dos fuentes de datos:
  1. Los anchos de la desembocadura (medidos en otro proceso) y el estado
     de apertura (abierto/cerrado).
  2. La marea reconstruida en el punto de interés (Cáhuil).

La idea es sencilla: cuando el canal está abierto, la marea entra y sale;
la diferencia entre la marea más alta y la más baja dentro de cada período
de apertura es una estimación de la altura del canal (hc). El programa
identifica todos los períodos en que el canal estuvo abierto (según el
archivo de anchos), calcula la amplitud de la marea en cada período, y
luego promedia esas amplitudes por mes. El resultado es el valor de hc
para cada mes.

Entradas
-----------------------
- Un archivo NetCDF con los datos de ancho y estado de apertura del canal.
  Debe contener las variables: 'time', 'estado_apertura' (1=abierto, 0=cerrado)
  y opcionalmente 'ancho_desembocadura' (solo para graficar).
- Un archivo NetCDF con la marea reconstruida (por ejemplo, desde el análisis
  armónico con UTide). Debe contener 'time' y una variable de altura de marea
  (puede llamarse 'marea' o 'altura_cahuil_estimada').

Salidas
---------------------
- Un archivo de texto (analisis_hc_mensual_detallado.txt) con:
    * Lista de cada intervalo de apertura (fecha inicio, fin, duración,
      amplitud de marea).
    * Resumen mensual: promedio de hc, desviación, máximo, mínimo, etc.
- Dos gráficos:
    * Uno con toda la serie de tiempo, mostrando la marea y el ancho del
      canal, con sombreado verde en los períodos abiertos.
    * Un zoom de 7 días alrededor del intervalo con mayor amplitud, para
      ver el detalle.

Cómo usarlo
-----------
1. Asegúrese de tener los archivos NetCDF necesarios:
   - El de anchos (generado por el código 06 del análisis de imágenes).
   - El de marea (generado por el código 03 de mareas en MATLAB).
2. Abra el código 02 en un editor.
3. Modifique las rutas al principio del archivo:
   - RUTA_ANCHOS_NETCDF: ruta al archivo .nc de anchos.
   - RUTA_MAREA_NETCDF: ruta al archivo .nc de marea.
   - CARPETA_SALIDA: carpeta donde guardar los resultados.
   - NOMBRE_TXT, NOMBRE_FIG, NOMBRE_FIG_7DIAS: nombres de los archivos
     de salida (puede dejarlos como están).
4. Ejecute el código.
5. Revise la carpeta de salida: encontrará el archivo de texto y dos
   imágenes.
6. En el archivo de texto, busque la sección "RESUMEN MENSUAL". Allí
   aparece el valor de hc promedio para cada mes (en metros). Ese es
   el parámetro que necesita para el modelo lagunar.

Notas importantes
-----------------
- El programa solo considera intervalos donde el canal estuvo abierto
  (estado_apertura = 1). Si un intervalo tiene amplitud de marea cero
  (porque solo hay un punto), se ignora para el promedio mensual.
- Si un mes no tiene ningún intervalo abierto, no aparecerá en el resumen.
- La gráfica de 7 días ayuda a verificar visualmente que la amplitud
  calculada sea razonable.

===========================================================
¿Para qué sirven estos parámetros?
===========================================================

- Largo del canal (Lc): se usa en el modelo lagunar para calcular la
  fuerza gravitacional que mueve el agua a través del canal.
- Altura del canal (hc): representa la profundidad media del canal;
  influye en el área transversal y la fricción.

Una vez que tenga Lc (medido con el código 01) y hc (calculado con el
código 02), puede ingresarlos en el modelo lagunar (códigos 04 y 05
de la simulación numérica).

===========================================================
¿NECESITA AYUDA?
===========================================================

- Si las imágenes no se cargan, verifique que la carpeta existe y que
  los formatos son soportados.
- Para el código 02, asegúrese de que los archivos NetCDF tengan las
  variables correctas. Si la variable de marea tiene otro nombre,
  el programa la detecta automáticamente, pero puede ajustar el código
  si es necesario.

--- Fin del README ---