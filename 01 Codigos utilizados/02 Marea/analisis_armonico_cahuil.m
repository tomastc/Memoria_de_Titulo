%% analisis_armonico_cahuil.m
% Carga datos de marea desde NetCDF, realiza análisis armónico con UTide
% (usando OLS y White para evitar toolboxes) y guarda coeficientes en CSV.
% Además, permite reconstruir la marea para un período futuro y guardarla en NetCDF,
% generar un gráfico de la reconstrucción y almacenar todas las figuras.

clear; close all; clc;

%% --- CONFIGURACIÓN INICIAL ---
archivo_nc = 'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\02 Mareas\Resultados_mareas\Resultado_mareas_estimadas_10min.nc';
nombre_var_tiempo = 'time';
nombre_var_marea = 'altura_cahuil_estimada';
latitud = -34.479914;

archivo_salida_txt = 'coeficientes_utide_para_python.csv';

% Carpeta para guardar figuras
carpeta_figuras = 'figuras_2023-2024';
if ~exist(carpeta_figuras, 'dir')
    mkdir(carpeta_figuras);
    fprintf('Carpeta creada: %s\n', carpeta_figuras);
end

%% --- 1. LEER DATOS Y ATRIBUTOS ---
fprintf('Cargando datos desde %s...\n', archivo_nc);

tiempo_raw = ncread(archivo_nc, nombre_var_tiempo);
marea_raw = ncread(archivo_nc, nombre_var_marea);
fprintf('  - %d registros cargados.\n', length(marea_raw));

% Leer atributo 'units'
info_time = ncinfo(archivo_nc, nombre_var_tiempo);
units_attr = '';
for k = 1:length(info_time.Attributes)
    if strcmp(info_time.Attributes(k).Name, 'units')
        units_attr = info_time.Attributes(k).Value;
        break;
    end
end
if isempty(units_attr)
    error('No se encontró el atributo "units" para la variable time.');
end
fprintf('  - Atributo units: %s\n', units_attr);

%% --- 2. CONVERTIR TIEMPO A DATENUM ---
% Extraer fecha base con expresión regular
tokens = regexp(units_attr, 'minutes since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', 'tokens');
if isempty(tokens)
    error('No se pudo extraer la fecha base del atributo units.');
end
fecha_base_str = tokens{1}{1};
fecha_base_datenum = datenum(fecha_base_str, 'yyyy-mm-dd HH:MM:SS');
fprintf('  - Fecha base: %s (datenum = %0.10f)\n', fecha_base_str, fecha_base_datenum);

% Convertir minutos a días y sumar a la fecha base
minutos = double(tiempo_raw);
dias_desde_base = minutos / (24*60);
tiempo_datenum = fecha_base_datenum + dias_desde_base;

% Mostrar primeros valores con alta precisión
fprintf('  - Primeros 5 tiempos convertidos (con 15 decimales):\n');
for i = 1:min(5, length(tiempo_datenum))
    fprintf('      %0.15f\n', tiempo_datenum(i));
end

%% --- 3. LIMPIAR, ORDENAR Y ELIMINAR DUPLICADOS ---
% Eliminar NaNs
idx_validos = ~isnan(marea_raw);
tiempo_datenum = tiempo_datenum(idx_validos);
marea_raw = marea_raw(idx_validos);
fprintf('  - %d registros válidos tras eliminar NaNs.\n', length(marea_raw));

% Ordenar
[tiempo_datenum, idx_orden] = sort(tiempo_datenum);
marea_raw = marea_raw(idx_orden);

% Eliminar duplicados muy cercanos (tolerancia 1e-8 días)
tol = 1e-8;
duplicados = abs(diff(tiempo_datenum)) < tol;
if any(duplicados)
    fprintf('  - Se encontraron %d pares de tiempos casi duplicados (tol=%g días). Eliminando...\n', sum(duplicados), tol);
    idx_keep = [true; ~duplicados];
    tiempo_datenum = tiempo_datenum(idx_keep);
    marea_raw = marea_raw(idx_keep);
end

% Verificar monotonicidad
if any(diff(tiempo_datenum) <= 0)
    error('Los tiempos no son estrictamente crecientes incluso después de eliminar duplicados.');
else
    fprintf('  - Tiempos verificados: monótonos crecientes (dif mín = %g días, dif máx = %g días).\n', ...
        min(diff(tiempo_datenum)), max(diff(tiempo_datenum)));
end

% Asegurar tipo double
tiempo_datenum = double(tiempo_datenum);
marea_raw = double(marea_raw);

%% --- 4. ANÁLISIS ARMÓNICO CON UTide (usando OLS y White) ---
fprintf('Ejecutando UTide (ut_solv) con opciones OLS y White...\n');

% Usamos 'OLS' para evitar robustfit (Statistics Toolbox)
% Usamos 'White' para evitar cálculo espectral con ventanas (Signal Processing Toolbox)
coef = ut_solv(tiempo_datenum, marea_raw, [], latitud, 'auto', 'NoTrend', 'OLS', 'White');

fprintf('Análisis completado. %d constituyentes resueltos.\n', length(coef.name));
%% Det NRS
O1=coef.A(4);                                        %com lunar declinacion diurna
K1=coef.A(2);                                         %com lunisolar declinacion diurna
M2=coef.A(1);                                         %com lunar principal semidiurna
S2=coef.A(3);                                         %com solar principal semidiurna
N2=coef.A(5);                                        %com lunar eliptica mayor semidiurna
nivel_medio = coef.mean;
% NRS SHOA
NRS=nivel_medio-(O1+K1+M2+S2+N2);                                %fuente: publicaciones del SHOA
fprintf('NRS:  %s...\n', NRS);
%% --- 5. GUARDAR COEFICIENTES PARA PYTHON ---
fprintf('Guardando coeficientes en %s...\n', archivo_salida_txt);

nombres = coef.name(:);
frecuencias_cpd = coef.aux.frq(:) * 24;  % cph → cpd
amplitudes_m = coef.A(:);
fases_deg = coef.g(:);

fid = fopen(archivo_salida_txt, 'w');
fprintf(fid, '# Coeficientes de marea generados por UTide en MATLAB\n');
fprintf(fid, '# Fecha: %s\n', datestr(now));
fprintf(fid, '# Latitud: %.4f\n', latitud);
fprintf(fid, '# Método: OLS, NoTrend, White\n');
fprintf(fid, '# Formato: nombre, frecuencia_cpd, amplitud_m, fase_deg\n');
fprintf(fid, '# Nivel_medio: %.6f\n', nivel_medio);
fprintf(fid, 'nombre,frecuencia_cpd,amplitud_m,fase_deg\n');

for i = 1:length(nombres)
    fprintf(fid, '%s,%.10f,%.6f,%.4f\n', ...
        nombres{i}, frecuencias_cpd(i), amplitudes_m(i), fases_deg(i));
end
fclose(fid);

fprintf(' Archivo guardado: %s\n', archivo_salida_txt);

%% --- 6. GRÁFICO DE VERIFICACIÓN DEL AJUSTE ---
fprintf('Generando gráfico de verificación...\n');

marea_predicha = ut_reconstr(tiempo_datenum, coef);

fig_verif = figure;
plot(tiempo_datenum, marea_raw, 'b-', 'DisplayName', 'Observada');
hold on;
plot(tiempo_datenum, marea_predicha, 'r-', 'LineWidth', 1.5, 'DisplayName', 'UTide (ajuste)');
datetick('x', 'dd-mmm HH:MM');
xlabel('Tiempo');
ylabel('Altura de marea (m)');
title('Verificación del ajuste armónico con UTide (OLS + White)');
legend show;
grid on;

% Guardar figura de verificación
nombre_archivo_verif = fullfile(carpeta_figuras, 'verificacion_ajuste.png');
exportgraphics(fig_verif, nombre_archivo_verif, 'Resolution', 300);
fprintf('Gráfico de verificación guardado: %s\n', nombre_archivo_verif);

%% --- 7. RECONSTRUIR MAREA PARA UN PERÍODO FUTURO Y GUARDAR EN NETCDF ---
% ===== PARÁMETROS DE ENTRADA (modificar aquí) =====
% Define la fecha de inicio (formato: 'yyyy-mm-dd HH:MM:SS') y la duración en días
fecha_inicio_str = '2023-01-01 00:00:00';   % <--- CAMBIAR SEGÚN NECESIDAD
num_dias = 730;                               % <--- DURACIÓN EN DÍAS
% ==================================================

% Convertir fecha de inicio a datenum
fecha_inicio_datenum = datenum(fecha_inicio_str, 'yyyy-mm-dd HH:MM:SS');

% Crear vector de tiempo con paso de 5 minutos (5/(24*60) días)
dt_dias = 5 / (24*60);   % 5 minutos en días
t_futuro_datenum = fecha_inicio_datenum : dt_dias : (fecha_inicio_datenum + num_dias);
% Ajustar para asegurar que el último punto no exceda ligeramente
if t_futuro_datenum(end) > fecha_inicio_datenum + num_dias
    t_futuro_datenum(end) = [];
end

fprintf('Reconstruyendo marea para %d días desde %s con paso de 5 min...\n', ...
    num_dias, fecha_inicio_str);

% Reconstruir marea con UTide
marea_reconstruida = ut_reconstr(t_futuro_datenum, coef);

% Convertir tiempo a segundos desde 1970-01-01 (formato estándar en NetCDF)
ref_datenum_1970 = datenum('1970-01-01 00:00:00');
tiempo_segundos = (t_futuro_datenum - ref_datenum_1970) * 86400;  % 86400 seg/día

% Nombre del archivo NetCDF de salida
archivo_nc_salida = 'marea_reconstruida_utide_2023-2024.nc';

% Eliminar archivo existente para sobrescribir
if exist(archivo_nc_salida, 'file') == 2
    delete(archivo_nc_salida);
    fprintf('Archivo existente eliminado: %s\n', archivo_nc_salida);
end

% Crear archivo NetCDF
nTimes = length(t_futuro_datenum);
nccreate(archivo_nc_salida, 'time', 'Dimensions', {'time', nTimes}, ...
    'Format', 'netcdf4', 'Datatype', 'double');
nccreate(archivo_nc_salida, 'marea', 'Dimensions', {'time', nTimes}, ...
    'Format', 'netcdf4', 'Datatype', 'double', 'FillValue', NaN);

% Escribir datos
ncwrite(archivo_nc_salida, 'time', tiempo_segundos);
ncwrite(archivo_nc_salida, 'marea', marea_reconstruida);

% Agregar atributos
ncwriteatt(archivo_nc_salida, 'time', 'units', 'seconds since 1970-01-01 00:00:00');
ncwriteatt(archivo_nc_salida, 'time', 'long_name', 'time');
ncwriteatt(archivo_nc_salida, 'time', 'calendar', 'gregorian');
ncwriteatt(archivo_nc_salida, 'time', 'standard_name', 'time');

ncwriteatt(archivo_nc_salida, 'marea', 'units', 'm');
ncwriteatt(archivo_nc_salida, 'marea', 'long_name', 'sea surface height above reference');
ncwriteatt(archivo_nc_salida, 'marea', 'standard_name', 'sea_surface_height_above_reference_datum');

% Atributos globales
ncwriteatt(archivo_nc_salida, '/', 'title', 'Marea reconstruida mediante análisis armónico (UTide)');
ncwriteatt(archivo_nc_salida, '/', 'source', 'Coeficientes derivados de datos observados en Cáhuil');
ncwriteatt(archivo_nc_salida, '/', 'latitud', latitud);
ncwriteatt(archivo_nc_salida, '/', 'fecha_inicio_reconstruccion', fecha_inicio_str);
ncwriteatt(archivo_nc_salida, '/', 'duracion_dias', num_dias);
ncwriteatt(archivo_nc_salida, '/', 'paso_tiempo_min', 5);
ncwriteatt(archivo_nc_salida, '/', 'fecha_creacion', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

fprintf('Archivo NetCDF guardado: %s\n', archivo_nc_salida);

%% --- 8. GRÁFICO DE LA MAREA RECONSTRUIDA COMPARADA CON OBSERVACIONES ---
fprintf('Generando gráfico de la marea reconstruida con comparación observada...\n');

% Extraer datos observados en el mismo período desde el archivo original
% (se asume que ya se leyó el archivo al inicio, pero volvemos a leer por claridad)
tiempo_obs_raw = ncread(archivo_nc, nombre_var_tiempo);
marea_obs_raw = ncread(archivo_nc, nombre_var_marea);

% Convertir tiempo observado a datenum (misma lógica que al inicio)
minutos_obs = double(tiempo_obs_raw);
dias_obs_desde_base = minutos_obs / (24*60);
tiempo_obs_datenum = fecha_base_datenum + dias_obs_desde_base;

% Filtrar observaciones en el rango de la reconstrucción
idx_en_rango = (tiempo_obs_datenum >= fecha_inicio_datenum) & ...
               (tiempo_obs_datenum <= fecha_inicio_datenum + num_dias);
tiempo_obs_rango = tiempo_obs_datenum(idx_en_rango);
marea_obs_rango = marea_obs_raw(idx_en_rango);

fprintf('  - %d datos observados en el período seleccionado.\n', length(marea_obs_rango));

% Crear figura
fig_reconst = figure;
plot(t_futuro_datenum, marea_reconstruida, 'g-', 'LineWidth', 1.5, ...
     'DisplayName', 'Reconstruida (UTide, 5 min)');
hold on;
if ~isempty(marea_obs_rango)
    plot(tiempo_obs_rango, marea_obs_rango, 'b.', 'MarkerSize', 4, ...
         'DisplayName', 'Observada (original, 10 min)');
else
    warning('No hay datos observados en el período seleccionado para comparar.');
end
datetick('x', 'dd-mmm HH:MM');
xlabel('Tiempo');
ylabel('Altura de marea (m)');
title(sprintf('Marea reconstruida vs observada (%s a %s, %d días)', ...
      fecha_inicio_str, datestr(fecha_inicio_datenum+num_dias, 'yyyy-mm-dd'), num_dias));
legend show;
grid on;

% Guardar figura de reconstrucción
nombre_archivo_reconst = fullfile(carpeta_figuras, 'marea_reconstruida_comparada.png');
exportgraphics(fig_reconst, nombre_archivo_reconst, 'Resolution', 300);
fprintf('Gráfico de reconstrucción comparada guardado: %s\n', nombre_archivo_reconst);