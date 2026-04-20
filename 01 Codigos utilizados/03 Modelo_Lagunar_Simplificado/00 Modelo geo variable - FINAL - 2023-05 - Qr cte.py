# -*- coding: utf-8 -*-
# MODELO SIMPLIFICADO DE LAGUNA COSTERA - CON ANCHO REAL DESDE NETCDF Y ESTADOS DE CONEXIÓN
# SIMULACION POR INTERVALOS: solo se simula en los períodos donde hay datos de ancho.
# Los saltos entre intervalos se omiten.
# Qr constante (ingresado como parámetro), Zm INTERPOLADO DESDE MAREA RECONSTRUIDA POR UTIDE EN MATLAB
# SIMULACION CONTINUA EN PERIODO DE TIEMPO DONDE EXISTE bc.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime
import os

class ModeloLagunaCostera:
    def __init__(self, archivo_marea, archivo_batimetria, parametros, archivo_anchos_nc=None):
        """
        Inicializa el modelo con datos de marea, batimetría, ancho real desde NetCDF
        y caudal del río constante (incluido en parametros).
        """
        # Parámetros físicos desde el diccionario de configuración
        self.g = parametros['g']      # Aceleración de gravedad (m/s²)
        self.bc = parametros['bc']    # Ancho de la boca (m) - valor por defecto
        self.hc = parametros['hc']    # Profundidad en desembocadura (m)
        self.Lc = parametros['Lc']    # Longitud del canal (m)
        self.n = parametros['n']      # Coeficiente de Manning
        self.ZL0 = parametros['ZL0']  # Nivel inicial de la laguna (m)
        self.Qr_constante = parametros['Qr_constante']  # Caudal constante del río (m³/s)

        # Cargar datos externos (solo marea y batimetría)
        self.cargar_datos_marea(archivo_marea)
        self.cargar_batimetria_real(archivo_batimetria)
        
        # Cargar datos de ancho REAL desde NetCDF (si se proporciona)
        if archivo_anchos_nc:
            self.cargar_datos_ancho_netcdf(archivo_anchos_nc)
        else:
            self.bc_variable = False
            print("   Usando ancho constante del canal")

    def verificar_coherencia_temporal(self, fecha_inicio_sim, duracion_dias):
        """
        Verifica que el rango de la simulación se solape con los datos de entrada (marea y ancho).
        """
        # Rango de la simulación
        sim_start = np.datetime64(fecha_inicio_sim)
        sim_end = sim_start + np.timedelta64(int(duracion_dias * 24 * 60), 'm')
        
        # Mostrar rangos para diagnóstico
        print(f"\n  Verificando coherencia temporal:")
        print(f"    Simulación: {pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Rango de datos de marea
        marea_start = self.tiempo_marea[0]
        marea_end = self.tiempo_marea[-1]
        print(f"    Marea: {pd.to_datetime(marea_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(marea_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Verificar solapamiento con marea
        if sim_end < marea_start or sim_start > marea_end:
            raise ValueError(f"El rango de la simulación no se solapa con el de la marea. Ajusta FECHA_INICIO o DURACION_DIAS.")
        
        # Verificar solapamiento con datos de ancho (si se usan)
        if hasattr(self, 'bc_variable') and self.bc_variable:
            ancho_start = self.tiempos_ancho_nc[0]
            ancho_end = self.tiempos_ancho_nc[-1]
            print(f"    Ancho: {pd.to_datetime(ancho_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(ancho_end).strftime('%Y-%m-%d %H:%M')}")
            if sim_end < ancho_start or sim_start > ancho_end:
                raise ValueError(f"El rango de la simulación no se solapa con el de los datos de ancho.")
        
        print("   Coherencia temporal verificada: todos los datos cubren el período de simulación.\n")
    
    def cargar_datos_ancho_netcdf(self, archivo_anchos_nc):
        """Carga y procesa datos de ancho REAL desde archivo NetCDF con las reglas específicas"""
        
        try:
            # Abrir el archivo NetCDF con datos de anchos
            ds_anchos = xr.open_dataset(archivo_anchos_nc)
            
            # Verificar que existan las variables necesarias
            variables_requeridas = ['time', 'ancho_desembocadura', 'estado_apertura']
            for var in variables_requeridas:
                if var not in ds_anchos:
                    print(f"   Variable requerida '{var}' no encontrada en el archivo NetCDF")
                    self.bc_variable = False
                    return
            
            # Extraer datos
            tiempos_ancho = ds_anchos.time.values
            anchos = ds_anchos.ancho_desembocadura.values
            estados = ds_anchos.estado_apertura.values
            
            # Convertir a formato datetime64 para facilitar comparaciones
            tiempos_ancho = tiempos_ancho.astype('datetime64[s]')
            
            # Guardar datos originales para uso posterior
            self.tiempos_ancho_nc = tiempos_ancho
            self.anchos_nc = anchos
            self.estados_nc = estados
            
            # Para búsqueda rápida, crear arrays de segundos
            self.tiempos_ancho_segundos = tiempos_ancho.astype('int64')
            
            self.bc_variable = True
            
            # Mostrar estadísticas de los datos de ancho
            print(f"   Datos REALES de ancho cargados exitosamente")
            print(f"    • Registros: {len(tiempos_ancho)}")
            print(f"    • Rango temporal: {pd.to_datetime(tiempos_ancho[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(tiempos_ancho[-1]).strftime('%Y-%m-%d %H:%M')}")
            print(f"    • Rango de ancho: {anchos.min():.2f} a {anchos.max():.2f} m")
            print(f"    • Estados - Abierto: {np.sum(estados == 1)}, Cerrado: {np.sum(estados == 0)}")
            
            # Cerrar dataset (los datos ya están en memoria)
            ds_anchos.close()
            
        except Exception as e:
            print(f"   ERROR cargando datos de ancho desde NetCDF: {e}")
            self.bc_variable = False
    
    def obtener_ancho_estado_real(self, t):
        """
        Obtiene el ancho y estado para un tiempo t usando las reglas especificadas:
        1. Encontrar los dos instantes más cercanos (anterior y posterior)
        2. Si diferencia > 4 horas -> usar el más cercano
        3. Si ambos abiertos -> interpolación lineal
        4. Si ambos cerrados -> estado cerrado
        5. Si uno abierto y otro cerrado -> interpolar entre el ancho del abierto y 0.25 m para el cerrado
        """
        if not hasattr(self, 'bc_variable') or not self.bc_variable:
            return self.bc, 1  # Valor constante y estado abierto por defecto

        try:
            # Convertir tiempo t a segundos
            t_seconds = t.astype('int64')
            
            # Encontrar índices de los dos puntos más cercanos
            idx = np.searchsorted(self.tiempos_ancho_segundos, t_seconds)
            
            idx_before = max(idx - 1, 0)
            idx_after = min(idx, len(self.tiempos_ancho_segundos) - 1)
            
            if idx == 0:
                idx_before = idx_after
            if idx == len(self.tiempos_ancho_segundos):
                idx_after = idx_before
            
            t_before = self.tiempos_ancho_segundos[idx_before]
            t_after = self.tiempos_ancho_segundos[idx_after]
            
            diff_before = abs(t_seconds - t_before)
            diff_after = abs(t_seconds - t_after)
            
            umbral_4_horas = 4 * 3600  # 4 horas en segundos

            # REGLA: Si diferencia > 4 horas -> usar el más cercano
            if diff_before > umbral_4_horas and diff_after > umbral_4_horas:
                if diff_before <= diff_after:
                    ancho = self.anchos_nc[idx_before]
                    estado = self.estados_nc[idx_before]
                else:
                    ancho = self.anchos_nc[idx_after]
                    estado = self.estados_nc[idx_after]
            elif diff_before > umbral_4_horas:
                ancho = self.anchos_nc[idx_after]
                estado = self.estados_nc[idx_after]
            elif diff_after > umbral_4_horas:
                ancho = self.anchos_nc[idx_before]
                estado = self.estados_nc[idx_before]
            else:
                # Ambos dentro de 4 horas
                estado_before = self.estados_nc[idx_before]
                estado_after = self.estados_nc[idx_after]
                
                # REGLA: Si ambos abiertos -> interpolación lineal
                if estado_before == 1 and estado_after == 1:
                    if diff_before + diff_after > 0:
                        peso_after = diff_before / (diff_before + diff_after)
                        peso_before = diff_after / (diff_before + diff_after)
                        ancho = self.anchos_nc[idx_before] * peso_before + self.anchos_nc[idx_after] * peso_after
                    else:
                        ancho = self.anchos_nc[idx_before]
                    estado = 1
                
                # REGLA: Si ambos cerrados -> estado cerrado
                elif estado_before == 0 and estado_after == 0:
                    ancho = 0.01
                    estado = 0
                
                # REGLA: Si uno abierto y otro cerrado -> interpolar entre ancho del abierto y 0.25 m para el cerrado
                else:
                    if diff_before + diff_after > 0:
                        peso_after = diff_before / (diff_before + diff_after)
                        peso_before = diff_after / (diff_before + diff_after)
                    else:
                        peso_after = peso_before = 0.5
                    
                    if estado_before == 1 and estado_after == 0:
                        ancho = self.anchos_nc[idx_before] * peso_before + 0.25 * peso_after
                    else:  # estado_before == 0 and estado_after == 1
                        ancho = 0.25 * peso_before + self.anchos_nc[idx_after] * peso_after
                    estado = 1  # El canal se considera abierto (con ancho interpolado)

            # Ajustar ancho según estado
            if estado == 0:
                ancho_final = 0.01
            else:
                ancho_final = max(ancho, 0.01)
            
            return ancho_final, int(estado)

        except Exception as e:
            # Fallback al valor constante
            return self.bc, 1

    def obtener_ancho_estado(self, t):
        """Wrapper para mantener compatibilidad con el código existente"""
        return self.obtener_ancho_estado_real(t)

    def cargar_datos_marea(self, archivo):
        """
        Carga y procesa datos de marea desde NetCDF.
        Lee variable 'marea' y tiempo en segundos desde 1970-01-01.
        """
        try:
            ds = xr.open_dataset(archivo)
            
            if 'marea' in ds.variables:
                marea_original = ds.marea.values
                variable_usada = 'marea'
            elif 'altura_cahuil_estimada' in ds.variables:
                marea_original = ds.altura_cahuil_estimada.values
                variable_usada = 'altura_cahuil_estimada'
                print("   ADVERTENCIA: Usando 'altura_cahuil_estimada' en lugar de 'marea'.")
            else:
                for var_name in ds.variables:
                    if len(ds[var_name].dims) == 1 and 'time' in ds[var_name].dims:
                        marea_original = ds[var_name].values
                        variable_usada = var_name
                        print(f"   ADVERTENCIA: Usando variable '{var_name}' como marea.")
                        break
            
            # Leer tiempo (segundos desde 1970-01-01)
            tiempo_segundos = ds.time.values
            tiempo_dt = pd.to_datetime(tiempo_segundos, unit='s').to_numpy().astype('datetime64[s]')
            
            if len(tiempo_dt) < 2:
                raise ValueError(f"El archivo de marea tiene solo {len(tiempo_dt)} punto(s). Se necesitan al menos 2.")
            
            # Guardar datos originales para interpolación
            self.marea_data = marea_original
            self.tiempo_marea = tiempo_dt
            
            # CREAR INTERPOLADOR DE MAREA
            tiempos_segundos_int64 = self.tiempo_marea.astype('int64')
            self.marea_interp_func = interp1d(tiempos_segundos_int64, self.marea_data, 
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=(self.marea_data[0], self.marea_data[-1]))
            
            print(f"   Datos de marea cargados:")
            print(f"    • Registros: {len(tiempo_dt)}")
            print(f"    • Rango temporal: {pd.to_datetime(tiempo_dt[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(tiempo_dt[-1]).strftime('%Y-%m-%d %H:%M')}")
            print(f"    • Rango de marea: {marea_original.min():.2f} a {marea_original.max():.2f} m")
            print(f"    • Variable usada: {variable_usada}")
                
        except Exception as e:
            print(f"   ERROR cargando datos de marea: {e}")
            raise
    
    def cargar_batimetria_real(self, archivo_batimetria):
        """Carga batimetría real para cálculo directo de volumen"""
        try:
            ds_bat = xr.open_dataset(archivo_batimetria)
            
            if 'elevacion' in ds_bat.variables:
                self.batimetria = ds_bat.elevacion.values
                self.x_coords = ds_bat.x.values
                self.y_coords = ds_bat.y.values
            else:
                raise ValueError("No se encontró variable 'elevacion' en el archivo de batimetría")
            
            self.dx = abs(self.x_coords[1] - self.x_coords[0]) if len(self.x_coords) > 1 else 1.0
            self.dy = abs(self.y_coords[1] - self.y_coords[0]) if len(self.y_coords) > 1 else 1.0
            self.area_celda = self.dx * self.dy
            self.area_total = self.batimetria.size * self.area_celda
            
            print(f"   Batimetría cargada:")
            print(f"    • Dimensiones: {self.batimetria.shape[0]} x {self.batimetria.shape[1]} celdas")
            print(f"    • Área por celda: {self.area_celda:.2f} m²")
            print(f"    • Área total del dominio: {self.area_total / 1e6:.2f} km²")
            print(f"    • Rango de elevación: {self.batimetria.min():.2f} a {self.batimetria.max():.2f} m")
            
        except Exception as e:
            print(f"   ERROR cargando batimetría: {e}")
            raise
    
    def calcular_volumen_directo(self, Z):
        """
        Calcula el volumen directamente desde la batimetría para una altura Z dada
        Permite niveles negativos pero con volumen positivo cuando Z > batimetría
        """
        try:
            diferencia = Z - self.batimetria
            volumen = np.sum(np.where(diferencia > 0, diferencia, 0)) * self.area_celda
            return max(volumen, 0.0)
        except Exception as e:
            return 0.0
    
    def calcular_area_superficial(self, Z):
        """
        Calcula el área superficial de la laguna para un nivel Z dado
        Esto es A_L en la ecuación de continuidad
        """
        try:
            area_superficial = np.sum(np.where(Z > self.batimetria, 1, 0)) * self.area_celda
            return max(area_superficial, 1.0)
        except Exception as e:
            return 1.0
    
    def calcular_derivada_area_nivel(self, Z, epsilon=1e-3):
        """
        Calcula numéricamente dA_L/dZ_L usando la batimetría
        Usa diferencia central para mayor precisión
        
        Ecuación: dA_L/dZ_L ≈ [A_L(Z + ε) - A_L(Z - ε)] / (2ε)
        """
        try:
            A_L_plus = self.calcular_area_superficial(Z + epsilon)
            A_L_minus = self.calcular_area_superficial(Z - epsilon)
            
            dA_dZ = (A_L_plus - A_L_minus) / (2 * epsilon)
            dA_dZ = max(dA_dZ, 0.0)
            return dA_dZ
            
        except Exception as e:
            A_L = self.calcular_area_superficial(Z)
            return A_L / 10.0
        
    def marea_interp(self, t):
        """Interpolación de marea en tiempo t usando interpolación directa desde datos reales"""
        try:
            t_seconds = t.astype('datetime64[s]').astype('int64')
            marea_interp = float(self.marea_interp_func(t_seconds))
            marea_interp = np.clip(marea_interp, -2.0, 2.0)
            return marea_interp
            
        except Exception as e:
            try:
                t_seconds = t.astype('int64') / 1e9
                tiempos_ref = self.tiempo_marea.astype('int64') / 1e9
                marea_interp = np.interp(t_seconds, tiempos_ref, self.marea_data,
                                       left=self.marea_data[0], right=self.marea_data[-1])
                return marea_interp
            except:
                return 0.0
        
    def ecuaciones_gobernantes(self, t, y):
        """
        Sistema de EDOs para la laguna costera CON ANCHO REAL Y ESTADOS
        Basado EXACTAMENTE en las ecuaciones del documento LaTeX.
        Qr es constante (self.Qr_constante).
        """
        Qc, ZL = y
        
        try:
            Zm = self.marea_interp(t)      # Nivel del mar (marea)
            Qr = self.Qr_constante          # Caudal del río constante
            
            # OBTENER ANCHO Y ESTADO REAL DEL CANAL
            bc_actual, estado = self.obtener_ancho_estado(t)
            
            # LIMITAR VALORES PARA ESTABILIDAD NUMÉRICA
            ZL = np.clip(ZL, -10, 10)
            Qc = np.clip(Qc, -1000, 1000)
            
            # CALCULAR PARÁMETROS DE LA LAGUNA
            A_L = self.calcular_area_superficial(ZL)
            dA_dZ = self.calcular_derivada_area_nivel(ZL)
            
            # DENOMINADOR COMÚN PARA LA ECUACIÓN DE CONTINUIDAD
            denominador_continuidad = ZL * dA_dZ + A_L
            if abs(denominador_continuidad) < 1e-6:
                denominador_continuidad = 1e-6 if denominador_continuidad >= 0 else -1e-6
            
            # VERIFICAR ESTADO DEL CANAL
            if estado == 0:  # CANAL CERRADO
                Qc_efectivo = 0.0
                dQc_dt = 0.0
                if abs(denominador_continuidad) > 1e-6:
                    dZL_dt = Qr / denominador_continuidad
                    dZL_dt = np.clip(dZL_dt, -1, 1)
                else:
                    dZL_dt = 0
                    
            else:  # CANAL ABIERTO
                A_c = bc_actual * (self.hc + abs(ZL - Zm) / 2)
                A_c = max(A_c, 0.1)
                
                C_d = self.g * self.n**2 * self.hc**(-1/3)
                
                termino_gravedad = (self.g * A_c * (Zm - ZL)) / self.Lc
                termino_friccion = (C_d * abs(Qc) * Qc * bc_actual) / (A_c**2)
                dQc_dt = termino_gravedad - termino_friccion
                dQc_dt = np.clip(dQc_dt, -10, 10)
                
                if abs(denominador_continuidad) > 1e-6:
                    dZL_dt = (Qc + Qr) / denominador_continuidad
                    dZL_dt = np.clip(dZL_dt, -1, 1)
                else:
                    dZL_dt = 0
                
            return np.array([dQc_dt, dZL_dt])
            
        except Exception as e:
            return np.array([0.0, 0.0])
    
    def determinar_intervalos_simulacion(self, fecha_inicio, fecha_fin):
        """
        Determina los intervalos de tiempo donde hay datos de ancho,
        SIN EXTENDER (exactamente desde el primer al último dato de cada grupo).
        Retorna lista de tuplas (inicio, fin) en formato datetime64.
        """
        if not hasattr(self, 'tiempos_ancho_nc') or len(self.tiempos_ancho_nc) == 0:
            print("  ADVERTENCIA: No hay datos de ancho. Se simulará continuamente con ancho constante.")
            return [(fecha_inicio, fecha_fin)]
        
        # Filtrar tiempos dentro del rango de simulación
        mask = (self.tiempos_ancho_nc >= fecha_inicio) & (self.tiempos_ancho_nc <= fecha_fin)
        tiempos_dentro = self.tiempos_ancho_nc[mask]
        if len(tiempos_dentro) == 0:
            print("  ADVERTENCIA: No hay datos de ancho en el rango de simulación. Se usará ancho constante.")
            return [(fecha_inicio, fecha_fin)]
        
        # Ordenar
        tiempos_ordenados = np.sort(tiempos_dentro)
        
        # Agrupar por cercanía (umbral de 6 horas)
        umbral = np.timedelta64(6, 'h')
        grupos = []
        grupo_actual = [tiempos_ordenados[0]]
        for t in tiempos_ordenados[1:]:
            if t - grupo_actual[-1] <= umbral:
                grupo_actual.append(t)
            else:
                grupos.append(grupo_actual)
                grupo_actual = [t]
        grupos.append(grupo_actual)
        
        # Crear intervalos SIN EXTENDER
        intervalos = []
        for grupo in grupos:
            inicio = np.min(grupo)
            fin = np.max(grupo)
            inicio = max(inicio, fecha_inicio)
            fin = min(fin, fecha_fin)
            intervalos.append((inicio, fin))
        
        # Fusionar intervalos solapados o muy cercanos (menos de 1 minuto de separación)
        intervalos_fusionados = []
        for intervalo in intervalos:
            if not intervalos_fusionados:
                intervalos_fusionados.append(intervalo)
            else:
                ultimo = intervalos_fusionados[-1]
                if intervalo[0] <= ultimo[1] + np.timedelta64(1, 'm'):
                    nuevo_fin = max(ultimo[1], intervalo[1])
                    intervalos_fusionados[-1] = (ultimo[0], nuevo_fin)
                else:
                    intervalos_fusionados.append(intervalo)
        
        print(f"  • Intervalos de simulación determinados: {len(intervalos_fusionados)}")
        for i, (ini, fin) in enumerate(intervalos_fusionados):
            print(f"    Intervalo {i+1}: {pd.to_datetime(ini)} a {pd.to_datetime(fin)}")
        
        return intervalos_fusionados

    def runge_kutta_4_intervalo(self, t_start, t_end, y0, dt_minutes):
        """
        Versión modificada de Runge-Kutta que simula desde t_start hasta t_end
        con paso dt_minutes. Devuelve arrays de tiempo, estado, ancho y estado del canal.
        """
        dt_segundos = dt_minutes * 60
        dt_timedelta = np.timedelta64(int(dt_segundos), 's')
        duracion_total = (t_end - t_start).astype('timedelta64[s]').astype(int)
        n_steps = int(duracion_total / dt_segundos)
        
        t_array = np.zeros(n_steps + 1, dtype='datetime64[s]')
        y_array = np.zeros((n_steps + 1, 2))
        t_array[0] = t_start
        y_array[0] = y0
        
        anchos_intervalo = []
        estados_intervalo = []
        
        print(f"    Simulando intervalo: {pd.to_datetime(t_start)} a {pd.to_datetime(t_end)} ({n_steps} pasos)")
        
        for i in range(n_steps):
            t_current = t_array[i]
            y_current = y_array[i]
            
            if hasattr(self, 'bc_variable') and self.bc_variable:
                bc_actual, estado_actual = self.obtener_ancho_estado(t_current)
                anchos_intervalo.append(bc_actual)
                estados_intervalo.append(estado_actual)
            
            k1 = self.ecuaciones_gobernantes(t_current, y_current)
            k2 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k1 * dt_segundos/2)
            k3 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k2 * dt_segundos/2)
            k4 = self.ecuaciones_gobernantes(t_current + dt_timedelta, y_current + k3 * dt_segundos)
            
            y_new = y_current + (k1 + 2*k2 + 2*k3 + k4) * (dt_segundos / 6)
            
            if hasattr(self, 'bc_variable') and self.bc_variable and estado_actual == 0:
                y_new[0] = 0.0
            
            y_new[0] = np.clip(y_new[0], -1000, 1000)
            y_new[1] = np.clip(y_new[1], -10, 10)
            
            y_array[i+1] = y_new
            t_array[i+1] = t_current + dt_timedelta
        
        if hasattr(self, 'bc_variable') and self.bc_variable:
            bc_final, estado_final = self.obtener_ancho_estado(t_array[-1])
            anchos_intervalo.append(bc_final)
            estados_intervalo.append(estado_final)
        
        return t_array, y_array, np.array(anchos_intervalo), np.array(estados_intervalo)

    def simular(self, fecha_inicio, duracion_dias, dt_minutes):
        """
        Ejecuta la simulación completa, dividiendo en intervalos según disponibilidad de datos de ancho.
        """
        fecha_inicio = np.datetime64(fecha_inicio)
        fecha_fin = fecha_inicio + np.timedelta64(int(duracion_dias * 24 * 60), 'm')
        
        Qc0 = 0.0
        ZL0 = self.ZL0
        y_actual = np.array([Qc0, ZL0])
        
        tiempos_total = []
        estados_total = []
        anchos_total = []
        estados_canal_total = []
        
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"INICIANDO SIMULACIÓN POR INTERVALOS")
        print(f"════════════════════════════════════════════════════════════════")
        print(f"  • Fecha inicio: {pd.to_datetime(fecha_inicio).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Fecha fin: {pd.to_datetime(fecha_fin).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Duración: {duracion_dias:.2f} días")
        print(f"  • Paso temporal: {dt_minutes:.2f} minutos")
        print(f"  • Nivel inicial laguna: {ZL0:.2f} m")
        print(f"  • Caudal río constante: {self.Qr_constante:.3f} m³/s")
        
        # Verificar coherencia temporal (solo marea y ancho)
        self.verificar_coherencia_temporal(fecha_inicio, duracion_dias)
        
        # Determinar intervalos de simulación
        intervalos = self.determinar_intervalos_simulacion(fecha_inicio, fecha_fin)
        
        for idx, (t_start, t_end) in enumerate(intervalos):
            print(f"\n  --- Intervalo {idx+1} de {len(intervalos)} ---")
            t_intervalo, y_intervalo, a_intervalo, e_intervalo = self.runge_kutta_4_intervalo(
                t_start, t_end, y_actual, dt_minutes
            )
            
            tiempos_total.extend(t_intervalo)
            estados_total.extend(y_intervalo)
            anchos_total.extend(a_intervalo)
            estados_canal_total.extend(e_intervalo)
            
            y_actual = y_intervalo[-1]
        
        # Convertir a arrays numpy
        tiempos = np.array(tiempos_total)
        estados = np.array(estados_total)
        anchos_canal = np.array(anchos_total)
        estados_canal = np.array(estados_canal_total)
        
        self.anchos_simulacion = anchos_canal
        self.estados_simulacion = estados_canal
        
        # Calcular variables a graficar (marea y caudal constante)
        print(f"\n  Calculando variables a graficar...")
        mareas_sim = np.array([self.marea_interp(t) for t in tiempos])
        caudales_rio = np.full(len(tiempos), self.Qr_constante)  # Constante
        
        # Calcular volúmenes y áreas desde batimetría
        print(f"  Calculando volúmenes desde batimetría...")
        volumenes = np.array([self.calcular_volumen_directo(z) for z in estados[:, 1]])
        areas_superficiales = np.array([self.calcular_area_superficial(z) for z in estados[:, 1]])
        derivadas_area = np.array([self.calcular_derivada_area_nivel(z) for z in estados[:, 1]])
        
        resultados = {
            'tiempos': tiempos,
            'caudal_boca': estados[:, 0],
            'nivel_laguna': estados[:, 1],
            'marea': mareas_sim,
            'caudal_rio': caudales_rio,
            'volumen': volumenes,
            'area_superficial': areas_superficiales,
            'derivada_area': derivadas_area,
            'estado_canal': estados_canal,
            'ancho_canal': anchos_canal,
            'fecha_inicio': fecha_inicio,
            'fecha_fin': fecha_fin,
            'duracion_dias': duracion_dias,
            'dt_minutes': dt_minutes
        }
        
        return resultados

    def graficar_resultados(self, resultados, archivo_salida=None):
        """
        Genera gráficos completos de los resultados.
        Copia exacta del formato del Código 1, adaptado a Qr constante.
        """
        import matplotlib.dates as mdates
        from datetime import timedelta

        # Crear directorio si no existe
        if archivo_salida:
            directorio = os.path.dirname(archivo_salida)
            if directorio and not os.path.exists(directorio):
                os.makedirs(directorio)

        # Crear figura con 6 subplots en el orden del Código 1, compartiendo eje X
        fig, axes = plt.subplots(6, 1, figsize=(16, 22), sharex=True,
                                 gridspec_kw={'hspace': 0.45, 'top': 0.96, 'bottom': 0.06})
        ax1, ax2, ax3, ax4, ax5, ax6 = axes

        # Convertir tiempos a datetime para plotting
        tiempos_datetime = [pd.to_datetime(t) for t in resultados['tiempos']]

        # Muestrear para evitar sobrecarga (solo para series continuas densas)
        step_plot = max(1, len(tiempos_datetime) // 2000)

        # Configurar formato de eje X (se aplica al último eje, los anteriores heredan)
        for ax in axes:
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
        # Formato para el eje x inferior
        ax6.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax6.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        # ------------------------------------------------------------
        # Función auxiliar para insertar NaN en saltos temporales grandes
        # ------------------------------------------------------------
        def insert_nan_at_gaps(times, values, gap_threshold_minutes=30):
            """Inserta NaN en values cuando hay gaps > gap_threshold."""
            if len(times) == 0:
                return times, values
            gap_threshold = gap_threshold_minutes  # minutos
            new_times = [times[0]]
            new_vals = [values[0]]
            for i in range(1, len(times)):
                gap = (times[i] - times[i-1]) / np.timedelta64(1, 'm')
                if gap > gap_threshold:
                    # Insertar NaN para romper la línea
                    new_times.append(times[i-1] + np.timedelta64(int(gap_threshold), 'm'))
                    new_vals.append(np.nan)
                    new_times.append(times[i] - np.timedelta64(int(gap_threshold), 'm'))
                    new_vals.append(np.nan)
                new_times.append(times[i])
                new_vals.append(values[i])
            return np.array(new_times), np.array(new_vals)

        # ------------------------------------------------------------
        # Preparar serie continua de marea y Qr constante para todo el período
        # ------------------------------------------------------------
        t_start = resultados['fecha_inicio']
        t_end   = resultados['fecha_fin']
        dt_min  = resultados['dt_minutes']
        num_steps_cont = int((t_end - t_start) / np.timedelta64(int(dt_min*60), 's')) + 1
        t_cont = np.array([t_start + i * np.timedelta64(int(dt_min*60), 's')
                           for i in range(num_steps_cont)])
        Zm_cont = np.array([self.marea_interp(t) for t in t_cont])
        Qr_cont = np.full(num_steps_cont, self.Qr_constante)  # constante

        t_cont_dt = [pd.to_datetime(t) for t in t_cont]

        # ------------------------------------------------------------
        # 1) GRÁFICO: Marea reconstruida (línea roja continua) y Z_L (línea azul con gaps)
        # ------------------------------------------------------------
        ax1.plot(t_cont_dt[::step_plot], Zm_cont[::step_plot],
                 'r-', linewidth=1.5, label='Nivel marea Z_m (reconstruida)', alpha=0.9)

        # Z_L con gaps
        t_sim = resultados['tiempos']
        ZL = resultados['nivel_laguna']
        t_sim_gap, ZL_gap = insert_nan_at_gaps(t_sim, ZL, gap_threshold_minutes=2*resultados['dt_minutes'])
        t_sim_gap_dt = [pd.to_datetime(t) for t in t_sim_gap]
        ax1.plot(t_sim_gap_dt, ZL_gap, 'b-', linewidth=1.5, label='Nivel laguna Z_L (simulado)')

        ax1.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
        ax1.set_title('NIVELES DE AGUA - Marea y Laguna', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # ------------------------------------------------------------
        # 2) GRÁFICO: Ancho de la boca (bc) - simulado con gaps + puntos reales
        # ------------------------------------------------------------
        if 'ancho_canal' in resultados:
            bc_sim = resultados['ancho_canal']
            t_sim_gap, bc_sim_gap = insert_nan_at_gaps(t_sim, bc_sim, gap_threshold_minutes=2*resultados['dt_minutes'])
            t_sim_gap_dt = [pd.to_datetime(t) for t in t_sim_gap]
            ax2.plot(t_sim_gap_dt, bc_sim_gap, 'y-', linewidth=1.5, label='Ancho simulado')

        # Puntos originales del NetCDF
        if hasattr(self, 'bc_variable') and self.bc_variable and hasattr(self, 'tiempos_ancho_nc'):
            t_orig = self.tiempos_ancho_nc
            bc_orig = self.anchos_nc
            mask = (t_orig >= resultados['tiempos'][0]) & (t_orig <= resultados['tiempos'][-1])
            if np.any(mask):
                t_orig_filt = [pd.to_datetime(t) for t in t_orig[mask]]
                bc_orig_filt = bc_orig[mask]
                ax2.plot(t_orig_filt, bc_orig_filt, 'bo', markersize=3,
                         label='Ancho real (datos originales)', alpha=0.7)

        ax2.set_ylabel('bc (m)', fontsize=12, fontweight='bold')
        ax2.set_title('ANCHO DE LA DESEMBOCADURA - Simulado y real', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.tick_params(axis='y', labelsize=10)

        # ------------------------------------------------------------
        # 3) GRÁFICO: Caudal Qc (con gaps)
        # ------------------------------------------------------------
        Qc = resultados['caudal_boca']
        t_sim_gap, Qc_gap = insert_nan_at_gaps(t_sim, Qc, gap_threshold_minutes=2*resultados['dt_minutes'])
        t_sim_gap_dt = [pd.to_datetime(t) for t in t_sim_gap]
        Qc_min = np.min(Qc)
        Qc_max = np.max(Qc)
        ax3.plot(t_sim_gap_dt, Qc_gap, 'g-', linewidth=1.5,
                 label=f'Qc (min={Qc_min:.2f}, max={Qc_max:.2f} m³/s)')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
        ax3.set_ylabel('Qc (m³/s)', fontsize=12, fontweight='bold')
        ax3.set_title('CAUDAL DE LA DESEMBOCADURA Qc', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='upper right')
        ax3.tick_params(axis='y', labelsize=10)
        ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # ------------------------------------------------------------
        # 4) GRÁFICO: Área transversal Ac (con gaps)
        # ------------------------------------------------------------
        ZL = resultados['nivel_laguna']
        Zm = resultados['marea']
        bc = resultados['ancho_canal']
        hc = self.hc
        Ac = bc * (hc + np.abs(ZL - Zm) / 2.0)
        t_sim_gap, Ac_gap = insert_nan_at_gaps(t_sim, Ac, gap_threshold_minutes=2*resultados['dt_minutes'])
        t_sim_gap_dt = [pd.to_datetime(t) for t in t_sim_gap]
        ax4.plot(t_sim_gap_dt, Ac_gap, 'm-', linewidth=1.5, label='Área transversal Ac')
        ax4.set_ylabel('Ac (m²)', fontsize=12, fontweight='bold')
        ax4.set_title('ÁREA TRANSVERSAL DEL CANAL Ac', fontsize=14, fontweight='bold', pad=15)
        ax4.legend(fontsize=11, loc='upper right')
        ax4.tick_params(axis='y', labelsize=10)

        # ------------------------------------------------------------
        # 5) GRÁFICO: Caudal Qr constante (continuo)
        # ------------------------------------------------------------
        Qr_const = self.Qr_constante
        ax5.plot(t_cont_dt[::step_plot], Qr_cont[::step_plot],
                 'b-', linewidth=1.5, label=f'Qr constante = {Qr_const:.3f} m³/s', alpha=0.8)
        ax5.set_ylabel('Qr (m³/s)', fontsize=12, fontweight='bold')
        ax5.set_title('CAUDAL DEL ESTERO NILAHUE (constante)', fontsize=14, fontweight='bold', pad=15)
        ax5.legend(fontsize=11, loc='upper right')
        ax5.tick_params(axis='y', labelsize=10)
        ax5.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # ------------------------------------------------------------
        # 6) GRÁFICO: Volumen V_L (con gaps)
        # ------------------------------------------------------------
        volumen = resultados['volumen']
        t_sim_gap, vol_gap = insert_nan_at_gaps(t_sim, volumen, gap_threshold_minutes=2*resultados['dt_minutes'])
        t_sim_gap_dt = [pd.to_datetime(t) for t in t_sim_gap]
        volumen_hm3 = vol_gap / 1e6
        ax6.plot(t_sim_gap_dt, volumen_hm3, 'purple', linewidth=1.5, label='Volumen laguna')
        ax6.set_ylabel('Volumen (hm³)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Fecha y Hora', fontsize=12, fontweight='bold')
        ax6.set_title('VOLUMEN DE LA LAGUNA', fontsize=14, fontweight='bold', pad=15)
        ax6.legend(fontsize=11, loc='upper right')
        ax6.tick_params(axis='y', labelsize=10)
        ax6.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # Ajustar layout y guardar
        fig.subplots_adjust(top=0.96, bottom=0.06, left=0.07, right=0.95, hspace=0.45)

        if archivo_salida:
            try:
                plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
                print(f"   Gráfico guardado: {archivo_salida}")
            except Exception as e:
                print(f"   Error guardando gráfico: {e}")

        plt.show()
    
    def guardar_resumen_txt(self, resultados, directorio_resultados, nombre_archivo=None):
        """
        Guarda un archivo TXT con un resumen detallado de todos los resultados de la simulación
        (idéntico al del Código 1, pero con Qr constante).
        """
        if nombre_archivo is None:
            nombre_archivo = f'resumen_resultados_simulacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        ruta_completa = os.path.join(directorio_resultados, nombre_archivo)
        
        print(f"  Guardando resumen detallado de resultados en: {ruta_completa}")
        
        try:
            with open(ruta_completa, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RESUMEN DETALLADO DE RESULTADOS DE SIMULACIÓN - LAGUNA COSTERA\n")
                f.write("=" * 100 + "\n\n")
                
                # Información general de la simulación
                f.write("INFORMACIÓN GENERAL DE LA SIMULACIÓN\n")
                f.write("-" * 50 + "\n")
                f.write(f"Fecha de inicio: {pd.to_datetime(resultados['fecha_inicio']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fecha de fin: {pd.to_datetime(resultados['fecha_fin']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duración: {resultados['duracion_dias']:.2f} días\n")
                f.write(f"Paso temporal: {resultados['dt_minutes']:.2f} minutos\n")
                f.write(f"Total de pasos de simulación: {len(resultados['tiempos'])}\n")
                f.write(f"Nivel inicial de la laguna: {self.ZL0:.4f} m\n")
                f.write(f"Caudal del río (constante): {self.Qr_constante:.4f} m³/s\n")
                f.write(f"Ancho del canal (por defecto): {self.bc:.2f} m\n")
                f.write(f"Profundidad del canal: {self.hc:.2f} m\n")
                f.write(f"Longitud del canal: {self.Lc:.2f} m\n")
                f.write(f"Coeficiente de Manning: {self.n:.4f}\n\n")
                
                # Estadísticas generales
                f.write("ESTADÍSTICAS GENERALES DE LA SIMULACIÓN\n")
                f.write("-" * 50 + "\n")
                
                Qc = resultados['caudal_boca']
                f.write("\nCAUDAL DE LA BOCA (Qc):\n")
                f.write(f"  Mínimo: {np.min(Qc):.6f} m³/s\n")
                f.write(f"  Máximo: {np.max(Qc):.6f} m³/s\n")
                f.write(f"  Promedio: {np.mean(Qc):.6f} m³/s\n")
                f.write(f"  Mediana: {np.median(Qc):.6f} m³/s\n")
                f.write(f"  Desviación estándar: {np.std(Qc):.6f} m³/s\n")
                f.write(f"  Caudal positivo (entrada mar→laguna): {np.sum(Qc > 0)} pasos\n")
                f.write(f"  Caudal negativo (salida laguna→mar): {np.sum(Qc < 0)} pasos\n")
                f.write(f"  Caudal cero: {np.sum(Qc == 0)} pasos\n")
                
                ZL = resultados['nivel_laguna']
                f.write("\nNIVEL DE LA LAGUNA (ZL):\n")
                f.write(f"  Mínimo: {np.min(ZL):.6f} m\n")
                f.write(f"  Máximo: {np.max(ZL):.6f} m\n")
                f.write(f"  Promedio: {np.mean(ZL):.6f} m\n")
                f.write(f"  Mediana: {np.median(ZL):.6f} m\n")
                f.write(f"  Desviación estándar: {np.std(ZL):.6f} m\n")
                f.write(f"  Nivel positivo: {np.sum(ZL > 0)} pasos\n")
                f.write(f"  Nivel negativo: {np.sum(ZL < 0)} pasos\n")
                
                Zm = resultados['marea']
                f.write("\nMAREA (Zm):\n")
                f.write(f"  Mínimo: {np.min(Zm):.6f} m\n")
                f.write(f"  Máximo: {np.max(Zm):.6f} m\n")
                f.write(f"  Promedio: {np.mean(Zm):.6f} m\n")
                f.write(f"  Mediana: {np.median(Zm):.6f} m\n")
                f.write(f"  Desviación estándar: {np.std(Zm):.6f} m\n")
                
                Qr = resultados['caudal_rio']
                f.write("\nCAUDAL DEL RÍO (Qr):\n")
                f.write(f"  Valor constante: {Qr[0]:.6f} m³/s\n")
                
                V = resultados['volumen']
                f.write("\nVOLUMEN DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(V):.6f} m³\n")
                f.write(f"  Máximo: {np.max(V):.6f} m³\n")
                f.write(f"  Promedio: {np.mean(V):.6f} m³\n")
                f.write(f"  Mediana: {np.median(V):.6f} m³\n")
                f.write(f"  Volumen mínimo en hm³: {np.min(V)/1e6:.4f} hm³\n")
                f.write(f"  Volumen máximo en hm³: {np.max(V)/1e6:.4f} hm³\n")
                
                A = resultados['area_superficial']
                f.write("\nÁREA SUPERFICIAL DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(A):.6f} m²\n")
                f.write(f"  Máximo: {np.max(A):.6f} m²\n")
                f.write(f"  Promedio: {np.mean(A):.6f} m²\n")
                f.write(f"  Mediana: {np.median(A):.6f} m²\n")
                f.write(f"  Área mínima en km²: {np.min(A)/1e6:.4f} km²\n")
                f.write(f"  Área máxima en km²: {np.max(A)/1e6:.4f} km²\n")
                
                if 'estado_canal' in resultados:
                    estado = resultados['estado_canal']
                    total_pasos = len(estado)
                    abierto = np.sum(estado == 1)
                    cerrado = np.sum(estado == 0)
                    f.write("\nESTADO DEL CANAL:\n")
                    f.write(f"  Total de pasos: {total_pasos}\n")
                    f.write(f"  Pasos con canal ABIERTO: {abierto} ({abierto/total_pasos*100:.2f}%)\n")
                    f.write(f"  Pasos con canal CERRADO: {cerrado} ({cerrado/total_pasos*100:.2f}%)\n")
                    
                    if abierto > 0 and 'ancho_canal' in resultados:
                        bc_abiertos = resultados['ancho_canal'][estado == 1]
                        f.write(f"  Ancho promedio cuando está ABIERTO: {np.mean(bc_abiertos):.4f} m\n")
                        f.write(f"  Ancho máximo cuando está ABIERTO: {np.max(bc_abiertos):.4f} m\n")
                        f.write(f"  Ancho mínimo cuando está ABIERTO: {np.min(bc_abiertos):.4f} m\n")
                
                # Resultados paso a paso (solo si no es demasiado grande)
                if len(resultados['tiempos']) <= 10000:
                    f.write("\n" + "=" * 100 + "\n")
                    f.write("RESULTADOS DETALLADOS PASO A PASO\n")
                    f.write("=" * 100 + "\n\n")
                    f.write(f"{'Fecha/Hora':<25} | {'Qc (m³/s)':<12} | {'ZL (m)':<10} | {'Zm (m)':<10} | {'Qr (m³/s)':<12} | {'Vol (m³)':<15} | {'Área (m²)':<15} | {'Estado':<8} | {'Ancho (m)':<10}\n")
                    f.write("-" * 150 + "\n")
                    
                    tiempos = resultados['tiempos']
                    Qc = resultados['caudal_boca']
                    ZL = resultados['nivel_laguna']
                    Zm = resultados['marea']
                    Qr = resultados['caudal_rio']
                    V = resultados['volumen']
                    A = resultados['area_superficial']
                    estado = resultados.get('estado_canal', np.ones(len(tiempos), dtype=int))
                    ancho = resultados.get('ancho_canal', np.ones(len(tiempos)) * self.bc)
                    
                    for i in range(len(tiempos)):
                        fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M:%S')
                        estado_str = "ABIERTO" if estado[i] == 1 else "CERRADO"
                        f.write(f"{fecha_str:<25} | {Qc[i]:<12.6f} | {ZL[i]:<10.6f} | {Zm[i]:<10.6f} | {Qr[i]:<12.6f} | {V[i]:<15.2f} | {A[i]:<15.2f} | {estado_str:<8} | {ancho[i]:<10.4f}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("RESUMEN FINAL\n")
                f.write("=" * 100 + "\n")
                f.write(f"Total de registros guardados: {len(resultados['tiempos'])}\n")
                f.write(f"Fecha de generación del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo utilizado: Laguna Costera con ancho real desde NetCDF y Qr constante\n")
                f.write("\nFin del informe.\n")
            
            print(f"   Resumen guardado exitosamente: {ruta_completa}")
            return ruta_completa
            
        except Exception as e:
            print(f"   ERROR guardando resumen TXT: {e}")
            return None


def main():
    """Función principal de ejecución - VERSIÓN CON Qr CONSTANTE Y DATOS REALES DE ANCHO"""
    import time  # <-- NUEVA
    
    # =============================================================================
    # CONFIGURACIÓN COMPLETA
    # =============================================================================
    
    # CONFIGURACIÓN DE ARCHIVOS (se elimina el archivo de caudal)
    CONFIG_ARCHIVOS = {
        'ARCHIVO_MAREA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Data_mareas_estimadas\marea_reconstruida_utide_2023-2024.nc',
        'ARCHIVO_BATIMETRIA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\01 Resultados_mallado_kriging\superficie_kriging.nc',
        'ARCHIVO_ANCHOS_NC': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\04 Resultados_Ancho_Desembocadura\Data_Ancho_Desde_Limites_limpio.nc',
        'DIRECTORIO_RESULTADOS': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Resultados_modelo_lagunar_variable-FINAL_2023-05-Qr cte'
    }
    
    # CONFIGURACIÓN TEMPORAL
    CONFIG_TEMPORAL = {
        'PASO_TEMPORAL_MINUTOS': 0.25,
        'FECHA_INICIO': '2023-05-01 00:00',
        'DURACION_DIAS': 30
    }

    # PARÁMETROS FÍSICOS DEL MODELO (incluye Qr_constante)
    PARAMETROS_MODELO = {
        'g': 9.81,
        'bc': 25,          # Valor por defecto, se sobreescribe con datos reales.
        'hc': 0.34,        # Profundidad aprox 2023-05: 0.3374 +- 0.2538m, máximo: 0.6208m. (codigos extras calculo hc)
        'Lc': 102,         # Largo aprox 102 ± 8 m.
        'n': 0.022,        # Ven Te Chow, 1973. canal de tierra excavado, limpio, recto.
        'ZL0': 1.0,        # De los datos de terreno para la fecha y hora de inicio.
        'Qr_constante': 1  # Caudal constante del río en m³/s (ajustar según mes a simular)
    }
    
    # =============================================================================
    # EJECUCIÓN PRINCIPAL
    # =============================================================================
    
    PASO_TEMPORAL_MINUTOS = CONFIG_TEMPORAL['PASO_TEMPORAL_MINUTOS']
    FECHA_INICIO = CONFIG_TEMPORAL['FECHA_INICIO']
    DURACION_DIAS = CONFIG_TEMPORAL['DURACION_DIAS']
    
    ARCHIVO_MAREA = CONFIG_ARCHIVOS['ARCHIVO_MAREA']
    ARCHIVO_BATIMETRIA = CONFIG_ARCHIVOS['ARCHIVO_BATIMETRIA']
    ARCHIVO_ANCHOS_NC = CONFIG_ARCHIVOS['ARCHIVO_ANCHOS_NC']
    DIRECTORIO_RESULTADOS = CONFIG_ARCHIVOS['DIRECTORIO_RESULTADOS']
    
    print("════════════════════════════════════════════════════════════════")
    print("MODELO DE LAGUNA COSTERA - DATOS REALES DE ANCHO")
    print("ECUACIONES GOBERNANTES EXACTAS DEL DOCUMENTO LATEX")
    print("SIMULACIÓN POR INTERVALOS (solo donde hay datos de ancho, sin extensión)")
    print("CAUDAL DEL RÍO CONSTANTE")
    print("════════════════════════════════════════════════════════════════")
    print(f"• Paso temporal: {PASO_TEMPORAL_MINUTOS:.2f} minutos")
    print(f"• Duración: {DURACION_DIAS:.2f} días")
    print(f"• Fecha inicio: {FECHA_INICIO}")
    print(f"• Caudal río constante: {PARAMETROS_MODELO['Qr_constante']:.2f} m³/s")
    print(f"• Directorio resultados: {DIRECTORIO_RESULTADOS}")
    print(f"• Archivo de anchos REALES: {ARCHIVO_ANCHOS_NC}")
    print("════════════════════════════════════════════════════════════════")
    
    if not os.path.exists(ARCHIVO_ANCHOS_NC):
        print(f"   No se encuentra el archivo de anchos NetCDF: {ARCHIVO_ANCHOS_NC}")
        print("  Usando ancho constante en su lugar.")
        ARCHIVO_ANCHOS_NC = None
    
    if not os.path.exists(DIRECTORIO_RESULTADOS):
        os.makedirs(DIRECTORIO_RESULTADOS)
        print(f"   Directorio creado: {DIRECTORIO_RESULTADOS}")
    
    try:
        # Inicializar modelo CON DATOS REALES DE ANCHO Y Qr CONSTANTE
        print("\n════════════════════════════════════════════════════════════════")
        print("INICIALIZANDO MODELO")
        print("════════════════════════════════════════════════════════════════")
        modelo = ModeloLagunaCostera(
            ARCHIVO_MAREA, 
            ARCHIVO_BATIMETRIA, 
            PARAMETROS_MODELO,
            archivo_anchos_nc=ARCHIVO_ANCHOS_NC
        )
        
        # Ejecutar simulación
        inicio_tiempo = time.time()                     # <-- NUEVA
        resultados = modelo.simular(FECHA_INICIO, DURACION_DIAS, PASO_TEMPORAL_MINUTOS)
        
        if resultados is None:
            print("   ERROR: La simulación falló")
            return
        
        # Mostrar estadísticas de resultados
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"RESULTADOS DE LA SIMULACIÓN")
        print(f"════════════════════════════════════════════════════════════════")
        print(f"• Período: {pd.to_datetime(resultados['fecha_inicio']).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(resultados['fecha_fin']).strftime('%Y-%m-%d %H:%M')}")
        print(f"• Pasos simulados reales: {len(resultados['tiempos'])}")
        
        # Análisis del caudal del río (constante)
        Qr = resultados['caudal_rio']
        print(f"\n  CAUDAL DEL RÍO (Qr):")
        print(f"    • Valor constante: {Qr[0]:.2f} m³/s")
        
        # Análisis de la derivada dA/dZ
        if 'derivada_area' in resultados:
            dA_dZ = resultados['derivada_area']
            print(f"\n  DERIVADA dA/dZ:")
            print(f"    • Mínimo: {dA_dZ.min():.2f} m²/m")
            print(f"    • Máximo: {dA_dZ.max():.2f} m²/m")
            print(f"    • Promedio: {dA_dZ.mean():.2f} m²/m")
        
        # Análisis de estados del canal
        if 'estado_canal' in resultados:
            estado_abierto = np.sum(resultados['estado_canal'] == 1)
            estado_cerrado = np.sum(resultados['estado_canal'] == 0)
            porcentaje_abierto = estado_abierto/len(resultados['estado_canal'])*100
            porcentaje_cerrado = estado_cerrado/len(resultados['estado_canal'])*100
            
            print(f"\n  ESTADOS DEL CANAL (DATOS REALES):")
            print(f"    • Tiempo abierto: {estado_abierto} pasos ({porcentaje_abierto:.1f}%)")
            print(f"    • Tiempo cerrado: {estado_cerrado} pasos ({porcentaje_cerrado:.1f}%)")
            
            if estado_abierto > 0:
                anchos_abiertos = resultados['ancho_canal'][resultados['estado_canal'] == 1]
                print(f"    • Anchos en estado abierto:")
                print(f"      - Min: {anchos_abiertos.min():.2f} m, Max: {anchos_abiertos.max():.2f} m")
                print(f"      - Promedio: {anchos_abiertos.mean():.2f} m")
        
        # Generar gráficos
        nombre_archivo = f'Resultados_modelo_laguna_variable_FINAL_2023-05_Qr_cte.png'
        archivo_grafico = os.path.join(DIRECTORIO_RESULTADOS, nombre_archivo)
        modelo.graficar_resultados(resultados, archivo_grafico)
        
        # Guardar resultados en NetCDF
        nombre_netcdf = f'resultados_simulacion_variable_FINAL_2023-05_Qr_cte.nc'
        archivo_resultados = os.path.join(DIRECTORIO_RESULTADOS, nombre_netcdf)
        
        try:
            ds = xr.Dataset({
                'caudal_boca': (['time'], resultados['caudal_boca']),
                'nivel_laguna': (['time'], resultados['nivel_laguna']),
                'marea': (['time'], resultados['marea']),
                'caudal_rio': (['time'], resultados['caudal_rio']),
                'volumen': (['time'], resultados['volumen']),
                'area_superficial': (['time'], resultados['area_superficial']),
                'derivada_area': (['time'], resultados['derivada_area']),
                'estado_canal': (['time'], resultados['estado_canal']),
                'ancho_canal': (['time'], resultados['ancho_canal'])
            }, coords={
                'time': resultados['tiempos']
            })
            
            ds.attrs = {
                'descripcion': 'Resultados de simulación de laguna costera con datos REALES de ancho y Qr constante',
                'fecha_inicio': str(resultados['fecha_inicio']),
                'fecha_fin': str(resultados['fecha_fin']),
                'duracion_dias': resultados['duracion_dias'],
                'dt_minutes': resultados['dt_minutes'],
                'modelo': 'Laguna Costera - Runge-Kutta 4to orden',
                'Qr_constante': f'{modelo.Qr_constante} m³/s',
                'parametros_canal': f'bc={modelo.bc:.2f}m (por defecto), hc={modelo.hc:.2f}m, Lc={modelo.Lc:.2f}m, n={modelo.n:.3f}',
                'nivel_inicial': f'{modelo.ZL0:.2f} m',
                'estados_canal': '0=cerrado, 1=abierto',
                'datos_marea': 'Interpolación directa desde datos reales',
                'metodo_numerico': 'Runge-Kutta 4to orden'
            }
            
            ds.to_netcdf(archivo_resultados)
            print(f"   Resultados guardados en NetCDF: {archivo_resultados}")
            
        except Exception as e:
            print(f"   Error guardando resultados NetCDF: {e}")
        
        # Guardar resumen detallado en TXT
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"GENERANDO RESUMEN DETALLADO EN ARCHIVO TXT")
        print(f"════════════════════════════════════════════════════════════════")
        
        archivo_resumen = modelo.guardar_resumen_txt(resultados, DIRECTORIO_RESULTADOS)
        
        # Calcular tiempo transcurrido y agregarlo al archivo TXT
        fin_tiempo = time.time()                                               # <-- NUEVA
        tiempo_simulacion_seg = fin_tiempo - inicio_tiempo                     # <-- NUEVA
        if archivo_resumen:                                                    # <-- NUEVA
            with open(archivo_resumen, 'a', encoding='utf-8') as f:            # <-- NUEVA
                f.write("\n" + "=" * 100 + "\n")                               # <-- NUEVA
                f.write("TIEMPO DE SIMULACIÓN\n")                              # <-- NUEVA
                f.write("=" * 100 + "\n")                                      # <-- NUEVA
                f.write(f"Tiempo total de ejecución: {tiempo_simulacion_seg:.2f} segundos")  # <-- NUEVA
                if tiempo_simulacion_seg > 60:                                 # <-- NUEVA
                    f.write(f" ({tiempo_simulacion_seg/60:.2f} minutos)")     # <-- NUEVA
                f.write("\n")                                                  # <-- NUEVA
            print(f"   Tiempo de simulación añadido al resumen.")              # <-- NUEVA
        
        if archivo_resumen:
            print(f"   Resumen detallado guardado exitosamente: {archivo_resumen}")
        
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"SIMULACIÓN COMPLETADA EXITOSAMENTE")
        print(f"════════════════════════════════════════════════════════════════")
        print(f"  Resultados guardados en: {DIRECTORIO_RESULTADOS}")
        print(f"  Archivos generados:")
        print(f"    1. Gráfico: {archivo_grafico}")
        print(f"    2. NetCDF: {archivo_resultados}")
        print(f"    3. Resumen TXT: {archivo_resumen}")
        print(f"════════════════════════════════════════════════════════════════")
        
    except Exception as e:
        print(f"   Error en la simulación: {e}")

if __name__ == "__main__":
    main()