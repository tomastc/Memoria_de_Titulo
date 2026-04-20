# -*- coding: utf-8 -*-
# MODELO SIMPLIFICADO DE LAGUNA COSTERA - CON ANCHO REAL DESDE NETCDF Y ESTADOS DE CONEXIÓN
# SIMULACION POR INTERVALOS: solo se simula en los períodos donde hay datos de ancho.
# Los saltos entre intervalos se omiten.
# Qr y bc INTERPOLADO DIRECTAMENTE, Zm INTERPOLADO DESDE MAREA RECONSTRUIDA POR UTIDE EN MATLAB
# SIMULACION CONTINUA EN PERIODO DE TIEMPO DONDE EXISTE bc.

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime
import os
import time

class ModeloLagunaCostera:
    def __init__(self, archivo_caudal, archivo_marea, archivo_batimetria, parametros, archivo_anchos_nc=None):
        """
        Inicializa el modelo con datos de caudal, marea, batimetría y ancho real desde NetCDF
        """
        # Parámetros físicos desde el diccionario de configuración
        self.g = parametros['g']      # Aceleración de gravedad (m/s²)
        self.bc = parametros['bc']    # Ancho de la boca (m) - valor por defecto
        self.hc = parametros['hc']    # Profundidad en desembocadura (m)
        self.Lc = parametros['Lc']    # Longitud del canal (m)
        self.n = parametros['n']      # Coeficiente de Manning
        self.ZL0 = parametros['ZL0']  # Nivel inicial de la laguna (m)
        
        # Cargar datos externos
        self.cargar_datos_caudal(archivo_caudal)
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
        Verifica que el rango de la simulación se solape con los datos de entrada.
        """
        # Rango de la simulación
        sim_start = np.datetime64(fecha_inicio_sim)
        sim_end = sim_start + np.timedelta64(int(duracion_dias * 24 * 60), 'm')
        
        # Mostrar rangos para diagnóstico
        print(f"\n  Verificando coherencia temporal:")
        print(f"    Simulación: {pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Rango de datos de caudal
        caudal_start = self.tiempo_caudal[0]
        caudal_end = self.tiempo_caudal[-1]
        print(f"    Caudal: {pd.to_datetime(caudal_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(caudal_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Comprobar solapamiento con caudal
        if sim_end < caudal_start or sim_start > caudal_end:
            raise ValueError(f"El rango de la simulación no se solapa con el del caudal")
        
        # Rango de datos de marea
        marea_start = self.tiempo_marea[0]
        marea_end = self.tiempo_marea[-1]
        print(f"    Marea: {pd.to_datetime(marea_start).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(marea_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Verificar solapamiento con marea (ahora es error, no solo advertencia)
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

    def cargar_datos_caudal(self, archivo):
        """Carga y procesa datos de caudal desde NetCDF - VERSIÓN MEJORADA"""
        try:
            ds = xr.open_dataset(archivo)
            self.caudal_rio = ds.caudal.values
            self.tiempo_caudal = ds.time.values
            
            # CREAR INTERPOLADOR MEJORADO PARA CAUDAL
            tiempos_segundos = self.tiempo_caudal.astype('datetime64[s]').astype('int64')
            self.caudal_interp = interp1d(tiempos_segundos, self.caudal_rio, 
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value=(self.caudal_rio[0], self.caudal_rio[-1]))
            
            # Mostrar estadísticas del caudal
            print(f"   Datos de caudal cargados:")
            print(f"    • Registros: {len(self.tiempo_caudal)}")
            print(f"    • Rango temporal: {pd.to_datetime(self.tiempo_caudal[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(self.tiempo_caudal[-1]).strftime('%Y-%m-%d %H:%M')}")
            print(f"    • Rango de caudal: {self.caudal_rio.min():.2f} a {self.caudal_rio.max():.2f} m³/s")
            print(f"    • Promedio: {self.caudal_rio.mean():.2f} m³/s")
            print(f"    • Desviación estándar: {self.caudal_rio.std():.2f} m³/s")
            
        except Exception as e:
            print(f"   ERROR cargando datos de caudal: {e}")
            raise
        
    def cargar_datos_marea(self, archivo):
        """
        Carga y procesa datos de marea desde NetCDF.
        MODIFICADO: Lee variable 'marea' y tiempo en segundos desde 1970-01-01.
        """
        try:
            ds = xr.open_dataset(archivo)
            
            # MODIFICADO: Buscar variable 'marea' (nombre usado en MATLAB)
            if 'marea' in ds.variables:
                marea_original = ds.marea.values
                variable_usada = 'marea'
            elif 'altura_cahuil_estimada' in ds.variables:
                marea_original = ds.altura_cahuil_estimada.values
                variable_usada = 'altura_cahuil_estimada'
                print("   ADVERTENCIA: Usando 'altura_cahuil_estimada' en lugar de 'marea'.")
            else:
                # Si no existe, usar la primera variable disponible
                for var_name in ds.variables:
                    if len(ds[var_name].dims) == 1 and 'time' in ds[var_name].dims:
                        marea_original = ds[var_name].values
                        variable_usada = var_name
                        print(f"   ADVERTENCIA: Usando variable '{var_name}' como marea.")
                        break
            
            # Leer tiempo (segundos desde 1970-01-01)
            tiempo_segundos = ds.time.values
            
            # MODIFICADO: Convertir segundos desde 1970 a datetime64
            # Usamos pandas para mayor robustez, luego convertimos a numpy datetime64
            tiempo_dt = pd.to_datetime(tiempo_segundos, unit='s').to_numpy().astype('datetime64[s]')
            
            # Verificar que hay al menos 2 puntos para interpolar
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
            
            # Mostrar estadísticas de la marea
            print(f"   Datos de marea cargados:")
            print(f"    • Registros: {len(tiempo_dt)}")
            print(f"    • Rango temporal: {pd.to_datetime(tiempo_dt[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(tiempo_dt[-1]).strftime('%Y-%m-%d %H:%M')}")
            print(f"    • Rango de marea: {marea_original.min():.2f} a {marea_original.max():.2f} m")
            print(f"    • Promedio: {marea_original.mean():.2f} m")
            print(f"    • Desviación estándar: {marea_original.std():.2f} m")
            print(f"    • Variable usada: {variable_usada}")
                
        except Exception as e:
            print(f"   ERROR cargando datos de marea: {e}")
            raise
    
    def cargar_batimetria_real(self, archivo_batimetria):
        """Carga batimetría real para cálculo directo de volumen"""
        try:
            ds_bat = xr.open_dataset(archivo_batimetria)
            
            # Usar variable de elevación
            if 'elevacion' in ds_bat.variables:
                self.batimetria = ds_bat.elevacion.values
                self.x_coords = ds_bat.x.values
                self.y_coords = ds_bat.y.values
            else:
                raise ValueError("No se encontró variable 'elevacion' en el archivo de batimetría")
            
            # Calcular área de cada celda
            self.dx = abs(self.x_coords[1] - self.x_coords[0]) if len(self.x_coords) > 1 else 1.0
            self.dy = abs(self.y_coords[1] - self.y_coords[0]) if len(self.y_coords) > 1 else 1.0
            self.area_celda = self.dx * self.dy
            self.area_total = self.batimetria.size * self.area_celda
            
            # Mostrar estadísticas de la batimetría
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
            # Calcular el volumen como la suma de (Z - batimetría) para batimetría < Z
            # Esto permite Z negativos mientras Z > batimetría
            diferencia = Z - self.batimetria
            volumen = np.sum(np.where(diferencia > 0, diferencia, 0)) * self.area_celda
            return max(volumen, 0.0)  # Volumen nunca negativo
        except Exception as e:
            return 0.0
    
    def calcular_area_superficial(self, Z):
        """
        Calcula el área superficial de la laguna para un nivel Z dado
        Esto es A_L en la ecuación de continuidad
        """
        try:
            # El área superficial es el área donde Z > batimetría
            area_superficial = np.sum(np.where(Z > self.batimetria, 1, 0)) * self.area_celda
            return max(area_superficial, 1.0)  # Mínimo 1 m² para evitar división por cero
        except Exception as e:
            return 1.0
    
    def calcular_derivada_area_nivel(self, Z, epsilon=1e-3):
        """
        Calcula numéricamente dA_L/dZ_L usando la batimetría
        Usa diferencia central para mayor precisión
        
        Ecuación: dA_L/dZ_L ≈ [A_L(Z + ε) - A_L(Z - ε)] / (2ε)
        """
        try:
            # Usar diferencia central para mejor precisión
            A_L_plus = self.calcular_area_superficial(Z + epsilon)
            A_L_minus = self.calcular_area_superficial(Z - epsilon)
            
            dA_dZ = (A_L_plus - A_L_minus) / (2 * epsilon)
            
            # Asegurar que no sea negativo (el área debería aumentar con Z)
            dA_dZ = max(dA_dZ, 0.0)
            
            return dA_dZ
            
        except Exception as e:
            # Fallback: usar el área como aproximación de la derivada
            A_L = self.calcular_area_superficial(Z)
            return A_L / 10.0  # Estimación conservadora
    
    def caudal_rio_interp(self, t):
        """Interpolación MEJORADA de caudal del río en tiempo t"""
        try:
            # Usar el interpolador creado en cargar_datos_caudal
            t_seconds = t.astype('datetime64[s]').astype('int64')
            caudal_interp = float(self.caudal_interp(t_seconds))
            return max(caudal_interp, 0.0)  # El caudal del río nunca es negativo
        except Exception as e:
            # Fallback a la interpolación básica
            try:
                t_seconds = t.astype('int64') / 1e9
                tiempos_ref = self.tiempo_caudal.astype('int64') / 1e9
                caudal_interp = np.interp(t_seconds, tiempos_ref, self.caudal_rio, 
                               left=self.caudal_rio[0], right=self.caudal_rio[-1])
                return max(caudal_interp, 0.0)
            except:
                return 0.0
        
    def marea_interp(self, t):
        """Interpolación de marea en tiempo t usando interpolación directa desde datos reales"""
        try:
            # Usar el interpolador creado en cargar_datos_marea
            t_seconds = t.astype('datetime64[s]').astype('int64')
            marea_interp = float(self.marea_interp_func(t_seconds))
            
            # Aplicar límites físicos razonables (marea típica en Chile)
            marea_interp = np.clip(marea_interp, -2.0, 2.0)
            
            return marea_interp
            
        except Exception as e:
            # Fallback a la interpolación básica
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
        Basado EXACTAMENTE en las ecuaciones del documento LaTeX:
        
        1. Ecuación de continuidad (masa) para la laguna (Ecuación 4 en LaTeX):
           dZ_L/dt = (Q_c + Q_r) / (Z_L * dA_L/dZ_L + A_L)
           
        2. Ecuación de cantidad de movimiento para el canal (Ecuación 5 en LaTeX):
           dQ_c/dt = [g * A_c * (Z_m - Z_L)] / L_c - [C_d * |Q_c| * Q_c * b_c] / (A_c^2)
           
        3. Para canal cerrado (Ecuaciones 8 y 9 en LaTeX):
           Q_c = 0
           dZ_L/dt = Q_r / (Z_L * dA_L/dZ_L + A_L)
        """
        Qc, ZL = y
        
        try:
            Zm = self.marea_interp(t)      # Nivel del mar (marea) - INTERPOLACIÓN DIRECTA
            Qr = self.caudal_rio_interp(t) # Caudal del río
            
            # OBTENER ANCHO Y ESTADO REAL DEL CANAL
            bc_actual, estado = self.obtener_ancho_estado(t)
            
            # LIMITAR VALORES PARA ESTABILIDAD NUMÉRICA
            ZL = np.clip(ZL, -10, 10)      # Nivel laguna entre -10m y 10m
            Qc = np.clip(Qc, -1000, 1000)  # Caudal entre -1000 y 1000 m³/s
            
            # CALCULAR PARÁMETROS DE LA LAGUNA NECESARIOS PARA AMBOS CASOS
            # A_L: Área superficial de la laguna
            A_L = self.calcular_area_superficial(ZL)
            
            # dA_L/dZ_L: Derivada del área con respecto al nivel (calculada numéricamente)
            dA_dZ = self.calcular_derivada_area_nivel(ZL)
            
            # DENOMINADOR COMÚN PARA LA ECUACIÓN DE CONTINUIDAD
            # Según LaTeX: Z_L * dA_L/dZ_L + A_L
            denominador_continuidad = ZL * dA_dZ + A_L
            
            # Evitar división por cero
            if abs(denominador_continuidad) < 1e-6:
                denominador_continuidad = 1e-6 if denominador_continuidad >= 0 else -1e-6
            
            # VERIFICAR ESTADO DEL CANAL
            if estado == 0:  # CANAL CERRADO - APLICAR ECUACIONES (8) Y (9) DEL LATEX
                # Ecuación (8): Q_c = 0 (no hay flujo a través del canal)
                Qc_efectivo = 0.0
                dQc_dt = 0.0  # El caudal no cambia (se mantiene en 0)
                
                # Ecuación (9): dZ_L/dt = Q_r / (Z_L * dA_L/dZ_L + A_L)
                if abs(denominador_continuidad) > 1e-6:
                    dZL_dt = Qr / denominador_continuidad
                    dZL_dt = np.clip(dZL_dt, -1, 1)  # Limitar tasa de cambio
                else:
                    dZL_dt = 0
                    
            else:  # CANAL ABIERTO - ECUACIONES ORIGINALES (4) Y (5)
                # ECUACIÓN (1): Área transversal del canal (Ecuación 10 en LaTeX)
                # A_c = b_c * (h_c + |Z_L - Z_m| / 2)
                A_c = bc_actual * (self.hc + abs(ZL - Zm) / 2)
                A_c = max(A_c, 0.1)  # Mínimo área para evitar problemas numéricos
                
                # ECUACIÓN (2): Coeficiente de fricción (Ecuación 11 en LaTeX)
                # C_d = g * n^2 * h_c^(-1/3)
                C_d = self.g * self.n**2 * self.hc**(-1/3)
                
                # ECUACIÓN (5): Ecuación de momentum para el caudal (Ecuación 5 en LaTeX)
                # dQ_c/dt = [g * A_c * (Z_m - Z_L)] / L_c - [C_d * |Q_c| * Q_c * b_c] / (A_c^2)
                termino_gravedad = (self.g * A_c * (Zm - ZL)) / self.Lc
                termino_friccion = (C_d * abs(Qc) * Qc * bc_actual) / (A_c**2)
                dQc_dt = termino_gravedad - termino_friccion
                
                # Limitar la aceleración para estabilidad numérica
                dQc_dt = np.clip(dQc_dt, -10, 10)
                
                # ECUACIÓN (4): Ecuación de continuidad para el nivel (Ecuación 4 en LaTeX)
                # dZ_L/dt = (Q_c + Q_r) / (Z_L * dA_L/dZ_L + A_L)
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
            # Sin datos de ancho, simular continuamente en todo el rango
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
        
        # Crear intervalos SIN EXTENDER (exactamente desde el mínimo al máximo de cada grupo)
        intervalos = []
        for grupo in grupos:
            inicio = np.min(grupo)
            fin = np.max(grupo)
            # Ajustar a los límites de la simulación (aunque ya deberían estar dentro)
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
        # Calcular número de pasos
        dt_segundos = dt_minutes * 60
        dt_timedelta = np.timedelta64(int(dt_segundos), 's')
        duracion_total = (t_end - t_start).astype('timedelta64[s]').astype(int)
        n_steps = int(duracion_total / dt_segundos)
        
        # Inicializar arrays
        t_array = np.zeros(n_steps + 1, dtype='datetime64[s]')
        y_array = np.zeros((n_steps + 1, 2))
        t_array[0] = t_start
        y_array[0] = y0
        
        # Listas para guardar ancho y estado en cada paso
        anchos_intervalo = []
        estados_intervalo = []
        
        print(f"    Simulando intervalo: {pd.to_datetime(t_start)} a {pd.to_datetime(t_end)} ({n_steps} pasos)")
        
        for i in range(n_steps):
            t_current = t_array[i]
            y_current = y_array[i]
            
            # Guardar ancho y estado actuales
            if hasattr(self, 'bc_variable') and self.bc_variable:
                bc_actual, estado_actual = self.obtener_ancho_estado(t_current)
                anchos_intervalo.append(bc_actual)
                estados_intervalo.append(estado_actual)
            
            # Calcular coeficientes RK4
            k1 = self.ecuaciones_gobernantes(t_current, y_current)
            k2 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k1 * dt_segundos/2)
            k3 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k2 * dt_segundos/2)
            k4 = self.ecuaciones_gobernantes(t_current + dt_timedelta, y_current + k3 * dt_segundos)
            
            # Actualizar
            y_new = y_current + (k1 + 2*k2 + 2*k3 + k4) * (dt_segundos / 6)
            
            # Forzar Qc=0 si el canal está cerrado
            if hasattr(self, 'bc_variable') and self.bc_variable and estado_actual == 0:
                y_new[0] = 0.0
            
            # Límites físicos
            y_new[0] = np.clip(y_new[0], -1000, 1000)
            y_new[1] = np.clip(y_new[1], -10, 10)
            
            y_array[i+1] = y_new
            t_array[i+1] = t_current + dt_timedelta
        
        # Guardar el último ancho/estado
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
        
        # Condiciones iniciales globales
        Qc0 = 0.0
        ZL0 = self.ZL0
        y_actual = np.array([Qc0, ZL0])
        
        # Acumuladores de resultados
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
        
        # Verificar coherencia temporal (solo una vez)
        self.verificar_coherencia_temporal(fecha_inicio, duracion_dias)
        
        # Determinar intervalos de simulación
        intervalos = self.determinar_intervalos_simulacion(fecha_inicio, fecha_fin)
        
        for idx, (t_start, t_end) in enumerate(intervalos):
            print(f"\n  --- Intervalo {idx+1} de {len(intervalos)} ---")
            # Ejecutar RK4 en este intervalo
            t_intervalo, y_intervalo, a_intervalo, e_intervalo = self.runge_kutta_4_intervalo(
                t_start, t_end, y_actual, dt_minutes
            )
            
            # Concatenar resultados
            tiempos_total.extend(t_intervalo)
            estados_total.extend(y_intervalo)
            anchos_total.extend(a_intervalo)
            estados_canal_total.extend(e_intervalo)
            
            # Actualizar condición inicial para el siguiente intervalo
            y_actual = y_intervalo[-1]
        
        # Convertir a arrays numpy
        tiempos = np.array(tiempos_total)
        estados = np.array(estados_total)
        anchos_canal = np.array(anchos_total)
        estados_canal = np.array(estados_canal_total)
        
        # Guardar en el objeto para uso posterior (gráficos)
        self.anchos_simulacion = anchos_canal
        self.estados_simulacion = estados_canal
        
        # Calcular variables derivadas
        print(f"\n  Calculando variables a graficar...")
        mareas_sim = np.array([self.marea_interp(t) for t in tiempos])
        caudales_rio = np.array([self.caudal_rio_interp(t) for t in tiempos])
        
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

    def verificar_datos_ancho(self, resultados):
        """Función de diagnóstico para verificar los datos de ancho REALES"""
        
        if 'ancho_canal' in resultados:
            anchos = resultados['ancho_canal']
            estados = resultados['estado_canal']
            
            # Analizar por estados
            mascara_abierto = estados == 1
            mascara_cerrado = estados == 0
            
            if np.any(mascara_abierto):
                anchos_abiertos = anchos[mascara_abierto]
            
            if np.any(mascara_cerrado):
                anchos_cerrados = anchos[mascara_cerrado]
        
        # Verificar datos originales de NetCDF
        if hasattr(self, 'bc_variable') and self.bc_variable:
            pass

    def verificar_interpolacion_caudal(self, tiempos_simulacion):
        """Verifica que la interpolación del caudal del río funcione correctamente - VERSIÓN CORREGIDA"""
        
        # Calcular caudal interpolado para tiempos de simulación
        caudales_interpolados = np.array([self.caudal_rio_interp(t) for t in tiempos_simulacion])
        
        # Verificar si hay valores constantes
        valores_unicos = np.unique(caudales_interpolados)
        if len(valores_unicos) <= 3:
            print(f"   ADVERTENCIA: El caudal interpolado tiene muy poca variabilidad")
    
    def graficar_resultados(self, resultados, archivo_salida=None):
        """
        Genera gráficos completos de los resultados.
        MODIFICACIONES FINALES:
        - Z_L, Qc, V_L como líneas con gaps (solo intervalos simulados).
        - Qr continuo desde datos originales (igual que marea).
        - Todos los subplots comparten eje X (sharex) para alineación perfecta.
        - Reorden: Z_m+Z_L, bc, Qc, Ac, Qr, V_L.
        """
        import matplotlib.dates as mdates
        from datetime import timedelta

        # Crear directorio si no existe
        if archivo_salida:
            directorio = os.path.dirname(archivo_salida)
            if directorio and not os.path.exists(directorio):
                os.makedirs(directorio)

        # Crear figura con 6 subplots en el nuevo orden, compartiendo eje X
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
        # Preparar serie continua de marea y caudal Qr para todo el período
        # ------------------------------------------------------------
        t_start = resultados['fecha_inicio']
        t_end   = resultados['fecha_fin']
        dt_min  = resultados['dt_minutes']
        num_steps_cont = int((t_end - t_start) / np.timedelta64(int(dt_min*60), 's')) + 1
        t_cont = np.array([t_start + i * np.timedelta64(int(dt_min*60), 's')
                           for i in range(num_steps_cont)])
        Zm_cont = np.array([self.marea_interp(t) for t in t_cont])
        Qr_cont = np.array([self.caudal_rio_interp(t) for t in t_cont])

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
        # 5) GRÁFICO: Caudal Qr continuo
        # ------------------------------------------------------------
        Qr_min = np.min(Qr_cont)
        Qr_max = np.max(Qr_cont)
        ax5.plot(t_cont_dt[::step_plot], Qr_cont[::step_plot],
                 'b-', linewidth=1.5, label=f'Qr (min={Qr_min:.2f}, max={Qr_max:.2f} m³/s)', alpha=0.8)
        ax5.set_ylabel('Qr (m³/s)', fontsize=12, fontweight='bold')
        ax5.set_title('CAUDAL DEL ESTERO NILAHUE Qr (continuo)', fontsize=14, fontweight='bold', pad=15)
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
        """
        if nombre_archivo is None:
            nombre_archivo = f'resumen_resultados_simulacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        ruta_completa = os.path.join(directorio_resultados, nombre_archivo)
        
        print(f"  Guardando resumen detallado de resultados en: {ruta_completa}")
        
        try:
            with open(ruta_completa, 'w', encoding='utf-8') as f:
                # Encabezado
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
                f.write(f"Ancho del canal (por defecto): {self.bc:.2f} m\n")
                f.write(f"Profundidad del canal: {self.hc:.2f} m\n")
                f.write(f"Longitud del canal: {self.Lc:.2f} m\n")
                f.write(f"Coeficiente de Manning: {self.n:.4f}\n\n")
                
                # Estadísticas generales
                f.write("ESTADÍSTICAS GENERALES DE LA SIMULACIÓN\n")
                f.write("-" * 50 + "\n")
                
                # Estadísticas del caudal de la boca (Qc)
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
                
                # Estadísticas del nivel de la laguna (ZL)
                ZL = resultados['nivel_laguna']
                f.write("\nNIVEL DE LA LAGUNA (ZL):\n")
                f.write(f"  Mínimo: {np.min(ZL):.6f} m\n")
                f.write(f"  Máximo: {np.max(ZL):.6f} m\n")
                f.write(f"  Promedio: {np.mean(ZL):.6f} m\n")
                f.write(f"  Mediana: {np.median(ZL):.6f} m\n")
                f.write(f"  Desviación estándar: {np.std(ZL):.6f} m\n")
                f.write(f"  Nivel positivo (sobre nivel del mar): {np.sum(ZL > 0)} pasos\n")
                f.write(f"  Nivel negativo (bajo nivel del mar): {np.sum(ZL < 0)} pasos\n")
                f.write(f"  Nivel cero: {np.sum(ZL == 0)} pasos\n")
                
                # Estadísticas de la marea (Zm)
                Zm = resultados['marea']
                f.write("\nMAREA (Zm):\n")
                f.write(f"  Mínimo: {np.min(Zm):.6f} m\n")
                f.write(f"  Máximo: {np.max(Zm):.6f} m\n")
                f.write(f"  Promedio: {np.mean(Zm):.6f} m\n")
                f.write(f"  Mediana: {np.median(Zm):.6f} m\n")
                f.write(f"  Desviación estándar: {np.std(Zm):.6f} m\n")
                
                # Estadísticas del caudal del río (Qr)
                Qr = resultados['caudal_rio']
                f.write("\nCAUDAL DEL RÍO (Qr):\n")
                f.write(f"  Mínimo: {np.min(Qr):.6f} m³/s\n")
                f.write(f"  Máximo: {np.max(Qr):.6f} m³/s\n")
                f.write(f"  Promedio: {np.mean(Qr):.6f} m³/s\n")
                f.write(f"  Mediana: {np.median(Qr):.6f} m³/s\n")
                f.write(f"  Desviación estándar: {np.std(Qr):.6f} m³/s\n")
                
                # Estadísticas del volumen
                V = resultados['volumen']
                f.write("\nVOLUMEN DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(V):.6f} m³\n")
                f.write(f"  Máximo: {np.max(V):.6f} m³\n")
                f.write(f"  Promedio: {np.mean(V):.6f} m³\n")
                f.write(f"  Mediana: {np.median(V):.6f} m³\n")
                f.write(f"  Desviación estándar: {np.std(V):.6f} m³\n")
                f.write(f"  Volumen mínimo en hm³: {np.min(V)/1e6:.4f} hm³\n")
                f.write(f"  Volumen máximo en hm³: {np.max(V)/1e6:.4f} hm³\n")
                
                # Estadísticas del área superficial
                A = resultados['area_superficial']
                f.write("\nÁREA SUPERFICIAL DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(A):.6f} m²\n")
                f.write(f"  Máximo: {np.max(A):.6f} m²\n")
                f.write(f"  Promedio: {np.mean(A):.6f} m²\n")
                f.write(f"  Mediana: {np.median(A):.6f} m²\n")
                f.write(f"  Desviación estándar: {np.std(A):.6f} m²\n")
                f.write(f"  Área mínima en km²: {np.min(A)/1e6:.4f} km²\n")
                f.write(f"  Área máxima en km²: {np.max(A)/1e6:.4f} km²\n")
                
                # Estadísticas de la derivada dA/dZ (opcional, pero la mantenemos)
                if 'derivada_area' in resultados:
                    dA_dZ = resultados['derivada_area']
                    f.write("\nDERIVADA dA/dZ:\n")
                    f.write(f"  Mínimo: {np.min(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Máximo: {np.max(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Promedio: {np.mean(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Mediana: {np.median(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Desviación estándar: {np.std(dA_dZ):.6f} m²/m\n")
                
                # Estadísticas del ancho del canal
                if 'ancho_canal' in resultados:
                    bc = resultados['ancho_canal']
                    f.write("\nANCHO DEL CANAL (bc):\n")
                    f.write(f"  Mínimo: {np.min(bc):.6f} m\n")
                    f.write(f"  Máximo: {np.max(bc):.6f} m\n")
                    f.write(f"  Promedio: {np.mean(bc):.6f} m\n")
                    f.write(f"  Mediana: {np.median(bc):.6f} m\n")
                    f.write(f"  Desviación estándar: {np.std(bc):.6f} m\n")
                
                # Estadísticas del estado del canal
                if 'estado_canal' in resultados:
                    estado = resultados['estado_canal']
                    total_pasos = len(estado)
                    abierto = np.sum(estado == 1)
                    cerrado = np.sum(estado == 0)
                    porcentaje_abierto = (abierto / total_pasos) * 100
                    porcentaje_cerrado = (cerrado / total_pasos) * 100
                    
                    f.write("\nESTADO DEL CANAL:\n")
                    f.write(f"  Total de pasos: {total_pasos}\n")
                    f.write(f"  Pasos con canal ABIERTO: {abierto} ({porcentaje_abierto:.2f}%)\n")
                    f.write(f"  Pasos con canal CERRADO: {cerrado} ({porcentaje_cerrado:.2f}%)\n")
                    
                    if abierto > 0:
                        anchos_abiertos = bc[estado == 1]
                        f.write(f"  Ancho promedio cuando está ABIERTO: {np.mean(anchos_abiertos):.4f} m\n")
                        f.write(f"  Ancho máximo cuando está ABIERTO: {np.max(anchos_abiertos):.4f} m\n")
                        f.write(f"  Ancho mínimo cuando está ABIERTO: {np.min(anchos_abiertos):.4f} m\n")
                
                # RESULTADOS DETALLADOS PASO A PASO
                f.write("\n" + "=" * 100 + "\n")
                f.write("RESULTADOS DETALLADOS PASO A PASO\n")
                f.write("=" * 100 + "\n\n")
                
                # Encabezado de la tabla
                f.write(f"{'Fecha/Hora':<25} | {'Qc (m³/s)':<12} | {'ZL (m)':<10} | {'Zm (m)':<10} | {'Qr (m³/s)':<12} | {'Vol (m³)':<15} | {'Área (m²)':<15} | {'dA/dZ':<10} | {'Estado':<8} | {'Ancho (m)':<10}\n")
                f.write("-" * 150 + "\n")
                
                # Escribir cada línea de resultados (todas las líneas)
                tiempos = resultados['tiempos']
                Qc = resultados['caudal_boca']
                ZL = resultados['nivel_laguna']
                Zm = resultados['marea']
                Qr = resultados['caudal_rio']
                V = resultados['volumen']
                A = resultados['area_superficial']
                
                dA_dZ = resultados.get('derivada_area', np.zeros(len(tiempos)))
                estado = resultados.get('estado_canal', np.ones(len(tiempos), dtype=int))
                ancho = resultados.get('ancho_canal', np.ones(len(tiempos)) * self.bc)
                
                for i in range(len(tiempos)):
                    fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M:%S')
                    estado_str = "ABIERTO" if estado[i] == 1 else "CERRADO"
                    
                    f.write(f"{fecha_str:<25} | {Qc[i]:<12.6f} | {ZL[i]:<10.6f} | {Zm[i]:<10.6f} | {Qr[i]:<12.6f} | {V[i]:<15.2f} | {A[i]:<15.2f} | {dA_dZ[i]:<10.2f} | {estado_str:<8} | {ancho[i]:<10.4f}\n")
                    
                    # Mostrar progreso cada 1000 líneas
                    if i % 1000 == 0 and i > 0:
                        print(f"    Guardando línea {i} de {len(tiempos)}...")
                
                # Resumen final
                f.write("\n" + "=" * 100 + "\n")
                f.write("RESUMEN FINAL\n")
                f.write("=" * 100 + "\n")
                f.write(f"Total de registros guardados: {len(tiempos)}\n")
                f.write(f"Fecha de generación del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo utilizado: Laguna Costera con ancho real desde NetCDF\n")
                f.write(f"Ecuaciones gobernantes: Exactas del documento LaTeX\n")
                f.write(f"Método numérico: Runge-Kutta de 4to orden\n")
                f.write("\nFin del informe.\n")
            
            print(f"   Resumen guardado exitosamente: {ruta_completa}")
            print(f"    Total de líneas escritas: {len(resultados['tiempos'])}")
            
            # Mostrar información sobre el tamaño del archivo
            tamaño_bytes = os.path.getsize(ruta_completa)
            tamaño_mb = tamaño_bytes / (1024 * 1024)
            print(f"    Tamaño del archivo: {tamaño_mb:.2f} MB")
            
            return ruta_completa
            
        except Exception as e:
            print(f"   ERROR guardando resumen TXT: {e}")
            return None

def main():
    """Función principal de ejecución - VERSIÓN CON DATOS REALES DE ANCHO Y SIMULACIÓN POR INTERVALOS (sin extensión)"""
    
    # =============================================================================
    # CONFIGURACIÓN COMPLETA
    # =============================================================================
    
    # CONFIGURACIÓN DE ARCHIVOS
    CONFIG_ARCHIVOS = {
        'ARCHIVO_CAUDAL': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\03 Resultados_caudal_altura_estero\caudales_alturas_serie_temporal.nc',
        'ARCHIVO_MAREA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Data_mareas_estimadas\marea_reconstruida_utide_2023-2024.nc',
        'ARCHIVO_BATIMETRIA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\01 Resultados_mallado_kriging\superficie_kriging.nc',
        'ARCHIVO_ANCHOS_NC': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\04 Resultados_Ancho_Desembocadura\Data_Ancho_Desde_Limites_limpio.nc',
        'DIRECTORIO_RESULTADOS': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Resultados_modelo_lagunar_variable-FINAL_2024-07'
    }
    
    # CONFIGURACIÓN TEMPORAL
    CONFIG_TEMPORAL = {
        'PASO_TEMPORAL_MINUTOS': 0.25,
        'FECHA_INICIO': '2024-07-10 08:00',
        'DURACION_DIAS': 20
    }

    # PARÁMETROS FÍSICOS DEL MODELO
    PARAMETROS_MODELO = {
        'g': 9.81,
        'bc': 25,       # Valor por defecto, se sobreescribe con datos reales.
        'hc': 0.79,     # Profundidad aprox 2024-07: 0.7908 +- 0.3930 mts. máximo: 1.4173 mts.
        'Lc': 102,      # Largo aprox 102 +- 8 mts.
        'n': 0.022,     # Ven Te Chow, 1973. canal de tierra excabado, limpio, recto.
        'ZL0': 0.25,    # De los Datos de terreno para la fecha y hora de inicio.
    }
    
    # =============================================================================
    # EJECUCIÓN PRINCIPAL
    # =============================================================================
    
    # Extraer configuraciones
    PASO_TEMPORAL_MINUTOS = CONFIG_TEMPORAL['PASO_TEMPORAL_MINUTOS']
    FECHA_INICIO = CONFIG_TEMPORAL['FECHA_INICIO']
    DURACION_DIAS = CONFIG_TEMPORAL['DURACION_DIAS']
    
    ARCHIVO_CAUDAL = CONFIG_ARCHIVOS['ARCHIVO_CAUDAL']
    ARCHIVO_MAREA = CONFIG_ARCHIVOS['ARCHIVO_MAREA']
    ARCHIVO_BATIMETRIA = CONFIG_ARCHIVOS['ARCHIVO_BATIMETRIA']
    ARCHIVO_ANCHOS_NC = CONFIG_ARCHIVOS['ARCHIVO_ANCHOS_NC']
    DIRECTORIO_RESULTADOS = CONFIG_ARCHIVOS['DIRECTORIO_RESULTADOS']
    
    print("════════════════════════════════════════════════════════════════")
    print("MODELO DE LAGUNA COSTERA - DATOS REALES DE ANCHO")
    print("ECUACIONES GOBERNANTES EXACTAS DEL DOCUMENTO LATEX")
    print("SIMULACIÓN POR INTERVALOS (solo donde hay datos de ancho, sin extensión)")
    print("════════════════════════════════════════════════════════════════")
    print(f"• Paso temporal: {PASO_TEMPORAL_MINUTOS:.2f} minutos")
    print(f"• Duración: {DURACION_DIAS:.2f} días")
    print(f"• Fecha inicio: {FECHA_INICIO}")
    print(f"• Total de pasos (si fuera continuo): {int(DURACION_DIAS * 24 * 60 / PASO_TEMPORAL_MINUTOS):,}")
    print(f"• Directorio resultados: {DIRECTORIO_RESULTADOS}")
    print(f"• Archivo de anchos REALES: {ARCHIVO_ANCHOS_NC}")
    print("════════════════════════════════════════════════════════════════")
    
    # Verificar que existe el archivo de anchos
    if not os.path.exists(ARCHIVO_ANCHOS_NC):
        print(f"   No se encuentra el archivo de anchos NetCDF: {ARCHIVO_ANCHOS_NC}")
        print("  Usando ancho constante en su lugar.")
        ARCHIVO_ANCHOS_NC = None
    
    # Crear directorio de resultados si no existe
    if not os.path.exists(DIRECTORIO_RESULTADOS):
        os.makedirs(DIRECTORIO_RESULTADOS)
        print(f"   Directorio creado: {DIRECTORIO_RESULTADOS}")
    
    try:
        # Inicializar modelo CON DATOS REALES DE ANCHO
        print("\n════════════════════════════════════════════════════════════════")
        print("INICIALIZANDO MODELO")
        print("════════════════════════════════════════════════════════════════")
        modelo = ModeloLagunaCostera(
            ARCHIVO_CAUDAL, 
            ARCHIVO_MAREA, 
            ARCHIVO_BATIMETRIA, 
            PARAMETROS_MODELO,
            archivo_anchos_nc=ARCHIVO_ANCHOS_NC
        )
        
        # Ejecutar simulación
        inicio_tiempo = time.time()
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
        
        # Análisis del caudal del río
        Qr = resultados['caudal_rio']
        print(f"\n  CAUDAL DEL RÍO (Qr):")
        print(f"    • Mínimo: {Qr.min():.6f} m³/s")
        print(f"    • Máximo: {Qr.max():.6f} m³/s")
        print(f"    • Promedio: {Qr.mean():.6f} m³/s")
        print(f"    • Desviación estándar: {Qr.std():.6f} m³/s")
        
        # Análisis de la derivada dA/dZ
        if 'derivada_area' in resultados:
            dA_dZ = resultados['derivada_area']
            print(f"\n  DERIVADA dA/dZ:")
            print(f"    • Mínimo: {dA_dZ.min():.2f} m²/m")
            print(f"    • Máximo: {dA_dZ.max():.2f} m²/m")
            print(f"    • Promedio: {dA_dZ.mean():.2f} m²/m")
            print(f"    • Desviación estándar: {dA_dZ.std():.2f} m²/m")
        
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
                print(f"      - Promedio: {anchos_abiertos.mean():.2f} m, Mediana: {np.median(anchos_abiertos):.2f} m")
        
        # Generar gráficos CON DATOS REALES DE ANCHO
        nombre_archivo = f'Resultados_modelo_lagunar_FINAL_2024-07.png'
        archivo_grafico = os.path.join(DIRECTORIO_RESULTADOS, nombre_archivo)
        modelo.graficar_resultados(resultados, archivo_grafico)
        
        # Guardar resultados en NetCDF
        nombre_netcdf = f'resultados_simulacion_variable_FINAL_2024-07.nc'
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
                'descripcion': 'Resultados de simulación de laguna costera con datos REALES de ancho - Ecuaciones exactas del documento LaTeX',
                'fecha_inicio': str(resultados['fecha_inicio']),
                'fecha_fin': str(resultados['fecha_fin']),
                'duracion_dias': resultados['duracion_dias'],
                'dt_minutes': resultados['dt_minutes'],
                'modelo': 'Laguna Costera - Runge-Kutta 4to orden con datos REALES de ancho',
                'ecuaciones_gobernantes': 'Exactas del documento LaTeX: dZ_L/dt = (Q_c + Q_r) / (Z_L * dA_L/dZ_L + A_L); dQ_c/dt = [g * A_c * (Z_m - Z_L)] / L_c - [C_d * |Q_c| * Q_c * b_c] / (A_c^2)',
                'parametros_canal': f'bc={modelo.bc:.2f}m (por defecto), hc={modelo.hc:.2f}m, Lc={modelo.Lc:.2f}m, n={modelo.n:.3f}',
                'nivel_inicial': f'{modelo.ZL0:.2f} m',
                'estados_canal': '0=cerrado, 1=abierto',
                'datos_ancho': 'Provenientes de NetCDF con reglas de interpolación específicas',
                'reglas_interpolacion': '1. Dif > 10h -> usar más cercano; 2. Ambos abiertos -> lineal; 3. Ambos cerrados -> cerrado; 4. Uno abierto/cerrado -> más cercano',
                'datos_marea': 'Interpolación directa desde datos reales de Cáhuil (sin análisis armónico)',
                'derivada_dA_dZ': 'Calculada numéricamente usando diferencia central con la batimetría',
                'metodo_numerico': 'Runge-Kutta 4to orden',
                'simulacion_por_intervalos': 'Solo se simula en períodos con datos de ancho (sin extensión)'
            }
            
            ds.to_netcdf(archivo_resultados)
            print(f"   Resultados guardados en NetCDF: {archivo_resultados}")
            
        except Exception as e:
            print(f"   Error guardando resultados NetCDF: {e}")
        
        # GUARDAR RESUMEN DETALLADO EN ARCHIVO TXT
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"GENERANDO RESUMEN DETALLADO EN ARCHIVO TXT")
        print(f"════════════════════════════════════════════════════════════════")
        
        archivo_resumen = modelo.guardar_resumen_txt(resultados, DIRECTORIO_RESULTADOS)
        fin_tiempo = time.time()
        tiempo_simulacion_seg = fin_tiempo - inicio_tiempo
        if archivo_resumen:
            with open(archivo_resumen, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("TIEMPO DE SIMULACIÓN\n")
                f.write("=" * 100 + "\n")
                f.write(f"Tiempo total de ejecución: {tiempo_simulacion_seg:.2f} segundos")
                if tiempo_simulacion_seg > 60:
                    f.write(f" ({tiempo_simulacion_seg/60:.2f} minutos)")
                f.write("\n")
        
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