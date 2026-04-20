# -*- coding: utf-8 -*-
# MODELO SIMPLIFICADO DE LAGUNA COSTERA - CON ANCHO CONSTANTE (bc=25) Y SIMULACIÓN CONTINUA
# Se mantienen todas las demás características del código original:
# Qr y Zm interpolados directamente, Z_L calculado con batimetría real, ecuaciones LaTeX,
# gráficos con orden Z_m+Z_L, bc, Qc, Ac, Qr, V_L, y resumen detallado en TXT.

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime
import os
import time  # para medir tiempos de procesamiento

class ModeloLagunaCostera:
    def __init__(self, archivo_caudal, archivo_marea, archivo_batimetria, parametros, archivo_anchos_nc=None):
        """
        Inicializa el modelo con datos de caudal, marea y batimetría.
        El ancho de boca es constante (tomado de parametros['bc']).
        El argumento archivo_anchos_nc se ignora (se mantiene por compatibilidad).
        """
        # Parámetros físicos desde el diccionario de configuración
        self.g = parametros['g']      # Aceleración de gravedad (m/s²)
        self.bc = parametros['bc']    # Ancho de la boca (m) - valor constante
        self.hc = parametros['hc']    # Profundidad en desembocadura (m)
        self.Lc = parametros['Lc']    # Longitud del canal (m)
        self.n = parametros['n']      # Coeficiente de Manning
        self.ZL0 = parametros['ZL0']  # Nivel inicial de la laguna (m)
        
        # Indicador de ancho constante (para compatibilidad con código existente)
        self.bc_variable = False
        print(f"   Usando ancho constante del canal: {self.bc} m")
        
        # Cargar datos externos
        self.cargar_datos_caudal(archivo_caudal)
        self.cargar_datos_marea(archivo_marea)
        self.cargar_batimetria_real(archivo_batimetria)

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
        
        # Verificar solapamiento con marea
        if sim_end < marea_start or sim_start > marea_end:
            raise ValueError(f"El rango de la simulación no se solapa con el de la marea. Ajusta FECHA_INICIO o DURACION_DIAS.")
        
        print("   Coherencia temporal verificada: todos los datos cubren el período de simulación.\n")
    
    # ------------------------------------------------------------
    # Métodos de carga de datos (sin cambios)
    # ------------------------------------------------------------
    def cargar_datos_caudal(self, archivo):
        """Carga y procesa datos de caudal desde NetCDF"""
        try:
            ds = xr.open_dataset(archivo)
            self.caudal_rio = ds.caudal.values
            self.tiempo_caudal = ds.time.values
            
            tiempos_segundos = self.tiempo_caudal.astype('datetime64[s]').astype('int64')
            self.caudal_interp = interp1d(tiempos_segundos, self.caudal_rio, 
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value=(self.caudal_rio[0], self.caudal_rio[-1]))
            
            print(f"   Datos de caudal cargados:")
            print(f"    • Registros: {len(self.tiempo_caudal)}")
            print(f"    • Rango temporal: {pd.to_datetime(self.tiempo_caudal[0]).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(self.tiempo_caudal[-1]).strftime('%Y-%m-%d %H:%M')}")
            print(f"    • Rango de caudal: {self.caudal_rio.min():.2f} a {self.caudal_rio.max():.2f} m³/s")
            print(f"    • Promedio: {self.caudal_rio.mean():.2f} m³/s")
        except Exception as e:
            print(f"   ERROR cargando datos de caudal: {e}")
            raise
        
    def cargar_datos_marea(self, archivo):
        """Carga y procesa datos de marea desde NetCDF."""
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
            
            tiempo_segundos = ds.time.values
            tiempo_dt = pd.to_datetime(tiempo_segundos, unit='s').to_numpy().astype('datetime64[s]')
            
            if len(tiempo_dt) < 2:
                raise ValueError(f"El archivo de marea tiene solo {len(tiempo_dt)} punto(s). Se necesitan al menos 2.")
            
            self.marea_data = marea_original
            self.tiempo_marea = tiempo_dt
            
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
        except Exception as e:
            print(f"   ERROR cargando batimetría: {e}")
            raise
    
    # ------------------------------------------------------------
    # Métodos de cálculo de volumen, área, derivada (sin cambios)
    # ------------------------------------------------------------
    def calcular_volumen_directo(self, Z):
        try:
            diferencia = Z - self.batimetria
            volumen = np.sum(np.where(diferencia > 0, diferencia, 0)) * self.area_celda
            return max(volumen, 0.0)
        except Exception:
            return 0.0
    
    def calcular_area_superficial(self, Z):
        try:
            area_superficial = np.sum(np.where(Z > self.batimetria, 1, 0)) * self.area_celda
            return max(area_superficial, 1.0)
        except Exception:
            return 1.0
    
    def calcular_derivada_area_nivel(self, Z, epsilon=1e-3):
        try:
            A_L_plus = self.calcular_area_superficial(Z + epsilon)
            A_L_minus = self.calcular_area_superficial(Z - epsilon)
            dA_dZ = (A_L_plus - A_L_minus) / (2 * epsilon)
            return max(dA_dZ, 0.0)
        except Exception:
            A_L = self.calcular_area_superficial(Z)
            return A_L / 10.0
    
    # ------------------------------------------------------------
    # Interpolación de caudal y marea (sin cambios)
    # ------------------------------------------------------------
    def caudal_rio_interp(self, t):
        try:
            t_seconds = t.astype('datetime64[s]').astype('int64')
            caudal_interp = float(self.caudal_interp(t_seconds))
            return max(caudal_interp, 0.0)
        except Exception:
            return 0.0
        
    def marea_interp(self, t):
        try:
            t_seconds = t.astype('datetime64[s]').astype('int64')
            marea_interp = float(self.marea_interp_func(t_seconds))
            return np.clip(marea_interp, -2.0, 2.0)
        except Exception:
            return 0.0
    
    # ------------------------------------------------------------
    # MÉTODOS MODIFICADOS PARA ANCHO CONSTANTE
    # ------------------------------------------------------------
    def obtener_ancho_estado_real(self, t):
        """
        Versión simplificada para ancho constante.
        Retorna el ancho fijo self.bc y estado siempre abierto (1).
        """
        return self.bc, 1

    def obtener_ancho_estado(self, t):
        """Wrapper para mantener compatibilidad."""
        return self.obtener_ancho_estado_real(t)

    def determinar_intervalos_simulacion(self, fecha_inicio, fecha_fin):
        """
        Para ancho constante, la simulación es continua en todo el período.
        Retorna un único intervalo.
        """
        print("  • Simulación continua en todo el período (ancho constante).")
        return [(fecha_inicio, fecha_fin)]

    # ------------------------------------------------------------
    # Ecuaciones gobernantes (sin cambios en la lógica, solo se usa self.bc y estado=1)
    # ------------------------------------------------------------
    def ecuaciones_gobernantes(self, t, y):
        Qc, ZL = y
        
        try:
            Zm = self.marea_interp(t)
            Qr = self.caudal_rio_interp(t)
            
            # Obtener ancho y estado (constantes)
            bc_actual, estado = self.obtener_ancho_estado(t)
            
            ZL = np.clip(ZL, -10, 10)
            Qc = np.clip(Qc, -1000, 1000)
            
            A_L = self.calcular_area_superficial(ZL)
            dA_dZ = self.calcular_derivada_area_nivel(ZL)
            denominador_continuidad = ZL * dA_dZ + A_L
            if abs(denominador_continuidad) < 1e-6:
                denominador_continuidad = 1e-6 if denominador_continuidad >= 0 else -1e-6
            
            if estado == 0:  # Esta rama nunca se ejecuta porque estado siempre es 1
                dQc_dt = 0.0
                if abs(denominador_continuidad) > 1e-6:
                    dZL_dt = Qr / denominador_continuidad
                    dZL_dt = np.clip(dZL_dt, -1, 1)
                else:
                    dZL_dt = 0
            else:
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
    
    # ------------------------------------------------------------
    # Método Runge-Kutta modificado para mostrar progreso cada 10k pasos
    # ------------------------------------------------------------
    def runge_kutta_4_intervalo(self, t_start, t_end, y0, dt_minutes):
        dt_segundos = dt_minutes * 60
        dt_timedelta = np.timedelta64(int(dt_segundos), 's')
        duracion_total = (t_end - t_start).astype('timedelta64[s]').astype(int)
        n_steps = int(duracion_total / dt_segundos)
        
        t_array = np.zeros(n_steps + 1, dtype='datetime64[s]')
        y_array = np.zeros((n_steps + 1, 2))
        t_array[0] = t_start
        y_array[0] = y0
        
        # Arrays para ancho y estado (constantes)
        ancho_constante = self.bc
        estado_constante = 1
        anchos = np.full(n_steps + 1, ancho_constante)
        estados = np.full(n_steps + 1, estado_constante)
        
        print(f"    Simulando intervalo completo: {pd.to_datetime(t_start)} a {pd.to_datetime(t_end)} ({n_steps} pasos)")
        print(f"    Mostrando progreso cada 10,000 pasos...")
        
        inicio_intervalo = time.time()
        paso_progreso = 10000
        
        for i in range(n_steps):
            t_current = t_array[i]
            y_current = y_array[i]
            
            # Guardar ancho y estado (ya no se obtiene de datos reales)
            # (los arrays ya están predefinidos, pero se podría actualizar si se quisiera)
            
            # Calcular coeficientes RK4
            k1 = self.ecuaciones_gobernantes(t_current, y_current)
            k2 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k1 * dt_segundos/2)
            k3 = self.ecuaciones_gobernantes(t_current + dt_timedelta/2, y_current + k2 * dt_segundos/2)
            k4 = self.ecuaciones_gobernantes(t_current + dt_timedelta, y_current + k3 * dt_segundos)
            
            y_new = y_current + (k1 + 2*k2 + 2*k3 + k4) * (dt_segundos / 6)
            
            # Límites físicos
            y_new[0] = np.clip(y_new[0], -1000, 1000)
            y_new[1] = np.clip(y_new[1], -10, 10)
            
            y_array[i+1] = y_new
            t_array[i+1] = t_current + dt_timedelta
            
            # Mostrar progreso cada 10,000 pasos
            if (i+1) % paso_progreso == 0:
                elapsed = time.time() - inicio_intervalo
                print(f"      ... paso {i+1}/{n_steps} completado (tiempo parcial: {elapsed:.2f} s)")
        
        fin_intervalo = time.time()
        print(f"    Intervalo completado en {fin_intervalo - inicio_intervalo:.2f} segundos.")
        
        return t_array, y_array, anchos, estados

    # ------------------------------------------------------------
    # Método simular (ahora trabaja con un solo intervalo)
    # ------------------------------------------------------------
    def simular(self, fecha_inicio, duracion_dias, dt_minutes):
        fecha_inicio = np.datetime64(fecha_inicio)
        fecha_fin = fecha_inicio + np.timedelta64(int(duracion_dias * 24 * 60), 'm')
        
        Qc0 = 0.0
        ZL0 = self.ZL0
        y_actual = np.array([Qc0, ZL0])
        
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"INICIANDO SIMULACIÓN CONTINUA (ANCHO CONSTANTE {self.bc} m)")
        print(f"════════════════════════════════════════════════════════════════")
        print(f"  • Fecha inicio: {pd.to_datetime(fecha_inicio).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Fecha fin: {pd.to_datetime(fecha_fin).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • Duración: {duracion_dias:.2f} días")
        print(f"  • Paso temporal: {dt_minutes:.2f} minutos")
        print(f"  • Nivel inicial laguna: {ZL0:.2f} m")
        
        self.verificar_coherencia_temporal(fecha_inicio, duracion_dias)
        
        # Determinar intervalos (ahora devuelve un único intervalo)
        intervalos = self.determinar_intervalos_simulacion(fecha_inicio, fecha_fin)
        
        tiempos_total = []
        estados_total = []
        anchos_total = []
        estados_canal_total = []
        
        tiempo_total_inicio = time.time()
        
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
        
        tiempo_total_fin = time.time()
        tiempo_total_sim = tiempo_total_fin - tiempo_total_inicio
        
        # Convertir a arrays numpy
        tiempos = np.array(tiempos_total)
        estados = np.array(estados_total)
        anchos_canal = np.array(anchos_total)
        estados_canal = np.array(estados_canal_total)
        
        self.anchos_simulacion = anchos_canal
        self.estados_simulacion = estados_canal
        
        print(f"\n  Calculando variables a graficar...")
        mareas_sim = np.array([self.marea_interp(t) for t in tiempos])
        caudales_rio = np.array([self.caudal_rio_interp(t) for t in tiempos])
        
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
            'dt_minutes': dt_minutes,
            'tiempo_simulacion_seg': tiempo_total_sim   # nuevo campo
        }
        
        print(f"\n  Tiempo total de simulación: {tiempo_total_sim:.2f} segundos.")
        
        return resultados

    # ------------------------------------------------------------
    # Métodos de graficación y guardado (sin cambios sustanciales)
    # ------------------------------------------------------------
    def graficar_resultados(self, resultados, archivo_salida=None):
        """
        Genera gráficos completos de los resultados.
        (Sin cambios respecto al original, excepto que no se dibujan puntos de ancho real)
        """
        import matplotlib.dates as mdates

        if archivo_salida:
            directorio = os.path.dirname(archivo_salida)
            if directorio and not os.path.exists(directorio):
                os.makedirs(directorio)

        fig, axes = plt.subplots(6, 1, figsize=(16, 22), sharex=True,
                                 gridspec_kw={'hspace': 0.45, 'top': 0.96, 'bottom': 0.06})
        ax1, ax2, ax3, ax4, ax5, ax6 = axes

        tiempos_datetime = [pd.to_datetime(t) for t in resultados['tiempos']]
        step_plot = max(1, len(tiempos_datetime) // 2000)

        for ax in axes:
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

        ax6.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax6.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        # Función auxiliar para insertar NaN en saltos (aunque ahora es continuo, se mantiene)
        def insert_nan_at_gaps(times, values, gap_threshold_minutes=30):
            if len(times) == 0:
                return times, values
            gap_threshold = gap_threshold_minutes
            new_times = [times[0]]
            new_vals = [values[0]]
            for i in range(1, len(times)):
                gap = (times[i] - times[i-1]) / np.timedelta64(1, 'm')
                if gap > gap_threshold:
                    new_times.append(times[i-1] + np.timedelta64(int(gap_threshold), 'm'))
                    new_vals.append(np.nan)
                    new_times.append(times[i] - np.timedelta64(int(gap_threshold), 'm'))
                    new_vals.append(np.nan)
                new_times.append(times[i])
                new_vals.append(values[i])
            return np.array(new_times), np.array(new_vals)

        # Serie continua de marea y Qr
        t_start = resultados['fecha_inicio']
        t_end   = resultados['fecha_fin']
        dt_min  = resultados['dt_minutes']
        num_steps_cont = int((t_end - t_start) / np.timedelta64(int(dt_min*60), 's')) + 1
        t_cont = np.array([t_start + i * np.timedelta64(int(dt_min*60), 's')
                           for i in range(num_steps_cont)])
        Zm_cont = np.array([self.marea_interp(t) for t in t_cont])
        Qr_cont = np.array([self.caudal_rio_interp(t) for t in t_cont])
        t_cont_dt = [pd.to_datetime(t) for t in t_cont]

        # 1) Marea y Z_L
        ax1.plot(t_cont_dt[::step_plot], Zm_cont[::step_plot],
                 'r-', linewidth=1.5, label='Nivel marea Z_m (reconstruida)', alpha=0.9)
        ax1.plot(tiempos_datetime[::step_plot], resultados['nivel_laguna'][::step_plot],
                 'b-', linewidth=1.5, label='Nivel laguna Z_L (simulado)')
        ax1.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
        ax1.set_title('NIVELES DE AGUA - Marea y Laguna', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='upper right')

        # 2) Ancho de la boca (constante)
        bc_sim = resultados['ancho_canal']
        ax2.plot(tiempos_datetime[::step_plot], bc_sim[::step_plot],
                 'y-', linewidth=1.5, label=f'Ancho constante = {self.bc} m')
        ax2.set_ylabel('bc (m)', fontsize=12, fontweight='bold')
        ax2.set_title('ANCHO DE LA DESEMBOCADURA (constante)', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='upper right')

        # 3) Caudal Qc
        Qc = resultados['caudal_boca']
        ax3.plot(tiempos_datetime[::step_plot], Qc[::step_plot],
                 'g-', linewidth=1.5, label=f'Qc (min={np.min(Qc):.2f}, max={np.max(Qc):.2f} m³/s)')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
        ax3.set_ylabel('Qc (m³/s)', fontsize=12, fontweight='bold')
        ax3.set_title('CAUDAL DE LA DESEMBOCADURA Qc', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11, loc='upper right')

        # 4) Área transversal Ac
        ZL = resultados['nivel_laguna']
        Zm = resultados['marea']
        bc = resultados['ancho_canal']
        Ac = bc * (self.hc + np.abs(ZL - Zm) / 2.0)
        ax4.plot(tiempos_datetime[::step_plot], Ac[::step_plot],
                 'm-', linewidth=1.5, label='Área transversal Ac')
        ax4.set_ylabel('Ac (m²)', fontsize=12, fontweight='bold')
        ax4.set_title('ÁREA TRANSVERSAL DEL CANAL Ac', fontsize=14, fontweight='bold', pad=15)
        ax4.legend(fontsize=11, loc='upper right')

        # 5) Caudal Qr continuo
        ax5.plot(t_cont_dt[::step_plot], Qr_cont[::step_plot],
                 'b-', linewidth=1.5, label=f'Qr (min={np.min(Qr_cont):.2f}, max={np.max(Qr_cont):.2f} m³/s)', alpha=0.8)
        ax5.set_ylabel('Qr (m³/s)', fontsize=12, fontweight='bold')
        ax5.set_title('CAUDAL DEL ESTERO NILAHUE Qr (continuo)', fontsize=14, fontweight='bold', pad=15)
        ax5.legend(fontsize=11, loc='upper right')

        # 6) Volumen V_L
        volumen_hm3 = resultados['volumen'] / 1e6
        ax6.plot(tiempos_datetime[::step_plot], volumen_hm3[::step_plot],
                 'purple', linewidth=1.5, label='Volumen laguna')
        ax6.set_ylabel('Volumen (hm³)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Fecha y Hora', fontsize=12, fontweight='bold')
        ax6.set_title('VOLUMEN DE LA LAGUNA', fontsize=14, fontweight='bold', pad=15)
        ax6.legend(fontsize=11, loc='upper right')

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
        Guarda un archivo TXT con un resumen detallado de todos los resultados.
        Se ha añadido el tiempo de simulación.
        """
        if nombre_archivo is None:
            nombre_archivo = f'resumen_resultados_simulacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        ruta_completa = os.path.join(directorio_resultados, nombre_archivo)
        
        print(f"  Guardando resumen detallado de resultados en: {ruta_completa}")
        
        try:
            with open(ruta_completa, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RESUMEN DETALLADO DE RESULTADOS DE SIMULACIÓN - LAGUNA COSTERA (ANCHO CONSTANTE)\n")
                f.write("=" * 100 + "\n\n")
                
                # Información general
                f.write("INFORMACIÓN GENERAL DE LA SIMULACIÓN\n")
                f.write("-" * 50 + "\n")
                f.write(f"Fecha de inicio: {pd.to_datetime(resultados['fecha_inicio']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fecha de fin: {pd.to_datetime(resultados['fecha_fin']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duración: {resultados['duracion_dias']:.2f} días\n")
                f.write(f"Paso temporal: {resultados['dt_minutes']:.2f} minutos\n")
                f.write(f"Total de pasos de simulación: {len(resultados['tiempos'])}\n")
                f.write(f"Nivel inicial de la laguna: {self.ZL0:.4f} m\n")
                f.write(f"Ancho del canal (constante): {self.bc:.2f} m\n")
                f.write(f"Profundidad del canal: {self.hc:.2f} m\n")
                f.write(f"Longitud del canal: {self.Lc:.2f} m\n")
                f.write(f"Coeficiente de Manning: {self.n:.4f}\n")
                f.write(f"Tiempo de simulación (segundos): {resultados.get('tiempo_simulacion_seg', 0):.2f}\n\n")
                
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
                
                ZL = resultados['nivel_laguna']
                f.write("\nNIVEL DE LA LAGUNA (ZL):\n")
                f.write(f"  Mínimo: {np.min(ZL):.6f} m\n")
                f.write(f"  Máximo: {np.max(ZL):.6f} m\n")
                f.write(f"  Promedio: {np.mean(ZL):.6f} m\n")
                f.write(f"  Mediana: {np.median(ZL):.6f} m\n")
                
                Zm = resultados['marea']
                f.write("\nMAREA (Zm):\n")
                f.write(f"  Mínimo: {np.min(Zm):.6f} m\n")
                f.write(f"  Máximo: {np.max(Zm):.6f} m\n")
                f.write(f"  Promedio: {np.mean(Zm):.6f} m\n")
                
                Qr = resultados['caudal_rio']
                f.write("\nCAUDAL DEL RÍO (Qr):\n")
                f.write(f"  Mínimo: {np.min(Qr):.6f} m³/s\n")
                f.write(f"  Máximo: {np.max(Qr):.6f} m³/s\n")
                f.write(f"  Promedio: {np.mean(Qr):.6f} m³/s\n")
                
                V = resultados['volumen']
                f.write("\nVOLUMEN DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(V):.6f} m³ ({np.min(V)/1e6:.4f} hm³)\n")
                f.write(f"  Máximo: {np.max(V):.6f} m³ ({np.max(V)/1e6:.4f} hm³)\n")
                f.write(f"  Promedio: {np.mean(V):.6f} m³\n")
                
                A = resultados['area_superficial']
                f.write("\nÁREA SUPERFICIAL DE LA LAGUNA:\n")
                f.write(f"  Mínimo: {np.min(A):.6f} m² ({np.min(A)/1e6:.4f} km²)\n")
                f.write(f"  Máximo: {np.max(A):.6f} m² ({np.max(A)/1e6:.4f} km²)\n")
                
                if 'derivada_area' in resultados:
                    dA_dZ = resultados['derivada_area']
                    f.write("\nDERIVADA dA/dZ:\n")
                    f.write(f"  Mínimo: {np.min(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Máximo: {np.max(dA_dZ):.6f} m²/m\n")
                    f.write(f"  Promedio: {np.mean(dA_dZ):.6f} m²/m\n")
                
                # Ancho y estado (constantes)
                f.write("\nANCHO DEL CANAL (constante):\n")
                f.write(f"  Valor: {self.bc:.2f} m\n")
                f.write("\nESTADO DEL CANAL:\n")
                f.write(f"  Siempre abierto (estado = 1)\n")
                
                # Tabla detallada (primeros 100 pasos)
                f.write("\n" + "=" * 100 + "\n")
                f.write("RESULTADOS DETALLADOS (primeros 100 pasos)\n")
                f.write("=" * 100 + "\n\n")
                f.write(f"{'Fecha/Hora':<25} | {'Qc (m³/s)':<12} | {'ZL (m)':<10} | {'Zm (m)':<10} | {'Qr (m³/s)':<12} | {'Vol (hm³)':<12} | {'Área (km²)':<12}\n")
                f.write("-" * 100 + "\n")
                
                tiempos = resultados['tiempos']
                for i in range(min(100, len(tiempos))):
                    fecha_str = pd.to_datetime(tiempos[i]).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{fecha_str:<25} | {Qc[i]:<12.6f} | {ZL[i]:<10.6f} | {Zm[i]:<10.6f} | {Qr[i]:<12.6f} | {V[i]/1e6:<12.6f} | {A[i]/1e6:<12.6f}\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("RESUMEN FINAL\n")
                f.write("=" * 100 + "\n")
                f.write(f"Total de registros guardados: {len(tiempos)}\n")
                f.write(f"Tiempo de simulación (segundos): {resultados.get('tiempo_simulacion_seg', 0):.2f}\n")
                f.write(f"Fecha de generación del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Fin del informe.\n")
            
            print(f"   Resumen guardado exitosamente: {ruta_completa}")
            return ruta_completa
            
        except Exception as e:
            print(f"   ERROR guardando resumen TXT: {e}")
            return None


def main():
    """Función principal de ejecución - ANCHO CONSTANTE y SIMULACIÓN CONTINUA"""
    
    # =============================================================================
    # CONFIGURACIÓN COMPLETA
    # =============================================================================
    
    CONFIG_ARCHIVOS = {
        'ARCHIVO_CAUDAL': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\03 Resultados_caudal_altura_estero\caudales_alturas_serie_temporal.nc',
        'ARCHIVO_MAREA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Data_mareas_estimadas\marea_reconstruida_utide_2023-2024.nc',
        'ARCHIVO_BATIMETRIA': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\01 Resultados_mallado_kriging\superficie_kriging.nc',
        'ARCHIVO_ANCHOS_NC': None,   # No se usan datos de ancho real
        'DIRECTORIO_RESULTADOS': r'C:\Users\Tomas\Desktop\Escritorio\memoria\03 Codigos\03 Codigo_modelo_simplificado\Resultados_modelo_lagunar_constante-FINAL_2024-07'
    }
    
    CONFIG_TEMPORAL = {
        'PASO_TEMPORAL_MINUTOS': 0.25,
        'FECHA_INICIO': '2024-07-10 08:00',
        'DURACION_DIAS': 20
    }

    PARAMETROS_MODELO = {
        'g': 9.81,
        'bc': 50,       # Valor constante en todo en tiempo.
        'hc': 0.79,     # Profundidad aprox 2024-07: 0.7908 +- 0.3930 mts. máximo: 1.4173 mts.
        'Lc': 102,      # Largo aprox 102 +- 8 mts.
        'n': 0.022,     # Ven Te Chow, 1973. canal de tierra excabado, limpio, recto.
        'ZL0': 0.25,    # De los Datos de terreno para la fecha y hora de inicio.
    }
    
    # =============================================================================
    # EJECUCIÓN PRINCIPAL
    # =============================================================================
    
    PASO_TEMPORAL_MINUTOS = CONFIG_TEMPORAL['PASO_TEMPORAL_MINUTOS']
    FECHA_INICIO = CONFIG_TEMPORAL['FECHA_INICIO']
    DURACION_DIAS = CONFIG_TEMPORAL['DURACION_DIAS']
    
    ARCHIVO_CAUDAL = CONFIG_ARCHIVOS['ARCHIVO_CAUDAL']
    ARCHIVO_MAREA = CONFIG_ARCHIVOS['ARCHIVO_MAREA']
    ARCHIVO_BATIMETRIA = CONFIG_ARCHIVOS['ARCHIVO_BATIMETRIA']
    ARCHIVO_ANCHOS_NC = CONFIG_ARCHIVOS['ARCHIVO_ANCHOS_NC']
    DIRECTORIO_RESULTADOS = CONFIG_ARCHIVOS['DIRECTORIO_RESULTADOS']
    
    print("════════════════════════════════════════════════════════════════")
    print("MODELO DE LAGUNA COSTERA - ANCHO CONSTANTE")
    print("SIMULACIÓN CONTINUA EN TODO EL PERIODO")
    print("════════════════════════════════════════════════════════════════")
    print(f"• Paso temporal: {PASO_TEMPORAL_MINUTOS:.2f} minutos")
    print(f"• Duración: {DURACION_DIAS:.2f} días")
    print(f"• Fecha inicio: {FECHA_INICIO}")
    print(f"• Total de pasos: {int(DURACION_DIAS * 24 * 60 / PASO_TEMPORAL_MINUTOS):,}")
    print(f"• Directorio resultados: {DIRECTORIO_RESULTADOS}")
    print("════════════════════════════════════════════════════════════════")
    
    # Crear directorio de resultados si no existe
    if not os.path.exists(DIRECTORIO_RESULTADOS):
        os.makedirs(DIRECTORIO_RESULTADOS)
        print(f"   Directorio creado: {DIRECTORIO_RESULTADOS}")
    
    try:
        # Inicializar modelo (sin archivo de anchos)
        print("\n════════════════════════════════════════════════════════════════")
        print("INICIALIZANDO MODELO")
        print("════════════════════════════════════════════════════════════════")
        modelo = ModeloLagunaCostera(
            ARCHIVO_CAUDAL,
            ARCHIVO_MAREA,
            ARCHIVO_BATIMETRIA,
            PARAMETROS_MODELO,
            archivo_anchos_nc=ARCHIVO_ANCHOS_NC   # None
        )
        
        # Ejecutar simulación
        resultados = modelo.simular(FECHA_INICIO, DURACION_DIAS, PASO_TEMPORAL_MINUTOS)
        
        if resultados is None:
            print("   ERROR: La simulación falló")
            return
        
        # Mostrar estadísticas básicas
        print(f"\n════════════════════════════════════════════════════════════════")
        print(f"RESULTADOS DE LA SIMULACIÓN")
        print(f"════════════════════════════════════════════════════════════════")
        print(f"• Período: {pd.to_datetime(resultados['fecha_inicio']).strftime('%Y-%m-%d %H:%M')} a {pd.to_datetime(resultados['fecha_fin']).strftime('%Y-%m-%d %H:%M')}")
        print(f"• Pasos simulados: {len(resultados['tiempos'])}")
        print(f"• Nivel laguna: min={np.min(resultados['nivel_laguna']):.3f} m, max={np.max(resultados['nivel_laguna']):.3f} m")
        print(f"• Caudal boca: min={np.min(resultados['caudal_boca']):.3f} m³/s, max={np.max(resultados['caudal_boca']):.3f} m³/s")
        
        # Generar gráficos
        nombre_grafico = f'Resultados_modelo_lagunar_constante_FINAL_2024-07.png'
        archivo_grafico = os.path.join(DIRECTORIO_RESULTADOS, nombre_grafico)
        modelo.graficar_resultados(resultados, archivo_grafico)
        
        # Guardar resultados en NetCDF
        nombre_netcdf = f'resultados_simulacion_constante_FINAL_2024-07.nc'
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
                'descripcion': 'Resultados de simulación de laguna costera con ancho CONSTANTE',
                'fecha_inicio': str(resultados['fecha_inicio']),
                'fecha_fin': str(resultados['fecha_fin']),
                'duracion_dias': resultados['duracion_dias'],
                'dt_minutes': resultados['dt_minutes'],
                'modelo': 'Laguna Costera - Runge-Kutta 4to orden, ancho constante',
                'ancho_constante': PARAMETROS_MODELO['bc'],
                'parametros_canal': f'bc={PARAMETROS_MODELO["bc"]:.2f}m, hc={PARAMETROS_MODELO["hc"]:.2f}m, Lc={PARAMETROS_MODELO["Lc"]:.2f}m, n={PARAMETROS_MODELO["n"]:.3f}',
                'nivel_inicial': f'{PARAMETROS_MODELO["ZL0"]:.2f} m',
                'estados_canal': 'siempre abierto (1)',
                'metodo_numerico': 'Runge-Kutta 4to orden',
                'simulacion': 'continua'
            }
            
            ds.to_netcdf(archivo_resultados)
            print(f"   Resultados guardados en NetCDF: {archivo_resultados}")
            
        except Exception as e:
            print(f"   Error guardando resultados NetCDF: {e}")
        
        # Guardar resumen TXT
        archivo_resumen = modelo.guardar_resumen_txt(resultados, DIRECTORIO_RESULTADOS)
        
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