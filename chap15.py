from modsim import *

def make_system(T_init, volume, r, t_end):
    return System(T_init=T_init,
                  T_final=T_init,
                  volume=volume,
                  r=r,
                  t_end=t_end,
                  T_env=22,
                  t_0=0,
                  dt=1)

from modsim import *

def change_func(T, t, system):
    r, T_env, dt = system.r, system.T_env, system.dt    
    return -r * (T - T_env) * dt

from modsim import *

def run_simulation(system, change_func):
    t_array = linrange(system.t_0, system.t_end, system.dt)
    n = len(t_array)
    
    series = TimeSeries(index=t_array)
    series.iloc[0] = system.T_init
    
    for i in range(n-1):
        t = t_array[i]
        T = series.iloc[i]
        series.iloc[i+1] = T + change_func(T, t, system)
    
    system.t_end = t_array[-1]
    system.T_final = series.iloc[-1]
    return series

