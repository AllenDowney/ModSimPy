from modsim import *

def calc_total_infected(results, system):
    s_0 = results.S[0]
    s_end = results.S[system.t_end]
    return s_0 - s_end

