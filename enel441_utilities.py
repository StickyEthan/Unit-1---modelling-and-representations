import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math


def partial_fraction_expansion_3nd_order_sys(num, den):

    poles = np.roots(den)
    regression_matrix = np.array([[1,1,1], [poles[1]+poles[2], poles[0]+poles[2], poles[0]+poles[1]], [poles[1]*poles[2], poles[0]*poles[2], poles[0]*poles[1]]])
    num_vec = np.zeros(3)
    num_vec[(3-num.shape[0]):3] = num

    coefficients_vec = np.linalg.inv(regression_matrix) @ num_vec  #note @ means matrix multiply in python
    return poles, coefficients_vec


def enel441_step_response(num, den, t):
    poles, coefficients = partial_fraction_expansion_3nd_order_sys(num, den)

    # Plot the step reponse (assume initial conditions are zero)
    out_step = coefficients[0]*np.exp(poles[0]*t) + coefficients[1]*np.exp(poles[1]*t) + coefficients[2]*np.exp(poles[2]*t)
    fig, ax = plt.subplots(1,1)
    ax.plot(t,out_step.real)
    ax.set_title(r'Step response')
    ax.set_xlabel('Time (s)')
    return fig, ax, out_step


def enel441_rise_time(t, out_step, ax):
    final_value = out_step[-1]
    
    ii = 0
    while out_step[ii] < final_value*0.1:
        ii += 1
    
    start_index = ii
    
    while out_step[ii] < final_value*0.9:
        ii += 1
        
    end_index = ii
    
    
    rise_time = t[end_index] - t[start_index]
    
    ax.plot(t[start_index], out_step[start_index], 'k.')
    ax.plot(t[end_index], out_step[end_index], 'k.')
    
    ax.plot([t[0], t[start_index]], [out_step[start_index], out_step[start_index]], 'k:')
    ax.plot([t[start_index], t[start_index]], [out_step[start_index], 0], 'k:')
    ax.plot([t[0], t[end_index]], [out_step[end_index], out_step[end_index]], 'k:')
    ax.plot([t[end_index], t[end_index]], [out_step[end_index], 0], 'k:')
    
    ax.arrow(t[start_index], out_step[start_index], rise_time, 0, head_length = 0.1, head_width = 0.025, length_includes_head = True )
    ax.arrow(t[end_index], out_step[start_index], -rise_time, 0, head_length = 0.1, head_width = 0.025, length_includes_head = True )
    
    ax.text(t[start_index] + rise_time/2, out_step[start_index] + 0.05, 'Tr', horizontalalignment='center')
    return rise_time


def enel441_peak_overshoot(t, out_step, ax):
    final_value = out_step[-1]
    
    index_peak_overshoot = np.argmax(out_step)
    peak_overshoot = out_step[index_peak_overshoot]
    peak_overshoot_percent = (peak_overshoot-final_value)/final_value*100
    
    ax.plot(t[index_peak_overshoot], peak_overshoot, 'k.')
    
    ax.plot([t[0], t[-1]], [final_value, final_value], 'k:')
    ax.plot([t[0], t[index_peak_overshoot]], [peak_overshoot, peak_overshoot], 'k:')
    ax.plot([t[index_peak_overshoot], t[index_peak_overshoot]], [peak_overshoot, 0], 'k:')
    
    ax.arrow(t[index_peak_overshoot], final_value, 0, peak_overshoot-final_value, head_length = 0.1, head_width = 0.025, length_includes_head = True )
    ax.arrow(t[index_peak_overshoot], peak_overshoot, 0, -peak_overshoot+final_value, head_length = 0.1, head_width = 0.025, length_includes_head = True )
    
    ax.text(t[index_peak_overshoot], final_value + (peak_overshoot-final_value)/2, 'PO', verticalalignment='center')
    
    return t[index_peak_overshoot], peak_overshoot_percent



def enel441_settling_time(t, out_step, ax):
    final_value = out_step[-1]
    
    ii = t.shape[0] - 1
    while out_step[ii] < 1.02*final_value and out_step[ii] > 0.98*final_value:
        ii -= 1
    
    index_settling_time = ii   
    
    ax.plot(t[index_settling_time], out_step[index_settling_time], 'k.')
    
    #uncomment to add annotations (when your function is ready)
    ax.plot([t[0], t[-1]], [0.98*final_value, 0.98*final_value], 'k:')
    ax.plot([t[0], t[-1]], [1.02*final_value, 1.02*final_value], 'k:')
    if out_step[index_settling_time] > final_value:
        ax.text(t[index_settling_time], out_step[index_settling_time], 'Ts', verticalalignment='bottom')
    else:
        ax.text(t[index_settling_time], out_step[index_settling_time], 'Ts', verticalalignment='top')
    
    return t[index_settling_time]


