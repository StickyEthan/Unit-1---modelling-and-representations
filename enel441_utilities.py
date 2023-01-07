import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math


def eval_poly(poly, k):
    n = poly.shape[0] - 1

    p_at_k = 0
    for ii in range(n+1):
        p_at_k += poly[ii]*k**(n-ii)
    
    return p_at_k


def enel441_partial_fraction_expansion(num, den):
    ## returns partial fraction expansion. This function cannot deal with repeated poles. I will fail in that case.

    poles = np.roots(den)
    num_poles = poles.shape[0]
    coeff = np.zeros(num_poles, dtype=np.csingle)

    for ii in range(num_poles):
        nn = eval_poly(num, poles[ii])
        dd = 1
        for jj in range(num_poles):
            if ii != jj:
                dd *= (poles[ii] - poles[jj])
        
        coeff[ii] = nn/dd
    return poles, coeff
            
            
def enel441_step_response(num, den, t, fig=[], ax=[]):
    if not fig: 
        fig, ax = plt.subplots(1,1)

    poles, coefficients = enel441_partial_fraction_expansion(num, den)

    # Plot the step reponse (assume initial conditions are zero)
    N = t.shape[0]
    out_step = np.zeros(N, dtype=np.csingle)
    for ii in range(poles.shape[0]):
        out_step += coefficients[ii]*np.exp(poles[ii]*t)
    
    ax.plot(t,out_step.real, label='Step Response')
    ax.set_title(r'Step response')
    ax.set_xlabel('Time (s)')

    plt.xlim(t[0], t[-1])

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


def enel441_s_plane_plot(num_sys, den_sys, fig=[], ax=[]):
    if not fig: 
        fig, ax = plt.subplots(1,1)
    
    poles_sys = np.roots(den_sys)
    for pp in poles_sys:
        ax.plot(np.real(pp), np.imag(pp), 'bx' )

    zeros_sys = np.roots(num_sys)
    for zz in zeros_sys:
        ax.plot(np.real(zz), np.imag(zz), 'ro')

    x_min1 = np.min(np.real(poles_sys))-0.5
    if zeros_sys.shape[0] > 0:
        x_min2 = np.min(np.real(zeros_sys))-0.5
    else: x_min2 = 0
    x_min3 = -0.5

    x_max1 = np.max(np.real(poles_sys))+0.5
    if zeros_sys.shape[0] > 0:
        x_max2 = np.max(np.real(zeros_sys))+0.5
    else:
        x_max2 = 0
    x_max3 = 0.5

    plt.xlim(np.min(np.array([x_min1, x_min2, x_min3])), np.max(np.array([x_max1, x_max2, x_max3])))
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.set_title('S-Plane Plot')
    ax.set_xlabel('Real[s]')
    ax.set_ylabel('Imag[s]')

    ax.xaxis.set_label_coords(0.5,-0.01)
    ax.yaxis.set_label_coords(-0.01,0.5)
    return fig, ax

def roots_to_polynomial(roots_poly):
    poly = np.array([1])
    for rr in roots_poly:
        poly = np.convolve(poly, np.array([1, -rr]))
    return np.real(poly)


def enel441_plot_step_response_bounds_per_pole(num, den, t, ax):
    N = t.shape[0]
    
    poles, coefficients = enel441_partial_fraction_expansion(num, den)
    num_poles = poles.shape[0]

    resp_poles = np.zeros((N,num_poles))
    pole_strs = []
    jj = 0
    is_step_resp = False
    for ii in range(num_poles):
        if np.abs(poles[ii]) > 1e-10:
            resp_poles[:,jj] = 2*np.abs(coefficients[ii])*np.exp(np.real(poles[ii])*t)  # bound is due to real portion of pole only
            pole_strs.append('{:.2f}'.format(poles[ii]))
            jj += 1
        else: 
            offset = np.abs(coefficients[ii])
            is_step_resp = True
    
    if is_step_resp:
        for ii in range(num_poles-1):
            ax.plot(t, resp_poles[:,ii]+offset, '--', color='C'+str(ii), label='p='+pole_strs[ii])
            ax.plot(t, -resp_poles[:,ii]+offset, '--', color='C'+str(ii))
    else:
        for ii in range(num_poles):
            ax.plot(t, resp_poles[:,ii], '--', color='C'+str(ii), label='p='+pole_strs[ii])
            ax.plot(t, -resp_poles[:,ii], '--', color='C'+str(ii))

    ax.legend()  
