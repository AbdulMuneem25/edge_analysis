import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.signal import correlate
import copy


def create_sin_curve(x_range, x_shift):
    amp = 1
    lam = 160
    phase = 12
    bias = 1
    inputdata = np.sin( (x - phase) * 2 * np.pi / lam ) + bias

    x_single_period = np.arange(0, lam)
    template = np.sin( x_single_period * 2 * np.pi / lam  ) + 1.0

    return inputdata, template


def create_rectangle(x_range, x_shift):
    width = 20
    inputdata = [0.0 for x in range(x_range)]
    print(inputdata[0])
    inputdata[x_shift : x_shift + width] = [1.0 for x in range(width)]
    template = [1.0 for x in range(width)]
    return inputdata, template


def create_pulse(x_range, x_shift):
    width = 100
    inputdata = [0.0 for x in range(x_range)]
    inputdata[x_shift] = 1.0
    template = [1]
    return inputdata, template



def draw_convolution_graph(inputdata, template, covlution, peaks, filename):
    fig, (ax_input_win, ax_filt) = plt.subplots(2, 1)
    #
    ax_input_win.set_title('input data')
    ax_input_win.plot(inputdata, label='input data')
    ax_input_win.plot(template, label='template', marker='o' )
    ax_input_win.grid()
    ax_input_win.legend()
    #
    ax_filt.set_title('covlution')
    ax_filt.plot(covlution)
    label_str = 'peaks: '
    for p in peaks:
        label_str += f'{p}, '
    ax_filt.plot(peaks, covlution[peaks], "x",color  = 'r', label=label_str[:-2])
    ax_filt.grid()
    ax_filt.legend()
    #
    fig.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.clf()
    return


def convolution_of_two_arrays(array1, array2):
    array2_clone = copy.deepcopy(array2)
    shift = len(array1) - len(array2)
    list_convolution = []    
    for s in range(shift):
        if s > 0:
            array2_clone = np.insert(array2_clone, 0, 0)
        valsum = 0.0
        for p,q in zip(array1, array2_clone):
            valsum += p*q
        list_convolution.append(valsum)
    return np.array(list_convolution)



if __name__ == '__main__':

    output_dir = "practice_phase_determinaition_sin_curve"
    os.makedirs(f'{output_dir}', exist_ok=True)

    x_range = 512 # 1024
    x_shift = 80
    x = np.arange(0, x_range)

    list_function = [
        {"func": create_sin_curve, "filename": "sin_curve"},
        {"func": create_rectangle, "filename": "rectangle"},
        {"func": create_pulse, "filename": "pulse"}
        ]

    list_method = ['full','valid','same']
    for f in list_function:
        inputdata, template = f["func"](x_range, x_shift)

        for m in list_method:
            covlution = signal.convolve(inputdata, template, mode=m, method='direct')
            peaks, _ = find_peaks(covlution)
            draw_convolution_graph(inputdata, template, covlution, peaks, f["filename"]+f"_{m}")

            covlution = convolution_of_two_arrays(inputdata, template)
            peaks, _ = find_peaks(covlution)
            draw_convolution_graph(inputdata, template, covlution, peaks, f["filename"]+"_myconv")



