import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.signal import correlate

if __name__ == '__main__':

    output_dir = "fft_plot"
    os.makedirs(f'{output_dir}', exist_ok=True)


    phi = np.arange(0, 6*2*np.pi, 0.01)
    
    amp = 1
    bias = 1
    phase = 0.5
    
    ft1 = np.sin(phi) + bias
    
    ft2 = amp * np.sin(phi+phase) + bias
    covlution = signal.convolve(ft1, ft2, mode='same')
    peaks, _ = find_peaks(covlution)
    print(peaks)
    #fft = signal.fftconvolve(ft1, ft2, mode='full')
    
    
    plt.plot(phi, ft1, c = 'r')
    plt.plot(phi, ft2, c = 'm')
    plt.savefig(f"{output_dir}/sin_curves_plot.png")
    plt.clf()

    fig = plt.figure(figsize=(10.0, 8.0))
    fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    ax_orig.plot(ft1)
    ax_orig.set_title('ft1')
    ax_orig.margins(0, 0.1)
    ax_win.plot(ft2)
    ax_win.set_title('ft2')
    ax_win.margins(0, 0.1)
    ax_filt.plot(covlution)
    ax_filt.set_title('covlution')
    ax_filt.margins(0, 0.1)
    #ax_filt.set_xlim(0,750)
    fig.tight_layout()
    results_half = peak_widths(covlution, peaks, rel_height=0.5)
    label_str = 'convolution peak: '
    label_str += f'peak = {peaks[0]}'
    plt.plot(peaks, covlution[peaks], "x",color  = 'r', label=label_str)
    ax_filt.legend()
    plt.savefig(f"{output_dir}/convolution_plot.png")
    plt.clf()

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(covlution)
    ax.set_title('covlution')
    #ax_filt.set_xlim(0,750)
    fig.tight_layout()
    results_half = peak_widths(covlution, peaks, rel_height=0.5)
    print('results_half',results_half)
    label_str = 'peaks: '
    label_str += f'pvals = {peaks[0]},{peaks[1]}, {peaks[2]}, {peaks[3]}, {peaks[4]}, {peaks[5]} '
    plt.plot(peaks, covlution[peaks], "x",color  = 'r', label=label_str)
    plt.hlines(*results_half[1:], color="blue")
    ax.legend()
    plt.savefig(f"{output_dir}/convolution_plot2.png")
    plt.clf()


    # fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    # ax_orig.plot(ft1)
    # ax_orig.set_title('ft1')
    # ax_orig.margins(0, 0.1)
    # ax_win.plot(ft2)
    # ax_win.set_title('ft2')
    # ax_win.margins(0, 0.1)
    # ax_filt.plot(fft)
    # ax_filt.set_title('fft')
    # ax_filt.margins(0, 0.1)
    # fig.tight_layout()
    # plt.savefig(f"{output_dir}/fft.png")
    # plt.clf()


