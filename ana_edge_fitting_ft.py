import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.optimize import curve_fit
from scipy import special


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean)**2 / standard_deviation ** 2))

def make_graphs(tracks, output_dir, title):
    vals, bin_edges = np.histogram(tracks, 30,(1,5))
    #print(vals)
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_middles, vals, np.sqrt(vals),  4.0/20.0,fmt='o',color='g')
    plt.title('Hist')
    plt.xlabel('randoms values')
    plt.savefig(f"{output_dir}/{title}.png")
    plt.clf()


if __name__ == '__main__':
    
    output_dir = "hist_ana_20210510"
    os.makedirs(f'{output_dir}', exist_ok=True)

    list_cordinate_x = np.random.uniform(0,5,20000)
    list_cordinate_x.sort()
    print('list_cordinate_x',list_cordinate_x)
    print('len of list_cordinate_x is: ',len(list_cordinate_x))

    make_graphs(list_cordinate_x, output_dir, "hist_ran_numbers_01")
    nums = []  
    mu = 0
    sigma = 0.2
        
    for i in range(20000):  
        temp = random.gauss(mu, sigma) 
        nums.append(temp)  
            
    print('nums',nums)
    print('len of num: ',len(nums))
    list_cordinate_x_err = []
    for p,m in zip(list_cordinate_x,nums):
        list_cordinate_x_er = p + m
        list_cordinate_x_err.append(list_cordinate_x_er)
    print('list_cordinate_x_err',list_cordinate_x_err)
    print('len of list_cordinate_x_err: ',len(list_cordinate_x_err))
    make_graphs(list_cordinate_x_err, output_dir, "hist_ran_numbers_with_err_01")
    bin_heights, bin_edges = np.histogram(list_cordinate_x_err, 30,(0,5))
    #print(vals)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, bin_heights, np.sqrt(bin_heights),  4.0/20.0,fmt='o', ecolor='g', color='blue')
    
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    print(popt)
    x_interval_for_fit = np.linspace(bin_edges[0], bin_edges[-1], 20000)
    plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='r', label='fit: mean=%5.3f, amp=%5.3f, std=%5.3f' % tuple(popt))
    plt.legend()
    plt.title('fitting')
    plt.xlim(-1,5.5)
    plt.xlabel('data_ran_num')
    plt.ylabel('Counts')
    plt.savefig(f"{output_dir}/fitting_ran_data.png")
    plt.show()
    plt.clf()
    plt.plot(list_cordinate_x_err, special.erf(list_cordinate_x_err))
    plt.text(1.1,0,f"sigma = {np.std(special.erf(list_cordinate_x_err)):2.2f}", size=10)
    print('special.erf(list_cordinate_x_err)',np.std(special.erf(list_cordinate_x_err)))
    plt.xlabel('list_cordinate_vals')
    plt.ylabel('$erf(x)$')
    plt.xlim(-1,6)
    plt.savefig(f"{output_dir}/error_function.png")
    plt.show()
    plt.clf()



 






