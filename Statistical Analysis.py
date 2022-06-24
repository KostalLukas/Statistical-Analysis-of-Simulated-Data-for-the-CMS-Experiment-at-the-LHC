'''
CMS Rest Mass Spectra Statistical Analysis v3.2

Lukas Kostal, Leonardo Rotondi, Theo Gatward, 24.6.2022, ICL
'''

import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.stats as stats
import scipy.special as special

import STOM_higgs_tools as sht

from numba import jit
import warnings
import time as tm

#remove warning messages from numba
warnings.filterwarnings("ignore")


'''
Defining the fucntions to be used
'''


#exponential function to be used for expectation values
def expon(x, A, lmbd):
    y = A * np.exp(-x/lmbd)
    return(y)

#Gaussian function to be used for signal values
def signal(x, A_sig, mu, sigma):
    y = A_sig/(np.sqrt(2* np.pi) * sigma ) * np.exp(- (x-mu)**2/(2*sigma**2))
    return(y)
    
#function for signal and background values
def SB(x, A, lmbd, A_sig, mu, sigma):
    y = expon(x, A, lmbd) + signal(x, A_sig, mu, sigma)
    return(y)

#define function similar to get_B_chi which also includes signal
#this is done by adding parameters A_signal, mu, sigma and changing the fucntion
#get_B_expectation into get_SB_expectation and also changing the number of parameters
#from 2 to 5 since we have added 3
def get_SB_chi(vals, mass_range, nbins, A_background, lmbd, A_sig, mu, sigma):
    
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = sht.get_SB_expectation(bin_edges + half_bin_width, A_background, lmbd, mu, sigma, A_sig)
    chi = 0
    
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
        
    return chi/float(nbins-5) #should this be 5 for SB and 2 for B ??


#optimalised loop for numerical estiamtion of parameters A and lambda
@jit(parallel=False)
def loop_num(A_trial, lmbd_trial, data, interval_lim, n_bins_lim):
    
    #calcualte the initial value of chi squared and set it as min for comparison
    chi_min = sht.get_B_chi(data, interval_lim, n_bins_lim, A_trial[0], lmbd_trial[0])
    
    #loop to calcualte and compare chi squared for every combination of A_trial and lmbd_trial
    #this calcualtes the numerically estimated parameters
    for i in range(1, len(A_trial)):
        for j in range(1, len(lmbd_trial)):
            chi_trial = sht.get_B_chi(data, interval_lim, n_bins_lim, A_trial[i], lmbd_trial[j])

            if chi_trial<chi_min:
                chi_min = chi_trial
                A_num = A_trial[i]
                lmbd_num = lmbd_trial[j]
                
            print('num loop', i, j, chi_trial)
            
    return(chi_min, A_num, lmbd_num)


#parallelised loop for iterating calcualtion of reducede chi squared values
@jit(parallel=True)
def loop_chi(n_chi, interval, n_bins, A, lmbd, A_sig):
    
    #define empty array to hold calcualted values of reduced chi squared
    chi_values = np.empty(0)
    
    #calculate reduced chi squared n_chi times and record the values to see random fluctuations
    #this loop uses analytically estimated parameters A_anal, lmbd_param
    for i in range (0, n_chi):
        data = sht.generate_data(A_sig)
        data = np.array(data)
        chi = sht.get_B_chi(data, interval, n_bins, A, lmbd)
        
        chi_values = np.append(chi_values, chi)
        
        print('chi loop', i, chi)
        
        #fix for a weird issue where @jit wont let me take the appended array out of the loop
        if i == n_chi-1:
            chi_values_2 = chi_values
    
    return(chi_values_2)

#optimised loop for iterating iterated calcualtion of values of reduced chi squared
#at different signal amplitudes to determine the critical signal amplitude
@jit(parallel=False)
def loop_sig(A_sig_trial, chi_critical, n_chi, interval, n_bins, A, lmbd):
    
    #define array to hold calcualted values of reduced chi squared
    chi_expectation_values = np.empty(0)
    
    #define the values to be returned to be 0 just in case  the if loop doesnt go ahead
    A_sig_critical = 0
    chi_expectation_values_2 = np.empty(0)

    #defien as 0so we canstill output value even if if statement doesnt go ahead
    A_sig_critical = 0

    #repeat the chi loop for multiple signal amplitudes to determine the
    #minimal signal amplitude for peak to be significant
    for i in range (0, len(A_sig_trial)):
        
        #the chi loop to be repeated
        chi_values = loop_chi(n_chi, interval, n_bins, A, lmbd, A_sig_trial[i])
        
        #average out the chi values to estimate their expectation value
        chi_expectation = np.sum(chi_values) / len(chi_values)
        
        #append the current reduced chi sqaured expectation value
        chi_expectation_values = np.append(chi_expectation_values, chi_expectation)
        
        print('sig loop', i, chi_expectation)
        
        #check if the chi expectation value is greater than the critical reduced chi squared value
        if chi_expectation>chi_critical:
            
            #note the signal amplitude and break the loop
            A_sig_critical = A_sig_trial[i]
            chi_expectation_values_2 = chi_expectation_values
            break

    return(A_sig_critical, chi_expectation_values_2)


#parallelised loop for numerical estimation of mass mu
@jit(parallel=True)
def loop_mu(mu_trial, data, interval, n_bins, A, lmbd, A_sig, sigma):
    
    #define an empty array to hold reduced chi sqaured values for each trial mass
    chi_mu_values = np.empty(0)
    
    #loop over increasing trial mass and determine the reduced chi schuared value
    for i in range (0, len(mu_trial)):
        
        chi = get_SB_chi(data, interval, n_bins, A, lmbd, A_sig, mu_trial[i], sigma)
        chi_mu_values = np.append(chi_mu_values, chi)
    
        print('mu loop', i, chi)
        
        #fix for taking out the appended chi squared array
        if i == len(mu_trial)-1:
            chi_mu_values_2 = chi_mu_values
            break
        
    return(chi_mu_values_2)


'''
Defining the parameters to be used
'''


#define the number of signal events (400)
n_signals = 400

#define the numberof bins (30)
n_bins = 30

#define the interval (104, 155)
interval = [104, 155]

#define limiting value at which signal appears (121)
lim = 121

#define number of increments for numerical parameter estimation of A and lmbd (200)
n_num = 200

#define number of iterations for reduced chi squared recalculation (10000)
n_chi = 10000

#define number of bins for plotting of iterated reduced chi squared values
n_bins_chi = 40

#define number of increments for increasing A_sig (100)
n_sig_trial = 0

#define signal amplituide
A_sig = 700

#define signal mean mu in GeV/c^2
mu = 125

#define signal standard deviation sigma in GeV/c^2
sigma = 1.5

#define number of increments for numerical parameter estiamtion of mass mu (100)
n_mu = 1000


#record time when code starts running
time_code_start = tm.time()


'''
Part 1
'''


#generate the data and convert it to a numpy array
data_orig = sht.generate_data(n_signals)
data_orig = np.array(data_orig)

#histogramm the data into bins
bin_frequencies, bin_edges, patches_data = plt.hist(data_orig, range=interval, bins=n_bins, alpha=0)

#find number of datapoints in each bin
n_inbin = len(data_orig) / n_bins

#prepare array to hold standard deviations for each bin
bin_error = np.empty(0)
  
#loop over all of the bins
for i in range (0, n_bins):
    
    #select the data points in the current bin
    data_inbin = data_orig[(data_orig>bin_edges[i]) & (data_orig<bin_edges[i+1])]
    
    #find the mean value of data in the current bin
    mean = np.sum(data_inbin) / n_inbin
    
    #estimate the error for the current bin using Gaussian estimators with Bessel correction
    #this is justified by the CLT
    error = np.sqrt(np.sum((data_inbin - mean)**2) / (n_inbin-1))
    
    #append the estiated error for the current bin
    bin_error = np.append(bin_error, error)


#find the centers of bins
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

#find variables for limited data which excludes the signal
data_lim = data_orig[data_orig<=lim]
bin_frequencies_lim = bin_frequencies[bin_centers<=lim]
n_bins_lim = len(bin_frequencies_lim)
bin_centers_lim = bin_centers[:n_bins_lim]
interval_lim = [bin_edges[0], bin_edges[n_bins_lim]]


'''
Part 2
'''


#analytically estimate paramer lambda
lmbd_anal = np.sum(data_lim) / len(data_lim)

#find the bin width
bin_width = (interval[1] - interval[0])/n_bins

#find the area under the limited bins
area_bins = np.sum(bin_frequencies_lim) * bin_width

#find the area under the exponential function
area_expon = integrate.quad(lambda x: expon(x, 1, lmbd_anal), interval_lim[0], interval_lim[1])[0]

#determine area corresponding to analytically estimated lambda
A_anal = area_bins / area_expon

#define arrays of trial values of A and lambda for numerical parameter estiamtion
A_trial = np.linspace(1/2 * A_anal, 3/2 * A_anal, n_num)
lmbd_trial = np.linspace(1/2 * lmbd_anal, 3/2 * lmbd_anal, n_num)

#record time at which numerical estimation starts
time_num_start = tm.time()

#optimalised loop for numerical estiamtion of parameters A and lambda
chi_min, A_num, lmbd_num = loop_num(A_trial, lmbd_trial, data_orig, interval_lim, n_bins_lim)

#record time at which numerical estimation ends and calulate time taken
time_num_end = tm.time()
time_num = time_num_end - time_num_start


'''
Part 3
'''


#determine the reduced chi squared values for analytical and numerical parameter estimates
#for the limited dataset which only includes the background
chi_lim_anal = sht.get_B_chi(data_orig, interval_lim, n_bins_lim, A_anal, lmbd_anal)
chi_lim_num = sht.get_B_chi(data_orig, interval_lim, n_bins_lim, A_num, lmbd_num)

#determine the reduced chi squared values for analytical and numerical parameter estimates
#for the whole dataset which includes the background and peak
chi_unlim_anal = sht.get_B_chi(data_orig, interval, n_bins, A_anal, lmbd_anal)
chi_unlim_num = sht.get_B_chi(data_orig, interval, n_bins, A_num, lmbd_num)


'''
Part 4
'''


#determine the numer of degrees of freedom
dof_lim = n_bins_lim - 2
dof_unlim = n_bins - 2

#determine the alpha values for the analytical and numerical parameter estimates
#for the limited dataset which only includes the background
alpha_lim_anal = stats.chi2.sf(chi_lim_anal * dof_lim, dof_lim)
alpha_lim_num = stats.chi2.sf(chi_lim_num * dof_lim, dof_lim)

#determine the alpha values for the analytical and numerical parameter estimates
#for the whole dataset which includes the background and peak
alpha_unlim_anal = stats.chi2.sf(chi_unlim_anal * dof_unlim, dof_unlim)
alpha_unlim_num = stats.chi2.sf(chi_unlim_num * dof_unlim, dof_unlim)

#record time when iteration of reduced chi squared starts
time_chi_start = tm.time()

#parallelised loop for iterating calcualtion of reducede chi squared values
chi_values = loop_chi(n_chi, interval, n_bins, A_num, lmbd_num, 0)

#record time when iteration of reduced chi squared ends and calculate time taken
time_chi_end = tm.time()
time_chi = time_chi_end - time_chi_start

#determine the interval for the histogram of reduced chi suqared values
interval_chi = [np.floor(np.amin(chi_values)), np.ceil(np.amax(chi_values))]

#create a histogram of the set of reduced chi squared values
bin_chi_frequencies, bin_chi_edges, patches_chi = plt.hist(chi_values, range=interval_chi, bins=n_bins_chi, alpha=0)

#find the centers of bins for chi distribution
bin_chi_centers = (bin_chi_edges[:-1] + bin_chi_edges[1:])/2

#determine the mean value of the reduced chi squared distribution using Gaussian estimator
chi_distribution_expectation = np.dot(bin_chi_frequencies, bin_chi_centers) / np.sum(bin_chi_frequencies)

#determine the standard devaition of the reduced chi squared distribution using Gaussian estiamtor with Bessel correction
chi_distribution_sigma = np.sqrt(np.sum( bin_chi_frequencies * (bin_chi_centers - chi_distribution_expectation)) / (np.sum(bin_chi_frequencies-1)))

#determine the amplitude of the reduced chi squared distribution by taking the peak frequency
chi_distribution_amplitude = np.amax(bin_chi_frequencies)

#determine the reduced chi squared value required for alpha to be less than 0.05
chi_critical = stats.chi2.isf(0.05, dof_unlim) / dof_unlim

#define arrays of integer trial values of signal amplitude to be tested
A_sig_trial = np.linspace(0, n_signals, n_sig_trial).astype(int)

#record time when iteration to find critical value of signal amplitude ends
time_sig_start = tm.time()

#optimised loop for iterating iterated calcualtion of values of reduced chi squared
#at different signal amplitudes to determine the critical signal amplitude
A_sig_critical, chi_expectation_values = loop_sig(A_sig_trial, chi_critical, n_chi, interval, n_bins, A_num, lmbd_num)

#record time when iteration to find critical value of signal amplitude ends
time_sig_end = tm.time()
time_sig = time_sig_end - time_sig_start


'''
Part 5
'''


#determine the reduced chi squared values for analytical and numerical parameter estimates
#for the full data set and with an expectation value function which includes the signal
chi_SB_anal = get_SB_chi(data_orig, interval, n_bins, A_anal, lmbd_anal, A_sig, mu, sigma)
chi_SB_num = get_SB_chi(data_orig, interval, n_bins, A_num, lmbd_num, A_sig, mu, sigma)

#determine the number of degrees of freedom since we include more parameters
dof_SB = n_bins - 5

#determine the alpha values for the analytical and numerical parameter estimates
#for the full data set and with an expectation value function which includes the signal
alpha_SB_anal = stats.chi2.sf(chi_SB_anal * dof_SB, dof_SB)
alpha_SB_num = stats.chi2.sf(chi_SB_num * dof_SB, dof_SB)

#find the histogram frequencies for the signal only by subtracting the background estimate
bin_frequencies_sig_anal = bin_frequencies - expon(bin_centers, A_anal, lmbd_anal)
bin_frequencies_sig_num = bin_frequencies - expon(bin_centers, A_num, lmbd_num)

#estimate values for mass mu in GeV/c^2 by finding the peak bin
mu_bin_anal = bin_centers[np.argmax(bin_frequencies_sig_anal)]
mu_bin_anal = bin_centers[np.argmax(bin_frequencies_sig_num)]

#determine the interval which contains the peak with the background estimate subtracted
interval_peak_anal = [np.argmax(bin_frequencies_sig_anal)-3, np.argmax(bin_frequencies_sig_anal)+3]
interval_peak_num = [np.argmax(bin_frequencies_sig_num)-3, np.argmax(bin_frequencies_sig_num)+3]

#crop the frequencies to only include the ones in the interval so only the peak
bin_frequencies_peak_anal = bin_frequencies_sig_anal[interval_peak_anal[0] : interval_peak_anal[1]]
bin_frequencies_peak_num = bin_frequencies_sig_num[interval_peak_num[0] : interval_peak_num[1]]

#crop the bin center arrays to correspond to the freqeuencies above so only ones in the peak
bin_centers_peak_anal = bin_centers[interval_peak_anal[0] : interval_peak_anal[1]]
bin_centers_peak_num = bin_centers[interval_peak_num[0] : interval_peak_num[1]]

#find estimates of the signal mean value mu using Gaussian estiamtors
mu_anal = np.dot(bin_frequencies_peak_anal, bin_centers_peak_anal) / np.sum(bin_frequencies_peak_anal)
mu_num = np.dot(bin_frequencies_peak_anal, bin_centers_peak_anal) / np.sum(bin_frequencies_peak_anal)

#find estimates of the signal standard deviation using Gaussian estiamtors with Bessel correction
sigma_anal = np.sqrt(np.sum( (bin_frequencies_peak_anal * (bin_centers_peak_anal - mu_anal) )**2 )) / (np.sum(bin_frequencies_peak_anal) -1)
sigma_num = np.sqrt(np.sum( (bin_frequencies_peak_num * (bin_centers_peak_num - mu_num) )**2 )) / (np.sum(bin_frequencies_peak_num) -1)

#define arrays of integer trial values of mass mu to be tested
mu_trial = np.linspace(interval[0], interval[1], n_mu).astype(int)

#record time when iteration to numerically estimate value of mass mu starts
time_mu_start = tm.time()

#parallelised loop for numerical estimation of mass mu
chi_mu_values = loop_mu(mu_trial, data_orig, interval, n_bins, A_num, lmbd_num, A_sig, sigma)

#record time when iteration to numerically estimate value of mass mu ends
time_mu_end = tm.time()
time_mu = time_mu_end - time_mu_start

#determine a numerical estimate of the mass mu using the minimum reduced chi squared value
mu_num_num = mu_trial[np.argmin(chi_mu_values)]

#determine the alpha values for the reduced chi squared values corresponding to masses mu trial
alpha_mu_value = stats.chi2.sf(chi_mu_values * dof_SB, dof_SB)

#determine the area under the Gaussian signal distribution with an amplitude of 1
area_signal = integrate.quad(lambda x: signal(x, 1, mu_num_num, sigma), interval[0], interval[1])

#estimate the signal amplitude by considering the area under the signal peak
#and divide by the area gaussian found previously
A_sig_anal = np.sum(bin_frequencies_peak_anal) / area_signal
A_sig_num = np.sum(bin_frequencies_peak_num) / area_signal

#use the percent point function to determine the numeber of sigmas for background only hypothesis test
sigmas_anal = stats.norm.ppf(1-alpha_unlim_anal)
sigmas_num = stats.norm.ppf(1-alpha_unlim_num)

#create a plot just to plot the hystograms which have alpha set to 0 (should be empty)
plt.show()

#record time when code ends and calcualte the time taken
time_code_end = tm.time()
time_code = time_code_end - time_code_start


'''
Plotting Graphs
'''


#set fonts and font size for plotting
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 1.2

#set title and axis lables for plotting
plt.title('Rest Mass Distribution of Simulated Data')
plt.ylabel('Number of Observations')
plt.xlabel('$m_{\gamma\gamma}$ (GeV)')

#create a linear array of x values for plotting expectation values
x = np.linspace(interval[0], interval[1], 1000)

#plot the histogrammed data and expectation values using analytical and numerical estiamtes
#plt.plot(bin_centers, bin_frequencies, marker='x', linestyle='dotted', color='red', label='Data')
plt.errorbar(bin_centers, bin_frequencies, xerr=bin_width/2, yerr=bin_error, fmt='.',
             linestyle='dotted', capsize=2, color='red', label='Data', zorder=0, linewidth=1)
plt.plot(x, expon(x, A_anal, lmbd_anal), color='green', label='B Analytical', zorder=1)
plt.plot(x, expon(x, A_num, lmbd_num), color='blue', label='B Numerical', zorder=2)
plt.plot(x, SB(x, A_num, lmbd_num, A_sig, mu, sigma), color='darkviolet', label='S + B', zorder=3)

#create a plot legend
plt.legend(loc = 1)

#save the plot
plt.savefig('Output/Rest Mass Distribution.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#set title and axis lables for plotting
plt.title('Reduced Chi Squared ${\chi_{r}}^2$ Distribution')
plt.ylabel('Frequency')
plt.xlabel('${\chi_{r}}^2$ (Unitless)')

#linear array of reduced chi squared values for plotting of theoretical curve
x_chi = np.linspace(0, np.ceil(np.amax(bin_chi_centers)), 1000)

#calcualte theoretical reduced chi squared distribution for the given number of degrees of freedom
chi_theoretical = stats.chi2.pdf(x_chi * dof_unlim , dof_unlim)

#scale the theoretical distribution to fit the number of observations
chi_theoretical = np.amax(bin_chi_frequencies)/np.amax(chi_theoretical) * chi_theoretical

#plot the histogrammed values of chi squared
plt.plot(x_chi, chi_theoretical, color='blue',zorder=3, label='Theoretical')
plt.plot(bin_chi_centers, bin_chi_frequencies, color='black', zorder=2, label='Experimental')
plt.hist(chi_values, range=interval_chi, bins=n_bins_chi, color='limegreen', edgecolor='forestgreen', zorder=1)

#create plot legend
plt.legend(loc = 1)

#save the plot
plt.savefig('Output/Chi Squared Distribution.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#set title and axis lables for plotting
plt.title('Mean Chi Squared $E({\chi_{r}}^2)$ Against Signal Amplitude')
plt.ylabel('$E({\chi_{r}}^2)$ (Unitless)')
plt.xlabel('Signal Amplitude (Unitless)')

#plot expectationvalue of reduced chi squared against signal amplitude up to the critical vlaue
plt.plot(A_sig_trial[:len(chi_expectation_values)],chi_expectation_values, color='black')

#save the plot
plt.savefig('Output/Expectation Chi Squared.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#set title and axis lables for plotting
plt.title('Higgs Decay Signal With Expected Background Removed')
plt.ylabel('Number of Observations')
plt.xlabel('$m_{\gamma\gamma}$ (GeV)')

#plot the signal after expected background subtraction
plt.plot(bin_centers, bin_frequencies_sig_anal, marker='x', linestyle='dotted', color='blue', label='S subtracted B Analytical' )
plt.plot(bin_centers, bin_frequencies_sig_num, marker='x', linestyle='dotted', color='green', label='S subtracted B Numerical')
plt.plot(x, signal(x, A_sig, mu, sigma), color='darkviolet', label='S Numerical')

#create a plot legend
plt.legend(loc = 1)

#save the plot
plt.savefig('Output/Signal Removed Background.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


#set title and axis lables for plotting
plt.title('Reduced Chi Squared ${\chi_{r}}^2$ Against Expected Mass $\mu$')
plt.ylabel('${\chi_{r}}^2$ (Unitless)')
plt.xlabel('$\mu$ (GeV)')

#plot the reduced chi squared values against the expected mass 
plt.plot(mu_trial, chi_mu_values, color='black')

#save the plot
plt.savefig('Output/Chi Squared Against Mass.png', bbox_inches='tight', dpi=300)
plt.show(block=False)

#set title and axis lables for plotting
plt.title('Alpha Value $\\alpha$ Against Expected  $\mu$')
plt.ylabel('$\\alpha$ (Unitless)')
plt.xlabel('$\mu$ (GeV)')

#plot the alpha values against the expected mass 
plt.plot(mu_trial, alpha_mu_value, color='black')

#save the plot
plt.savefig('Output/Alpha Values Against Mass.png', bbox_inches='tight', dpi=300)
plt.show(block=False)


'''
Printing Numerical Data
'''


print('------------------------------------------------------------------------')
print()
print('Times taken to perform sections of the program:')
print('Numerical parameter estiamtion of A and lambda       = %.0f s' % (time_num))
print('Iterated calculation of reduced chi squared          = %.0f s' % (time_chi))
print('Numerical estimation of critical signal amplitude    = %.0f s' % (time_sig))
print('Numerical parameter estimation of mass mu            = %.0f s' % (time_mu))
print()
print('Time taken to complete the data analsysi             = %.0f s' % (time_code))
print()
print('------------------------------------------------------------------------')
print()

print('Estimated parameters for the background:')
print('Analytical estiamte of lambda                        = ', lmbd_anal)
print('Numerical estiamte of lambda                         = ', lmbd_num)
print('Analytical estimate of amplitude                     = ', A_anal)
print('Numerical estimate of amplitude                      = ', A_num)
print()
print('------------------------------------------------------------------------')
print()

print('Values obtained for H0: B only and Data: B only')
print('Reduced chi squared using analytical parameters      = %.6g' % (chi_lim_anal))
print('Reduced chi squared using numerical parameters       = %.6g' % (chi_lim_num))
print('Alpha value using analytical parameters              = %.6g' % (alpha_lim_anal))
print('Alpha value using numerical parameters               = %.6g' % (alpha_lim_num))
print()
print('Values obtained for H0: B only and Data: B + S')
print('Reduced chi squared using analytical parameters      = %.6g' % (chi_unlim_anal))
print('Reduced chi squared using numerical parameters       = %.6g' % (chi_unlim_num))
print('Alpha value using analytical parameters              = %.6g' % (alpha_unlim_anal))
print('Alpha value using numerical parameters               = %.6g' % (alpha_unlim_num))
print()
print('Values obtained for H0: B + S and Data: B + S')
print('Reduced chi squared using analytical parameters      = %.6g' % (chi_SB_anal))
print('Reduced chi squared using numerical parameters       = %.6g' % (chi_SB_num))
print('Alpha value using analytical parameters              = %.6g' % (alpha_SB_anal))
print('Alpha value using numerical parameters               = %.6g' % (alpha_SB_num))
print()
print('------------------------------------------------------------------------')
print()

print('Estimated parameters for the distibution of chi squared:')
print('Amplitude                                            = %.6g' % (chi_distribution_amplitude))
print('Expectation value                                    = %.6g' % (chi_distribution_expectation))
print('Standard deviation                                   = %.6g' % (chi_distribution_sigma))
print()
print('------------------------------------------------------------------------')
print()

print('Numerically estiamted critical values at alpha = 0.05 :')
print('Critical reduced chi squared                         = %.6g' % (chi_critical))
print('Critical signal amplitude                            = %.6g' % (A_sig_critical))
print()
print('------------------------------------------------------------------------')
print()

print('Analytically estimated signal parameters:')
print('Signal amplitude using analytical parameters         = %.6g' % (A_sig_anal))
print('Signal amplitude using numerical parameters          = %.6g' % (A_sig_num))
print()
print('Mass mu using analytical background parameters       = %.6g GeV' % (mu_anal))
print('Mass mu using numerical background parameters        = %.6g GeV' % (mu_num))
print()
print('Standard deviation using analytical parameters       = %.6g GeV' % (sigma_anal))
print('Standard deviation using numerical parameters        = %.6g GeV' % (sigma_num))
print()
print('------------------------------------------------------------------------')
print()

print('Numerically estimated signal parameters:')
print('Numerically estimated mass mu                        = %.6g GeV' % (mu_num_num))
print()
print('------------------------------------------------------------------------')
print()

print('Final Significnace Levels of Observed Signal:')
print('Number of sigma using analytical parameters          = %.6g' % (sigmas_anal))
print('Number of sigma using numerical parameters           = %.6g' % (sigmas_num))
print()
print('------------------------------------------------------------------------')
print()

#finally its over