###############################################################################
# IODR_growth_rate
# 
# Dan Olson 5-19-2020
# Library for measuring growth rate from optical density data
#
# Notes on use:
# copied from IODR - LL1592 ethnol adaptation.ipynb notebook
# C:\Users\Dan\Documents\Lynd Lab research\Ctherm CBP project\high ethanol adaptation for C therm 9-30-2019\IODR - LL1592 ethanol adaptation v5.ipynb
###############################################################################

# perform required imports
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy import stats # for sliding window slope measurements

def linear_curve(t, a, b):
    """
    fit data to linear model
    """
    return a*t + b

def gompertz_curve(t, A, umax, lag, offset):
    """
    fit data to 3-parameter logistic Gompertz equation
    Modified form from Zwietering et al. 1990, "Modeling of the Bacterial Growth Curve"
    Parameters:
       t: time (hours)
       umax: maximum specific growth rate (hr^-1)
       lag: lag time
       A: log ratio of initial to final population
       offset: parameter for shifting the curve up and down
    """
    y = A * np.exp(-np.exp(((umax * np.exp(1))/(A))*(lag - t) + 1)) + offset
    return(y)
    

def growth_analysis(data, init_OD = 0.01, reliable_OD_range = (0.03, 1), peak_distance = 10, smoothing_window = 10, peak_prominence = 0.005, show_graphs = True, epsilon = 0.1):
    """
    data: a Pandas dataframe with the following columns:
        OD: absorbance data at 600 nm
        etime: elapsed time in days
    init_OD: initial OD. For a 1:100 dilution of a OD=1 culture, the init_OD value would be 0.01
    reliable_OD_range: tuple (min, max) giving the minimum and maximum OD values that are considered reliable
    smoothing_window: number of points to use for smoothing data
    show_graphs: boolean flag to show graphs of curve fits
    epsilon: error term for bounds when fitting fixed parameters to Gompertz curve
    
    Return a Pandas series with the following information:
        maxOD
        umax_gompertz: maximum growth rate as determined by Gompertz curve fit
        umax_gompertz_err: umax standard error from Gompertz fit
        umax_slope: maximum growth rate as determined by slope of log-transformed data
        umax_slope_err: emax standard error from slope fit
    """
    
    # set elapsed time to hours
    data['etime'] = data['etime']*24 # convert days to hours
    
    # smooth data to eliminate outliers
    data['smooth'] = data.OD.rolling(smoothing_window, center = True).mean()
    
    
    # determine min, max and midpoint of data
    minOD = data.smooth.min()
    maxOD = data.smooth.max()
    midOD = (maxOD - minOD)/2 + minOD
    
    # adjust OD so that minOD = init_OD
    data.OD = data.OD - minOD + init_OD
    data.smooth = data.smooth - minOD + init_OD
    
    # recalculate min and max OD
    minOD = data.smooth.min()
    maxOD = data.smooth.max()
    
    # determine midpoint crossings
    data['nextOD'] = data['smooth'].shift(-1) # column with the OD value of the subsequent timepoint
    data['cross'] = ((data.smooth <= midOD) & (data.nextOD > midOD))
    
    if data['cross'].sum() == 0:
        print('WARNING: no midpoint crossings')
        return # we can't do any more calculations, so return
    else:
        if data['cross'].sum() >= 2: 
            print('WARNING: more than 1 midpoint crossing')

        # find the index of the first crossing, if there are more than one    
        cross_idx = data.loc[data.cross, :].sort_values('etime', ascending = True).index[0] 


    # find the peak OD
    # the logistic function we're going to use can't account for decreasing OD
    peaks = find_peaks(data.smooth, 
                       height = midOD, # peak height must be above the midpoint OD 
                       distance = peak_distance, # if there are several peaks close together, just take the largest one
                       prominence = peak_prominence,
                      )[0]
    # if there are no peaks, use all of the data
    if len(data.iloc[peaks]) == 0:
        peak_idx = data.index[-1] # set the peak index to the last point of the dataframe
    else:
        peak_idx = data.iloc[peaks].index[0]
    
    # find troughs
    troughs = find_peaks(data.smooth*-1, 
                       height = midOD*-1, # peak height must be above the midpoint OD 
                       distance = peak_distance, # if there are several peaks close together, just take the largest one
                       prominence = peak_prominence,
                      )[0]
    # select the last trough before the midpoint crossing
    troughDf = data.iloc[troughs, :] # dataframe with just the trough points
    before_crossing = troughDf.index < cross_idx # boolean filter for points before crossing
    
    # if there are no troughs before the midpoint crossing, use all data points before the crossing
    if len(troughDf.loc[before_crossing, 'etime']) < 1:
        trough_idx = data.index[0]
    else:
        trough_idx = troughDf.loc[before_crossing, 'etime'].index[-1] # get the last index in the dataframe

    
    #print('trough_idx=', trough_idx)
    #print('cross_idx=', cross_idx)
    #print('peak_idx=', peak_idx)

    # select data for fitting curve
    # use the data from the first trough before the midpoint crossing to the first peak after the midpoint crossing
    data['selected'] = False
    data.loc[trough_idx:peak_idx, 'selected'] = True 
    data2 = data.loc[data['selected'], ['OD', 'etime']].copy()
    
    # use only the data in the reliable OD range
    data2 = data2.loc[data2.OD.between(*reliable_OD_range)]
    
    # log transform and drop non-plottable values    
    data2['lnOD'] = (data2['OD'].apply(np.log))
    data2 = data2.replace([np.inf, -np.inf], np.nan)
    data2 = data2.dropna()

    # perform non-linear curve fit
    A_init = (np.log(maxOD) - np.log(minOD)) # the "height" of the original data, from min to max
    umax_init = 0.25
    lag_init = data2.iloc[0].loc['etime']
    offset_init = np.log(minOD)
    p0 = [A_init, umax_init, lag_init, offset_init] # initial guess for A, umax, lag, offset
    #print('min=', data2.iloc[0].loc['etime'])
    #print('max=', data2.iloc[-1].loc['etime'])
    #print('p0= ', p0)
    try:

        popt, pcov = curve_fit(gompertz_curve, 
                               data2['etime'], # elapsed time (hours)
                               data2['lnOD'],  # log-transformed OD data
                               p0,             # initial guess    
                               method = 'trf',
                               bounds = ((A_init-epsilon, 0, 0,      offset_init-epsilon), 
                                         (A_init+epsilon, 1, np.inf, offset_init+epsilon)),
                              )
        gomp_x = np.linspace(data['etime'].min(), data['etime'].max(), 50)
        gomp_y = gompertz_curve(gomp_x, *popt)
        perr = np.sqrt(np.diag(pc))
    except:
        #print('exception')
        #return 
        raise
    
    # perform linear curve fit on sliding window
    fit_window = int(smoothing_window/2) # fit_window needs to be an integer that is half the size of the smoothing window
    data2['umax_slope'] = 0
    data2['umax_slope_err'] = 0
    data2['icept'] = 0
    for index, row in data2.iloc[fit_window:-fit_window].iterrows():
        data3 = data2.loc[index-window:index+window]
        slope, intercept, r_value, p_value, std_err = stats.linregress(data3.etime, data3.lnOD)
        #print(slope, ' ', std_err)
        data2.loc[index, 'u'] = slope
        data2.loc[index, 'u_err'] = std_err
        data2.loc[index, 'icept'] = intercept
    
    umax_index = data2.loc[data2.u == data2.u.max(), :].index[0]
    # make a dataframe with the points used for the linear fit, for plotting
    data3 = data2.loc[umax_index-window:umax_index+window]
    lin_x = np.linspace(data3.etime.min(), data3.etime.max(), 10)
    lin_y = linear_curve(lin_x, data2.loc[umax_index, 'u'], data2.loc[umax_index, 'icept'])
    
    # prepare series for return values
    result_dict = {'maxOD': maxOD,
                   'umax_gompertz': popt[1],              
                   'umax_gompertz_err': perr[1],  
                   'umax_slope': data2.loc[umax_index, 'u'],
                   'umax_slope_err': data2.loc[umax_index, 'u_err']}
    result_ser = pd.Series(result_dict)
    
    # plot the result
    if(show_graphs):
        # set up figure
        fig, (ax1, ax3, ax2) = plt.subplots(1, 3, sharex =False, figsize = (20,8))
        
        # First panel
        ax1.set_title('initial data')
        ax1.axhline(minOD, linestyle = "--", color = 'red', alpha = 0.5, label = 'min')
        ax1.axhline(midOD, linestyle = "--", color = 'red', alpha = 0.5, label = 'mid')
        ax1.axhline(maxOD, linestyle = "--", color = 'red', alpha = 0.5, label = 'max')
        ax1.plot(data['etime'], data['OD'], label = 'OD', marker = '.')
        
        ax1.scatter(data.etime.iloc[peaks], data.OD.iloc[peaks], label = 'peaks', marker = 'o', color = 'green', s = 100)
        ax1.scatter(data.etime.iloc[troughs], data.OD.iloc[troughs], label = 'troughs', marker = 'o', color = 'red', s = 100)
        ax1.scatter(data.etime.loc[cross_idx], data.OD.loc[cross_idx], label = 'midpoint rising cross', marker = 'x', color = 'green', s = 100)
        ax1.plot(data2.etime, data2.OD, color = 'orange', label = 'good points', linewidth = 12, alpha = 0.2)
        ax1.legend()
        
        # Middle panel
        ax3.set_title('smoothed data')
        ax3.plot(data['etime'], data['smooth'], label = 'smooth', color = 'brown')
        
        # Third panel
        ax2.set_title('log-transformed data')
        ax2.axhline(np.log(minOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'min')
        ax2.axhline(np.log(midOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'mid')
        ax2.axhline(np.log(maxOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'max')
        ax2.plot(data2['etime'], data2['lnOD'], label = 'log-OD', marker = '.')
        ax2.plot(gomp_x, gomp_y, label = 'gompertz fit', color = 'red', alpha = 0.5, linewidth = 3)
        ax2.plot(lin_x, lin_y, label = 'linear fit', color = 'green', alpha = 0.5, linewidth = 6)
        ax2.legend()

        #print('A, umax, lag, offset')
        #print(popt)
        #print('minOD, midOD, maxOD')
        #print(",".join("{:.2f}".format(x) for x in [minOD, midOD, maxOD]))
        plt.show()
    
    return result_ser
