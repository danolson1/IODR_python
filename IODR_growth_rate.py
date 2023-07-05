###############################################################################
# IODR_growth_rate
# 
# Dan Olson 5-19-2020
# Library for measuring growth rate from optical density data
#
# Notes on use:
# copied from IODR - LL1592 ethnol adaptation.ipynb notebook
# C:\Users\Dan\Documents\Lynd Lab research\Ctherm CBP project\high ethanol adaptation for C therm 9-30-2019\IODR - LL1592 ethanol adaptation v5.ipynb
# Dataframes with absorbance data should have a standard format where the first column is elapsed time in hours, and the second column is absorbance data
###############################################################################

# perform required imports
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy import stats # for sliding window slope measurements
from scipy.optimize import minimize_scalar # for optimization of blank value
from sklearn.linear_model import LinearRegression # for optimization of blank value
from sklearn.metrics import r2_score # for optimization of blank value
import logging

# set up logging
logger = logging.getLogger(__name__) # ensure log messages have correct module name

def test_blank_subtraction(blank_value, abs600, min_abs_ratio = 0.01, display_output = False, display_regression = False, ax = None, return_all_data = False):
    """
    Given a set of absorbance data and a blank value, subtract the blank value
    from the absorbance data, and perform a linear fit on log-transformed data. Note that the 
    absorbance data should be filtered to only include the exponential growth phase.
    Parameters:
      blank_value (float):          the blank value
      abs600 (pandas Dataframe):    a dataframe with elapsed time (hours) as the first column
                                    and raw absorbance data as the second column
                                    assume that data has been filtered to only include the log-phase region
      min_abs_ratio:                ignore log-transformed data below this ratio
      display_output (boolean):     display graphs of the log-transformed data
      display_regression (boolean): display best fit line on graph
      ax (matplotlib axes):         axes for plotting onto
      
    Return: 1-(R-squared value). The closer to zero, the better the fit
    """
    
    data = abs600.iloc[:, 0:2].dropna().copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'abs600'])
    
    # try a blank value
    data['blank'] = data['abs600'] - blank_value
    data['ln_blank'] = np.log(data['blank'])
    data.dropna(inplace = True)

    # when subtracting data 
    min_ln_value = np.log(min_abs_ratio)

    # find the first instance where values drop below the minimum value
    if data.ln_blank.min() < min_ln_value:
        final_idx = data[data.ln_blank < min_ln_value].index[-1]
        data = data.loc[data.ln_blank.index > final_idx]

    # check to make sure there's enough data to do linear regression
    logger.debug(f'test_blank_subtraction: {len(data)} rows of data for blank analysis')
    
    if len(data) <= 3:
        logger.warning('not enough data to do linear regression for blank analysis')
        return 1 # minimize function is trying to get to 0, so returning 1 tells the minimize function to go in the other direction
   
    # perform linear regression
    xdata = data.time.values.reshape(len(data), 1)
    ydata = data.ln_blank.values.reshape(len(data), 1)
    linear_regression = LinearRegression()
    linear_regression.fit(X = xdata, 
                          y = ydata, 
                         )
    ydata_pred = linear_regression.predict(xdata) # Make predictions using the testing set
    rval = r2_score(ydata, ydata_pred) # calculate the R-squared value

    if(display_output):
        if ax is None:
            fig, ax = plt.subplots() # create axes object for subsequent graphs
        
        ax.plot(xdata, ydata, color = 'r', alpha = 0.05, marker = '.', linestyle = 'None')
        if(display_regression):
            ax.plot(xdata, ydata_pred, color = "blue", linewidth = 1)
    
    if return_all_data:
        return data
    
    return (1-rval)


def optimize_blank_value(abs600, min_abs_ratio = 0.01, display_output = False):
    """
    Given a set of absorbance data, determine the blank value that best linearizes the early part of the curve
    Parameters:
      abs600 (pandas Series): dataframe with first column for time (hours) and second column for absorbance data
                              assume that data has been filtered to only include the log-phase region
      min_abs_ratio (float):  low cutoff for performing the linear fit, smaller values fit noisier data
      display_output:         display graphs of the fitting progress
      
    Return: the blank value that gave the best fit
    """
    data = abs600.iloc[:, 0:2].dropna().copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'abs600'])
    if len(data) < 1:
        logger.warning('not enough input data')
        return(0,0)
    
    if (display_output):
        fig, ax = plt.subplots() # create axes object for subsequent graphs
        my_args = (data, min_abs_ratio, True, False, ax, False)
    else:
        my_args = (data, min_abs_ratio, False, False, None, False)
    
    # determine the blank value that best linearizes the log-transformed data
    min_abs = data.abs600.rolling(10).median().min()
    res = minimize_scalar(test_blank_subtraction, 
                          args = my_args,
                          bounds = (0.0, min_abs), # don't allow any "overshoot", which could occur if a blank value > min_abs is chosen
                          method = 'Bounded',
                          #options = {'disp':3} # display detailed output of minimization routine
                         )
    logger.debug(res)
    if res.success is False:
        logger.warning('optimize_blank_value did not find a good linear fit')
        
    if (display_output):
        test_blank_subtraction(res.x, abs600, min_abs_ratio = min_abs_ratio, display_regression = True, display_output = True, ax=ax, return_all_data = False)
        
    return (res.x, 1-res.fun) # return the blank value and the R-squared value


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


def linear_fit(data_in, min_rsq = 0.98, window_size = 3, show_graphs = False, name = ''):
    """
    Find the best linear fit to a subset of experimenal data  

    Parameters:
      data_in:            (Pandas dataframe) The first column is x-data,
                                             the second column is y-data, 
                                             the third column is a boolean indicator of the data to fit
      min_rsq:            (float) The minimum R-squared value allowed for linear fits
      window_size:        (float) The time-window for each subset, in the same units of time as the 
                                  input dataframe (usually hours)
      show_graphs:        (bool) Show graphs that visually indicate how data is being selected
      name:               (string) For labeling graphs when analyzing several sets of data

    Returns: a Pandas series with fitting parameters
        umax: maximum specific growth rate
        u_err: umax error
        icept: y-intercept
        i_err: y-intercept error
        rsq: R-squared value
        time: time at umax
    """
    data = data_in.iloc[:, 0:3].copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'data', 'good'])
    good_data = data.loc[data.good == True, :]
    half_window = window_size/2
    
    # find the edges of the raw data where there's enough room for a rolling window
    min_time = data['time'].min() + half_window
    max_time = data['time'].max() - half_window
    
    # dataframe to hold eventual result values
    result = pd.Series([0,0,0,0,0,0], index = ['umax', 'u_err', 'icept', 'i_err', 'rsq', 'umax_time']).astype('float')

    if (good_data.time.max() - good_data.time.min()) < window_size:
        logger.warning(f'Data {name}: not enough data points for the specified time window')
        return result, data

    for index, row in good_data.query('time > @min_time and time < @max_time').iterrows():
        # take a subset of the raw (not good) data for linear regression
        # this allows for regression right up to the edge of the good data
        upper_limit = data['time'] > (row.time - half_window)
        lower_limit = data['time'] < (row.time + half_window)
        subset = data.loc[lower_limit & upper_limit, :]
        if len(subset) > 3: 
            lin_result = stats.linregress(subset['time'], subset['data'])
            data.loc[index, 'umax'] = lin_result.slope
            data.loc[index, 'u_err'] = lin_result.stderr
            data.loc[index, 'icept'] = lin_result.intercept # y-intercept
            data.loc[index, 'i_err'] = lin_result.intercept_stderr
            data.loc[index, 'rsq'] = lin_result.rvalue


    # identify the maximum slope
    if 'rsq' not in data:
        logger.warning(f'Data {name}: no R^2 values present')
        return result, data
    
    if len(data[data['rsq'] > min_rsq]) < 1:
        logger.warning(f'Data {name}: no linear fit above R^2 threshold found')
        return result, data

    else:
        umax_index = data.loc[data['rsq'] > min_rsq, 'umax'].idxmax()
        result = data.loc[umax_index, ['umax', 'u_err', 'icept', 'i_err', 'rsq', 'time']].astype('float')
        result.rename({'time':'umax_time'}, inplace = True)
        

        # make a dataframe with the points used for the linear fit, for plotting
        lin_x = np.linspace(result['umax_time'] - half_window, result['umax_time'] + half_window, 10)
        lin_y = linear_curve(lin_x, result.umax, result.icept)    

    if show_graphs:
        fig, ax2 = plt.subplots(1, 1, sharex =True, figsize = (8,8))
        ax2.set_title(f'{name} Linear curve fit \n umax:{result.umax:.2f}')
        ax2.plot(data['time'], data['data'], label = 'input data', color = 'brown')
        ax2.plot(data.time, data.data.mask(~data.good), label = 'good data', color = 'blue', alpha = 0.5, linewidth = 4)
        good_r2 = data.rsq.notnull() & (data.rsq > min_rsq) # find rows with good R-squared data
        ax2.plot(data.time, data.data.mask(~good_r2), label = 'good $R^2$', color = 'orange', alpha = 0.3, linewidth = 8)
        ax2.plot(lin_x, lin_y, label = 'fit', color = 'green', linewidth = 10, alpha = 0.7)
        ax2.legend()

    return result, data

    
def sigmoidal_fit(data_in, epsilon = 0.5, show_graphs = False, name = ''):
    """
    Given a set of experimental data that should follow a sigmoidal pattern, fit a sigmoidal curve to the data  

    Parameters:
      data_in:            (Pandas dataframe) The first column is x-data,
                                             the second column is y-data, 
                                             the third column is a boolean indicator of the data to fit
      show_graphs:        (bool) Show graphs that visually indicate how data is being selected
      name:               (string) For labeling graphs when analyzing several sets of data
    
    Returns: a Pandas dataframe with fitting parameters (both value and standard deviation)
      A: the dynamic range of the curve
      umax: maximum growth rate
      lag_time: lag time
      offset: offset of the initial curve from zero
    """

    data = data_in.iloc[:, 0:3].copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'data', 'good'])
    
    # perform non-linear curve fit
    A_init = (data['data'].max() - data['data'].min())  # the dynamic range of the input data
    umax_init = np.gradient(data_in['log_ratio']).max() # the maximum slope of the input data
    lag_init = 0
    offset_init = data['data'].min()
    p0 = [A_init, umax_init, lag_init, offset_init] # initial guess for A, umax, lag, offset
    try:
        popt, pcov = curve_fit(gompertz_curve, 
                               data.loc[data.good == True, 'time'], 
                               data.loc[data.good == True, 'data'],  
                               p0,             # initial guess    
                               method = 'trf',
                               bounds = ((-np.inf, 0,  0,      offset_init-epsilon), # lower bounds
                                         (np.inf,  10, np.inf, offset_init+epsilon)),# upper bounds
                               maxfev = 1000
                              )
        gomp_x = np.linspace(data['time'].min(), data['time'].max(), 50)
        gomp_y = gompertz_curve(gomp_x, *popt)
        perr = np.sqrt(np.diag(pcov)) # one standard deviation
    except:
        logger.warning(f'Data {name}: curve fitting failed')
         # raise # for debugging purposes               
             
    # compile results
    result = pd.DataFrame([popt, perr], 
                          index = ['value', 'error'], 
                          columns = ['A', 'umax', 'lag_time', 'offset']
                          ).T  
    if show_graphs:
        fig, ax1 = plt.subplots(1, 1, sharex =True, figsize = (8,8))
        ax1.set_title(f'{name} Gompertz curve fit  \n umax:{result.loc["umax", "value"]:.2f}')
        ax1.plot(data['time'], data['data'], label = 'input data', color = 'brown')
        ax1.scatter(data.loc[data.good == True, 'time'], data.loc[data.good == True, 'data'], label = 'fit data', color = 'orange', alpha = 0.3, s = 6**2)
        ax1.plot(gomp_x, gomp_y, label = 'fit', color = 'green')
        ax1.legend()            

    return result    



def flag_log_phase_data(abs600, smoothing_window = 0.2, show_graphs = False, return_all_data = False, name = ''):
    """
    Given a dataframe with raw absorbance data, determine the range of data from the minimum value
    to the point of maximum slope. This is the data where the maximum growth rate should be present
    
    Parameters:
      abs600:            Pandas dataframe with at least two columns
                           1st column - time (float) in hours
                           2nd column - raw absorbance data (float)
      smoothing_window: (float) Smoothing window, value from 0 to 1 that represents the fraction
                           of data to use for smoothing. In general, increasing this value will create 
                           a smoother curve, which causes more data to be flagged as good.
      show_graphs:      (bool) Show graphs that visually indicate how data is being selected
      return_all_data:  (bool) Return the dataframe with intermediate calculations, for troubleshooting
      name:             (string) Name of well, for plotting
      
    Return: A boolean series corresponding to the data flagged as good
    """

    data = abs600.iloc[:, 0:2].dropna().copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'data'])
    data['good_data'] = False # column to hold good data flag
    data['key_points'] = '' # column to hold information about peaks, troughs, zero crossings, etc.

    # smooth data to eliminate outliers
    smoothing_window_pts = int(len(data)*smoothing_window)
    if smoothing_window_pts >= len(data):
        smoothing_window_pts  = len(data)-1 # if the smoothing window is larger than the number of points, make it smaller
    if (smoothing_window_pts % 2) == 0:
        smoothing_window_pts += 1 # if the smoothing window has an even number of points, add one
                                  # this is important for the Savitsky-Golay filtering
    logger.info(f'smoothing_window_pts: {smoothing_window_pts}')        
    data['smooth'] = savgol_filter(data['data'], 
                                 window_length = smoothing_window_pts, 
                                 polyorder = 1, 
                                 deriv = 0,
                                 mode = 'interp')

    # differentiate data to find the region of maximum growth
    data['diff'] = np.gradient(data['smooth'])


    # find maximum and minimum values, use rolling window to filter out noise
    max_abs = data.data.rolling(10).median().max()
    min_abs = data.data.rolling(10).median().min()
    max_index = data.data.idxmax()
    min_index = data.data.idxmin()
    max_slope_index = data.query('diff == diff.max()').head(1).index[0]
    
    # identify key points
    data.loc[max_index, 'key_points'] = 'max'
    data.loc[min_index, 'key_points'] = 'min'
    data.loc[max_slope_index, 'key_points'] = 'max_slope'

    # flag data between 'min' and 'max_slope' as good
    data.loc[(data.index > min_index) & (data.index < max_slope_index), 'good_data'] = True
    logger.info(f'min_abs={min_abs}, max_abs={max_abs}')
    
    if show_graphs:    
        fig, (ax1) = plt.subplots(1, 1, sharex =True, figsize = (8,8))

        # First panel
        ax1.set_title(f'{name} raw data')
        ax1.plot(data['time'], data['data'], label = 'raw input', color = 'green')
        ax1.plot(data['time'], data['smooth'], label = 'smooth', color = 'brown')
        ax1.plot(data.loc[data.good_data, 'time'], data.loc[data.good_data, 'data'], label = 'good data', linewidth = 8, color = 'green', alpha = 0.3)
        #ax1.scatter(data.loc[zero_cross.index]['time'], data.loc[zero_cross.index]['smooth'], marker = 'o', color = 'blue', zorder = 2.5, label = 'zero crossing')
        ax1.scatter(data.loc[data.key_points != '', 'time'], data.loc[data.key_points != '', 'data'], marker = 'o', color = 'blue', zorder = 2.5, label = 'key points')
        ax1.legend()
        
    if return_all_data:
        return data
    
    return data['good_data']
    
    

def flag_sigmoidal_data(data_in, smoothing_window = 0.2, peak_height_factor = 0.75, show_graphs = False, name = ''):
    """
    Given a set of experimental data that should follow a sigmoidal pattern, identify the data that could
    in theory be modeled by a sigmoidal equation.  
    
    Parameters:
      data_in:            (Pandas dataframe) The first is time, and the second is log-transformed growth data
      smoothing_window:   (float) Smoothing window, value from 0 to 1 that represents the fraction
                          of data to use for smoothing. In general, increasing this value will create 
                          a smoother curve, which causes more data to be flagged as good.
      peak_height_factor: (float) Allows for selecting secondary peaks. Float with values of 0 to 1. 
                          Values closer to 1 select more data.
      show_graphs:        (bool) Show graphs that visually indicate how data is being selected
      name:               (string) For labeling graphs when analyzing several sets of data
      
    Returns: a Pandas boolean series indicating data that should be used for sigmoidal fitting
    """
    
    data = data_in.iloc[:, 0:2].dropna().copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'data'])
    data['good_data'] = False # column to hold good data flag
    
    # smooth data to eliminate outliers
    smoothing_window_pts = int(len(data)*smoothing_window)
    if smoothing_window_pts >= len(data):
        smoothing_window_pts  = len(data)-1 # if the smoothing window is larger than the number of points, make it smaller
    if (smoothing_window_pts % 2) == 0:
        smoothing_window_pts += 1 # if the smoothing window has an even number of points, add one
                                  # this is important for the Savitsky-Golay filtering
    logger.info(f'smoothing_window_pts: {smoothing_window_pts}')        
    data['smooth'] = savgol_filter(data['data'], 
                                 window_length = smoothing_window_pts, 
                                 polyorder = 1, 
                                 deriv = 0,
                                 mode = 'interp')
    
    # differentiate data to find the region of maximum growth
    data['diff'] = np.gradient(data['smooth'])

    diff_max = data['diff'].max()
    if np.isnan(diff_max):
        logger.warning(f'well {well_id} has no maximum value for the differentiated data: no usable data found')
        diff_max = 0
        #return data['good_data']
    
    data['key_points'] = '' # column to hold information about peaks, troughs, zero crossings, etc.
    data.loc[data['diff'].dropna().head(1).index, 'key_points'] = 'start'
    data.loc[data['diff'].dropna().tail(1).index, 'key_points'] = 'end'
    
    # the highest point is a peak (although the scipy peak finder won't choose it)
    data.loc[data['diff'].idxmax(),'key_points'] = 'peak'
    
    # find peaks and troughs
    peaks, _ = find_peaks(data['diff'], height = diff_max * peak_height_factor, distance = smoothing_window_pts) 
    data.loc[data.iloc[peaks].index,'key_points'] = 'peak'
    
    troughs, _ = find_peaks(data['diff']*-1, distance = smoothing_window_pts, height = -diff_max*(1-peak_height_factor)) 
    data.loc[data.iloc[troughs].index,'key_points'] = 'trough'
    
    # find zero crossings
    s = data['diff'].apply(np.sign).diff().dropna().abs()
    zero_cross = s[s > 0.0]
    data.loc[zero_cross.index, 'key_points'] = 'zero crossing'
    
    # select good data
    kp_df = data[data.key_points != ''].copy()
    
    # if there's more than one peak, the data may have to be manually analyzed
    if len(kp_df[kp_df.key_points == 'peak']) > 1:
        logger.warning(f'well {well_id} has more than one differential peak, data may need to be manually masked')
        
    # calculate the width of each peak
    kp_df['peak_width'] = 0 # column to hold information about peak width
    kp_df['idx_start'] = '' # columns to hold pairs of index values to mark the start and end of each peak
    kp_df['idx_end'] = '' # columns to hold pairs of index values to mark the start and end of each peak
    for index, row in kp_df[kp_df.key_points == 'peak'].iterrows():
        
        # find all of the key points (that aren't another peak) before the current peak
        before_points = kp_df[(kp_df.key_points != 'peak') & (kp_df.time < row.time)]
        # if there are no key points before the current peak
        if len(before_points) == 0:
            peak_start = row.name # set the start index to the current peak
        else:
            # find the last of these points, and save the index value
            peak_start = before_points.iloc[-1].name
        
        # find all of the key points (that aren't another peak) after the current peak
        after_points = kp_df[(kp_df.key_points != 'peak') & (kp_df.time > row.time)]
        # if there are no key points after the current peak
        if len(after_points) == 0:
            peak_end = row.name # set the start index to the current peak
        else:
            # find the first of these points, and save the index value
            peak_end = after_points.iloc[0].name
                   
        # calculate the peak width
        kp_df.loc[index, 'peak_width'] = kp_df.loc[peak_end, 'time'] - kp_df.loc[peak_start, 'time']
        kp_df.loc[index, 'idx_start'] = peak_start
        kp_df.loc[index, 'idx_end'] = peak_end
        

    # Find the widest peak, and flag that data as good
    if len(kp_df[kp_df.key_points == 'peak']) > 0:
        best_peak = kp_df.peak_width.idxmax()
        best_start_idx = kp_df.loc[best_peak, 'idx_start']
        best_end_idx = kp_df.loc[best_peak, 'idx_end']
        data.loc[best_start_idx:best_end_idx, 'good_data'] = True
    
    
    if show_graphs:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex =True, figsize = (20,8))

        # First panel
        ax1.set_title(f'{name} input data')
        ax1.plot(data['time'], data['smooth'], label = 'smooth', color = 'brown')
        ax1.plot(data['time'], data['data'], label = 'raw input', color = 'green')
        ax1.plot(data.loc[data.good_data, 'time'], data.loc[data.good_data, 'data'], label = 'good data', linewidth = 8, color = 'green', alpha = 0.3)
        ax1.scatter(data.loc[zero_cross.index]['time'], data.loc[zero_cross.index]['smooth'], marker = 'o', color = 'blue', zorder = 2.5, label = 'zero crossing')
        ax1.legend()
        
        # Second panel
        ax2.set_title(f'{name} derivative')
        ax2.axhline(0, linestyle = "--", color = 'blue', alpha = 0.5, label = 'zero crossing')
        ax2.axhline(peak_height_factor * diff_max, linestyle = "--", color = 'green', alpha = 0.5, label = 'min peak height')
        ax2.axhline((1- peak_height_factor) * diff_max, linestyle = "--", color = 'red', alpha = 0.5, label = 'max trough height')    
        ax2.scatter(data.loc[data.key_points == 'peak', 'time'], data.loc[data.key_points == 'peak', 'diff'], marker = 'o', color = 'green', zorder = 2.5, label = 'peak')
        ax2.scatter(data.iloc[troughs]['time'], data.iloc[troughs]['diff'], marker = 'o', color = 'red', zorder = 2.5, label = 'valley')
        ax2.scatter(data.loc[zero_cross.index]['time'], data.loc[zero_cross.index]['diff'], marker = 'o', color = 'blue', zorder = 2.5, label = 'zero crossing')
        ax2.plot(data.loc[data.good_data, 'time'], data.loc[data.good_data, 'diff'], label = 'good data', linewidth = 8, color = 'green', alpha = 0.3)
        ax2.plot(data['time'], data['diff'], label = 'diff', marker = '.', color = 'darkorange' )
        ax2.legend()
        
    return data['good_data']


###############################################
# Note, I'm keeping the growth_analysis function for backward compatibility, but don't use it going forward


def growth_analysis(data_in, init_OD = 0.01, reliable_OD_range = (0.03, 1), peak_distance = 10, smoothing_window = 10, peak_prominence = 0.005, show_graphs = True, epsilon = 0.1):
    """
    Find the min, max, and midpoint of data
    Find all of the locations where the data crosses the midpoint. This should only happen once. If it happens more than once, take the first crossing, and send a warning message (could be multiple growth curves in the same dataset, could be noise in the data)
    Find all of the peaks in the data set higher than the midpoint
    Find all of the troughs in the data set lower than the midpoint (invert the data and find peaks)
    From the midpoint, find the previous trough
    From the midpoint, find the next peak
    Select the region of data that includes: trough, midpoint, peak
    Fit a Gompertz curve to the data. Plot the results. Show the fit 
    
    data: a Pandas dataframe with the following columns:
        OD: absorbance data at 600 nm
        time: elapsed time in hours
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
    # make a copy of the input dataframe
    data = data_in.copy()
    
    data = data_in.iloc[:, 0:2].dropna().copy() # make a copy to avoid modifying the original dataframe
    data.columns = pd.Index(['time', 'OD'])
    data['good_data'] = False # column to hold good data flag
    
    ## set elapsed time to hours
    #data['etime'] = data['etime']*24 # convert days to hours
    
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
        logger.info('WARNING: no midpoint crossings')
        return # we can't do any more calculations, so return. This will probably cause an error because the calling function is expecting a series
    else:
        if data['cross'].sum() >= 2: 
            logger.info('WARNING: more than 1 midpoint crossing')

        # find the index of the first crossing, if there are more than one    
        cross_idx = data.loc[data.cross, :].sort_values('time', ascending = True).index[0] 


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
    if len(troughDf.loc[before_crossing, 'time']) < 1:
        trough_idx = data.index[0]
    else:
        trough_idx = troughDf.loc[before_crossing, 'time'].index[-1] # get the last index in the dataframe

    
    logger.debug(f'trough_idx={trough_idx}')
    logger.debug(f'cross_idx={cross_idx}')
    logger.debug(f'peak_idx={peak_idx}')

    # select data for fitting curve
    # use the data from the first trough before the midpoint crossing to the first peak after the midpoint crossing
    data['selected'] = False
    data.loc[trough_idx:peak_idx, 'selected'] = True 
    data2 = data.loc[data['selected'], ['OD', 'time']].copy()
    
    # use only the data in the reliable OD range
    data2 = data2.loc[data2.OD.between(*reliable_OD_range)]
    
    # log transform and drop non-plottable values    
    data2['lnOD'] = (data2['OD'].apply(np.log))
    data2 = data2.replace([np.inf, -np.inf], np.nan)
    data2 = data2.dropna()

    # perform non-linear curve fit
    A_init = (np.log(maxOD) - np.log(minOD)) # the "height" of the original data, from min to max
    umax_init = 0.25
    lag_init = data2.iloc[0].loc['time']
    offset_init = np.log(minOD)
    p0 = [A_init, umax_init, lag_init, offset_init] # initial guess for A, umax, lag, offset
    logger.debug(f"min={data2.iloc[0].loc['time']}")
    logger.debug(f"max={data2.iloc[-1].loc['time']}")
    logger.debug(f"p0 ={p0}")
    try:

        popt, pcov = curve_fit(gompertz_curve, 
                               data2['time'], # elapsed time (hours)
                               data2['lnOD'],  # log-transformed OD data
                               p0,             # initial guess    
                               method = 'trf',
                               bounds = ((A_init-epsilon, 0, 0,      offset_init-epsilon), 
                                         (A_init+epsilon, 1, np.inf, offset_init+epsilon)),
                              )
        gomp_x = np.linspace(data['time'].min(), data['time'].max(), 50)
        gomp_y = gompertz_curve(gomp_x, *popt)
        perr = np.sqrt(np.diag(pcov))
    except:
        raise
    
    # perform linear curve fit on sliding window
    fit_window = int(smoothing_window/2) # fit_window needs to be an integer that is half the size of the smoothing window
    data2['umax_slope'] = 0
    data2['umax_slope_err'] = 0
    data2['icept'] = 0
    for index, row in data2.iloc[fit_window:-fit_window].iterrows():
        data3 = data2.loc[index-fit_window:index+fit_window]
        slope, intercept, r_value, p_value, std_err = stats.linregress(data3.time, data3.lnOD)
        data2.loc[index, 'u'] = slope
        data2.loc[index, 'u_err'] = std_err
        data2.loc[index, 'icept'] = intercept
    
    umax_index = data2.loc[data2.u == data2.u.max(), :].index[0]
    # make a dataframe with the points used for the linear fit, for plotting
    data3 = data2.loc[umax_index-fit_window:umax_index+fit_window]
    lin_x = np.linspace(data3.time.min(), data3.time.max(), 10)
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
        ax1.plot(data['time'], data['OD'], label = 'OD', marker = '.')
        
        ax1.scatter(data.time.iloc[peaks], data.OD.iloc[peaks], label = 'peaks', marker = 'o', color = 'green', s = 100)
        ax1.scatter(data.time.iloc[troughs], data.OD.iloc[troughs], label = 'troughs', marker = 'o', color = 'red', s = 100)
        ax1.scatter(data.time.loc[cross_idx], data.OD.loc[cross_idx], label = 'midpoint rising cross', marker = 'x', color = 'green', s = 100)
        ax1.plot(data2.time, data2.OD, color = 'orange', label = 'good points', linewidth = 12, alpha = 0.2)
        ax1.legend()
        
        # Middle panel
        ax3.set_title('smoothed data')
        ax3.plot(data['time'], data['smooth'], label = 'smooth', color = 'brown')
        
        # Third panel
        ax2.set_title('log-transformed data')
        ax2.axhline(np.log(minOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'min')
        ax2.axhline(np.log(midOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'mid')
        ax2.axhline(np.log(maxOD), linestyle = "--", color = 'red', alpha = 0.5, label = 'max')
        ax2.plot(data2['time'], data2['lnOD'], label = 'log-OD', marker = '.')
        ax2.plot(gomp_x, gomp_y, label = 'gompertz fit', color = 'red', alpha = 0.5, linewidth = 3)
        ax2.plot(lin_x, lin_y, label = 'linear fit', color = 'green', alpha = 0.5, linewidth = 6)
        ax2.legend()

        logger.debug('A, umax, lag, offset')
        logger.debug(popt)
        logger.debug('minOD, midOD, maxOD')
        logger.debug(",".join("{:.2f}".format(x) for x in [minOD, midOD, maxOD]))
        plt.show()
    
    return result_ser
