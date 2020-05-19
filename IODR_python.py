###############################################################################
# IODR_pathway
# 
# Dan Olson 5-19-2020
# Library for downloading data from the Internet Optical Density Reader (IODR)
#
# Notes on use:
#  - get_all_data() is the main function for getting IODR data
#  - 
#
# copied from IODR - LL1592 ethnol adaptation.ipynb notebook
# C:\Users\Dan\Documents\Lynd Lab research\Ctherm CBP project\high ethanol adaptation for C therm 9-30-2019\IODR - LL1592 ethanol adaptation v5.ipynb
###############################################################################


# perform required imports
import pandas as pd


# set ThingSpeak variables
# 3 sets of data, since there are 3 IODR devices
tsBaseUrl = r'https://api.thingspeak.com/channels'
chIDs = [405675, 441742, 469909]
readAPIkeys = ['18QZSI0X2YZG8491', 'CV0IFVPZ9ZEZCKA8', '27AE8M5DG8F0ZE44']
local_tz = pytz.timezone('US/Eastern')


def local_to_UTC(date_string):
    """
    take a local date/time string and convert to UTC date/time string
    helper function for thingspeak_to_df()
    """
    local_dt = dt.datetime.strptime(date_string, "%Y-%m-%d %H:%M").astimezone(local_tz)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt.strftime("%Y-%m-%d %H:%M")


def got_all_data(start_date, dataframe):
    """
    Check to see if the dataframe begins at the requested start date
    This is useful to see if all of the requested data is actually in the dataframe
    """
    dataframe_start = dataframe.index[0] # the first timestamp of the dataframe
    desired_start = pd.to_datetime(local_to_UTC(start_date)).tz_localize('UTC')
    time_diff = dataframe_start - desired_start
    
    if (time_diff.total_seconds() > 200):
        return False
    else:
        return True


def thingspeak_to_df(chID, start_date, end_date = dt.datetime.now().strftime("%Y-%m-%d %H:%M")):
    """
    Get data from Thingspeak, single request
    Return as a dataframe with time in UTC timezone
    If no end date is provided, use the current time as the end date
    Thinkspeak can only get 8000 datapoints per request. If more datapoints are requested, Thingspeak
      returns the most recent 8000 points. To get all of the data, you need to do a second request to 
      get the earlier data
    """
    
    start_date_UTC = local_to_UTC(start_date)
    end_date_UTC = local_to_UTC(end_date)
    print('getting data from {start} to {end}, in UTC timezone'.format(start = start_date_UTC, end = end_date_UTC))
    myUrl = 'https://api.thingspeak.com/channels/{channel_id}/feeds.csv?start={start}&end={end}'.format(
        channel_id = chID, start = start_date_UTC, end = end_date_UTC)
    print(myUrl)
    r = requests.get(myUrl)
    print('First 100 characters of HTTP request, for troubleshooting')
    display(r.content.decode('utf-8')[0:100])
    
    # put the thingspeak data in a dataframe
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))


    df2 = df.copy()
    if 'entry_id' in df2.columns:
        df2.drop('entry_id', axis = 'columns', inplace = True)
    
    df2['time'] = pd.to_datetime(df2['created_at']).dt.tz_convert('UTC') # convert UTC to eastern time
    df2 = df2.drop('created_at', axis = 'columns')
    df2.set_index('time', inplace = True)

    return df2
    
    
def get_all_data(chID, start_date, end_date):
    """
    Get all requested data from Thingspeak, making multiple requests if needed
    Return the results as a Pandas DataFrame
    """

    # perform the request to get data from thingspeak
    df = thingspeak_to_df(chID, start_date, end_date)
    
    # if all of the data wasn't collected, because it was more than 8000 points, do another request
    while (got_all_data(start_date, df) is False):
        print('getting more data...')
        df_extra = thingspeak_to_df(chID, start_date, df.index[0].strftime("%Y-%m-%d %H:%M"))
        df = pd.concat([df, df_extra]).drop_duplicates().sort_index()
        
    df2 = df.copy()
    df2.index = df2.index.tz_convert('US/Eastern') # perform timezone conversion
    
    return df2
    