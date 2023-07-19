# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:13:08 2021

@author: DIASF
"""

import os
import re
import zipfile
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely

def which_zip_is_which_data_source(input_data_folder):
    '''
    Function that searches the input data folder for zip files and determines 
    which zipfiles contain the data needed for this task. The function also 
    identifies the type of data/data source for each of those zipfiles. 
    For example: 'texas_inrix_npmrds_15min(1).zip' contains the 'NPMRDS from 
    INRIX (Passenger vehicles)' data.
    
    Parameters
    ----------
    input_data_folder : STR
        String that indicates the folder to be investigated for the zipfiles 
        containing the raw data

    Returns
    -------
    data_paths_dict: DICT
        Dictionary that contains information about where the files for each 
        data source is located. It should be structured as follows:
            {'data_origin_1':{'zip_file':'zip_file_1_full_path.zip',
                              'raw_data_file':'raw_data_file_name_1.csv'},
             'data_origin_2':{'zip_file':'zip_file_2_full_path.zip',
                              'raw_data_file':'raw_data_file_name_2.csv'},
             ...}
    '''
    
    # Dictionary that is used to match data origin to the RegEx string
    dict_for_origin_match = {
        'inrix':
            '.*INRIX TMC.*',
        'npmrds_from_inrix_pass_vehicles':
            '.*NPMRDS from INRIX \(Passenger vehicles\).*',
        'npmrds_from_inrix_trucks':
            '.*NPMRDS from INRIX \(Trucks\).*',
        'npmrds_from_inrix_trucks_and_passveh':
            '.*NPMRDS from INRIX \(Trucks and passenger vehicles\).*'}
    
    
    # Find all zip files in input folder
    folder_files = [f for f in os.listdir(input_data_folder) if 
                    os.path.isfile(os.path.join(input_data_folder, f))]
    
    # Filter only zipfiles
    folder_zips = list(filter(re.compile(".*zip$").match, folder_files))
    
    # Dictionary that will store the output
    data_paths_dict = {}
    
    # Looping over zipfiles
    for this_zip_file in folder_zips:
        with zipfile.ZipFile(os.path.join(input_data_folder, 
                                          this_zip_file)) as this_zip:
            
            # Checking if this is a data extract from RITIS' massive data downloader
            files_in_zip = this_zip.namelist()
            if 'Contents.txt' in files_in_zip:
                
                #Extracting the name of the raw data CSV file inside this zipfile
                raw_data_file = [this_file for this_file in files_in_zip if 
                                 this_file !='Contents.txt' and 
                                 this_file !='TMC_Identification.csv'][0]
                
                # Performing a RegEx search to find which data source this 
                # zipfile originally came from
                with this_zip.open('Contents.txt','r') as content_file:
                    this_content = content_file.readline().decode('utf-8')
                    for this_data_origin, this_regex_string in (
                            dict_for_origin_match.items()):
                        regex_search = re.match(this_regex_string,this_content)
                        if regex_search:
                            data_paths_dict[this_data_origin] = {
                                'zip_file':os.path.join(input_data_folder,
                                                        this_zip_file),
                                'raw_data_file':raw_data_file}
    return data_paths_dict

def read_csv_get_specific_road_segments(data_origin,
                                        raw_data_zipfile,
                                        raw_data_filename_in_zip,
                                        road_str,chunk_size,
                                        raw_data_chunks=None,
                                        tmc_data_parts=None):
    """
    Function used to read in raw speed and TMC segment data. This function will 
    likely be called multiple times because of the several different sources of 
    input files. For example: INRIX, NPMRDS from INRIX, etc.
    
    INPUT VARIABLES:
    ----------------
    data_origin: STR 
       String that characterizes the origin of the data. Sample values: 
       'inrix','npmrds_from_inrix_trucks', 'npmrds_from_pass_vehicles'
    raw_data_zipfile: STR
        String that contains the file/folder location of the zipfile to be read
    raw_data_filename_in_zip: STR
        String that contains the filename of the raw data inside the zipfile
    road_str: STR used to filter road segments based on their names. The TMC 
        segments will be filtered based on whether or not the 'road' column 
        contains this string. To get the entire dataset back, just use an 
        empty string ('').
    chunk_size: INT
        Integer used to determine number of rows read at a time by Pandas' 
        read_csv method.
    raw_data_chunks: LIST 
        List containing the several chunks of input files thus far.
        The first time this function is called, this should just be an empty list.
    tmc_data_parts: LIST 
        List containing the several tmc_data inputs from the multiple times this 
        function is called.
                        
    OUTPUT:
    -------
    output_dict : DICT
        Dictionary that contains two values: 'raw_data_chunks' and 'tmc_data_parts':
        raw_data_chunks: LIST 
            List of pd.DataFrames that contain the several chunks of all the 
            input  data-files, including the chunks created in the current 
            execution of this method. 
            Note: It is expected that these chunks will later be concatenated 
            into one large DataFrame afterwards.
        tmc_data_parts: LIST 
            List of pd.DataFrames containing the analogous TMC data 
            (i.e., the data in the "TMC_Identification.csv" files).
    """ 
    
    if not raw_data_chunks:
        raw_data_chunks = []
    if not tmc_data_parts:
        tmc_data_parts = []
    
    # Opening the zipfile
    with zipfile.ZipFile(raw_data_zipfile) as this_zip:
        
        # Reading in the TMC data from the zipfile
        with this_zip.open('TMC_Identification.csv','r') as tmc_data_file:
            tmc_data = pd.read_csv(tmc_data_file, low_memory=False)
    
        # Adding extra column about data origin and storing the final result
        tmc_data['data_origin'] = data_origin
        tmc_data['road'] = tmc_data['road'].fillna('')
        
        # Fixing column names
        tmc_data = tmc_data.rename({'tmc':'tmc_code',
                                    'intersection':'intersection_',
                                    'state':'state_',
                                    'type':'type_'},axis=1)
        
        # Querying main searched road
        tmc_data = tmc_data.query(f'road.str.contains("{road_str}")', 
                                  engine='python')
        
        # Sometimes, this DataFrame has multiple rows for the same TMC. 
        # This step is taken to de-duplicate the TMCs data.
        tmc_data = (tmc_data
                    .sort_values(by=['tmc_code','active_end_date'])
                    .reset_index(drop=True))
        tmc_data = tmc_data.groupby('tmc_code').last().reset_index()
        tmc_data_parts.append(tmc_data.copy())
        
        # Subset of the TMC data with only the relevant columns
        tmc_data_sub = tmc_data[['tmc_code','road','data_origin']]
        
        # Reading in the raw data in chunks and only keeping segments that 
        # are related to the main searched road
        with this_zip.open(raw_data_filename_in_zip,'r') as tmc_data_file:
            with pd.read_csv(tmc_data_file, 
                             chunksize=chunk_size, 
                             dtype={'tmc_code':'str'}) as reader:
                for raw_data in reader:
                    raw_data = raw_data.merge(tmc_data_sub, 
                                              how='left', on='tmc_code')
                    raw_data = raw_data.loc[raw_data.road.notnull()]
                    raw_data_chunks.append(raw_data.copy())
    
    # Since we need to return more than one output, the multiple outputs have 
    # been added to a dictionary.
    output_dict = {'raw_data_chunks':raw_data_chunks,
                   'tmc_data_parts':tmc_data_parts}
     
    return output_dict

def read_one_set_of_raw_data(input_data_folder,road_str,chunk_size,data_origin):
    '''
    Looks into the input folder and reads in the raw data contained in only 
    one of the zipfiles.
    
    Parameters
    ----------
    input_data_folder : STR
        String that indicates the folder to be investigated for the zipfiles 
        containing the raw data
    road_str: STR 
        String used to filter road segments based on their names. This is also 
        referred to as "the main searched road" in other places of this script.
        The TMC segments will be filtered based on whether or not the 'road' 
        column contains this string. To get the entire dataset back, just use 
        an empty string ('').
    chunk_size : INT
        Integer that defines the chunk size for Pandas' `read_csv` method.
    data_origin: STR 
       String that characterizes the origin of the data. Sample values: 
       'inrix','npmrds_from_inrix_trucks', 'npmrds_from_pass_vehicles'

    Returns
    -------
    output_dict: DICT
        Dictionary with two entries: "main_data" and "main_tmc_data".
        main_data : pd.DataFrame
            Pandas DataFrame that contains the actual raw speed data for the main 
            searched road
        main_tmc_data : pd.DataFrame
            Pandas DataFrame that contains the associated TMC data for all the 
            TMC segments on the main searched road

    '''

    # Searching the input folder for zipfiles and determining where the relevant
    # raw data files are. This function also tells you which "data_origin" is 
    # associated with each of the zipfiles.
    data_paths_dict = which_zip_is_which_data_source(input_data_folder)
    
    
    # This is an empty list that will store the DataFrame chunks from reading 
    # in the raw speed data.
    raw_data_chunks = []
    
    # This is an empty list that will store the DataFrames containing the TMC-
    # segment  link data that is associated with each data source (i.e., the 
    # data in the "TMC_Identification.csv" files)
    tmc_data_parts = []
    
    
    raw_data_zipfile = data_paths_dict[data_origin]['zip_file']
    raw_data_filename_in_zip = data_paths_dict[data_origin]['raw_data_file']
    results_dict = read_csv_get_specific_road_segments(
                             data_origin=data_origin,
                             raw_data_zipfile=raw_data_zipfile,
                             raw_data_filename_in_zip=raw_data_filename_in_zip,
                             road_str=road_str,
                             chunk_size=chunk_size,
                             raw_data_chunks=raw_data_chunks,
                             tmc_data_parts=tmc_data_parts)
    raw_data_chunks = results_dict['raw_data_chunks']
    tmc_data_parts = results_dict['tmc_data_parts']
    
    # Concatenating all raw data chunks into one single DataFrame
    main_data = pd.concat(raw_data_chunks, ignore_index=True).reset_index(drop=True)
    
    # Making sure there are no duplicates. If there are, they are averaged out.
    #main_data = main_data.groupby(['data_origin','tmc_code','measurement_tstamp']).mean().reset_index()
    main_data = main_data.drop_duplicates(subset=['data_origin','tmc_code','measurement_tstamp']).reset_index(drop=True)
    
    # Concatenating all TMC data parts into one single DataFrame
    main_tmc_data = pd.concat(tmc_data_parts, ignore_index=True).reset_index(drop=True)
    
    # Since we need to return more than one output, the multiple outputs have 
    # been added to a dictionary.
    output_dict = {'main_data':main_data,
                   'main_tmc_data':main_tmc_data}
    
    return output_dict


def read_batch_of_raw_data(input_data_folder,road_str,chunk_size):
    '''
    Looks into the input folder and reads all of the zipfiles.
    
    Parameters
    ----------
    input_data_folder : STR
        String that indicates the folder to be investigated for the zipfiles 
        containing the raw data
    road_str: STR 
        String used to filter road segments based on their names. This is also 
        referred to as "the main searched road" in other places of this script.
        The TMC segments will be filtered based on whether or not the 'road' 
        column contains this string. To get the entire dataset back, just use 
        an empty string ('').
    chunk_size : INT
        Integer that defines the chunk size for Pandas' `read_csv` method.

    Returns
    -------
    output_dict: DICT
        Dictionary with two entries: "main_data" and "main_tmc_data".
        main_data : pd.DataFrame
            Pandas DataFrame that contains the actual raw speed data for the main 
            searched road
        main_tmc_data : pd.DataFrame
            Pandas DataFrame that contains the associated TMC data for all the 
            TMC segments on the main searched road            

    '''
    # Searching the input folder for zipfiles and determining where the relevant
    # raw data files are. This function also tells you which "data_origin" is 
    # associated with each of the zipfiles.
    data_paths_dict = which_zip_is_which_data_source(input_data_folder)
    
    
    # This is an empty list that will store the DataFrames containing the raw 
    # speed data.
    main_data_parts = []
    
    # This is an empty list that will store the DataFrames containing the TMC-
    # segment link data that is associated with each data source (i.e., the 
    # data in the "TMC_Identification.csv" files)
    tmc_data_parts = []
    
    # Actually running the batch input process
    for i,this_data_origin in enumerate(data_paths_dict):
        this_batch_results = read_one_set_of_raw_data(input_data_folder,
                                                      road_str,
                                                      chunk_size,
                                                      this_data_origin)
        main_data_parts.append(this_batch_results['main_data'])
        tmc_data_parts.append(this_batch_results['main_tmc_data'])
    
    # Concatenating all main data parts into one single DataFrame
    main_data = pd.concat(main_data_parts, ignore_index=True).reset_index(drop=True)
    
    # Making sure there are no duplicates. If there are, they are averaged out.
    #main_data = main_data.groupby(['data_origin','tmc_code','measurement_tstamp']).mean().reset_index()
    main_data = main_data.drop_duplicates(subset=['data_origin','tmc_code','measurement_tstamp']).reset_index(drop=True)
    
    # Dropping observations/rows where there is no speed data. 
    # This whole process is geared towards finding average (and percentiles) of
    # speeds. If the data point provides us with no speed info, there is 
    # nothing else we can use that data point for.
    main_data = main_data.loc[main_data['speed'].notna()].reset_index(drop=True)
    
    # Concatenating all TMC data parts into one single DataFrame
    main_tmc_data = pd.concat(tmc_data_parts, ignore_index=True).reset_index(drop=True)
    
    # Since we need to return more than one output, the multiple outputs have 
    # been added to a dictionary.
    output_dict = {'main_data':main_data,
                   'main_tmc_data':main_tmc_data}
    
    return output_dict


def fix_datetime_columns(main_data):
    '''
    Generates an actual datetime column in the "main_data" DataFrame by parsing 
    the text-based timestamp column. Also extracts day-of-week and time info 
    into separate columns.

    Parameters
    ----------
    main_data : pd.DataFrame
        The pandas DataFrame that contains all the raw data from the RITIS
        website (INRIX/NPMRDS speeds)

    Returns
    -------
    main_data : pd.DataFrame
        The same DataFrame as the input, except that now, the DataFrame has a 
        few new datetime-related columns. Namely:
            -day_of_week: indicates the row's day of the week as a number from 
                0 (Monday) to 6 (Sunday)
            -day_of_week_str: indicates the row's day of the week as a string
                of text
            -time: indicates the row's TIME (without date)

    '''
    # Transforming STRING timestamp into an actual datetime format
    main_data['measurement_tstamp'] = pd.to_datetime(main_data['measurement_tstamp'])
    
    # Extracting day-of-week data and making it more readable.
    # Monday=0, Sunday=6
    main_data['day_of_week'] = main_data.measurement_tstamp.dt.day_of_week
    main_data['day_of_week_str'] = (main_data['day_of_week']
                                    .apply(lambda x: {0:'0 - Monday',
                                                      1:'1 - Tuesday',
                                                      2:'2 - Wednesday',
                                                      3:'3 - Thursday',
                                                      4:'4 - Friday',
                                                      5:'5 - Saturday',
                                                      6:'6 - Sunday'}[x]))
    
    
    # Extracting day-of-year data
    main_data['day_of_year'] = main_data.measurement_tstamp.dt.day_of_year
    
    # Extracting the time value, which was coded originally in 15 minute intervals
    main_data['time'] = main_data.measurement_tstamp.dt.time
    
    return main_data

class time_slot():
    '''
    Class that is used to label the observations in the `main_data` DataFrame 
    (that contains all the raw data from the RITIS speeds database) according 
    to the time of day. For example: am_peak, pm_peak, etc.
    ''' 
    def __init__(self,time_start,time_end,include_start, include_end, 
                 inside_outside,slot_name):
        '''
        Instantiates `time_slot`.
        
        Parameters
        ----------
        time_start : datetime.time
            Start time of the time slot
        time_end : datetime.time
            End time of the time slot
        include_start : BOOL
            Indicates whether to use >= or just > for time_start
        include_end : BOOL
            Indicates whether to use <= or just < for time_end
        inside_outside : STR
            Indicates whether the time slot refers to the time inside or outside
            of the start and end times. To be more specific:
                If inside_outside=="inside", then the time slot refers to the 
                time AFTER time_start but BEFORE time_end. 
                If inside_outside=="outside" , then the time slot refers to the 
                time BEFORE time_start but AFTER time_end (e.g.: before 6am and 
                after 10pm). 
        slot_name : STR
            Describes the name of the time slot. Typical names include "am_peak",
            "pm_peak", "off_peak".

        Returns
        -------
        The newly-created instance of this class.

        '''
        self.time_start     = time_start
        self.time_end       = time_end
        self.include_start  = include_start
        self.include_end    = include_end
        self.inside_outside = inside_outside
        self.slot_name      = slot_name
        
    
    def get_filter(self, main_data):
        '''
        Gets the filter/mask that indicates which of the INRIX observations belong
        to this specific time slot.
        
        Parameters
        ----------
        main_data : pd.DataFrame
            The pandas DataFrame that contains all the raw data from the RITIS
            website (INRIX/NPMRDS speeds)
        
        Returns
        -------
        ts_filter : pd.Series (bool)
            An array of BOOL variables that indicates whether or not each 
            observation belongs to this specific time slot. The array has length
            equal to the number of rows in main_data.
        '''
        
        try:
            return self.ts_filter
        except: 
            if self.inside_outside == 'inside':
                this_filter = ((self.time_start < main_data['time']) & 
                               (main_data['time'] < self.time_end))
            elif self.inside_outside == 'outside':
                this_filter = ((main_data['time'] < self.time_start) | 
                               (self.time_end < main_data['time']))
            if self.include_start:
                this_filter = (this_filter | 
                               (main_data['time'] == self.time_start))
            if self.include_end:
                this_filter = (this_filter | 
                               (main_data['time'] == self.time_end))
            self.ts_filter = this_filter
            return self.ts_filter
        
    def add_time_slot_data_to_main_data(self, main_data):
        '''
        Adds the 'time_slot' column to the data and applies `time_slot`'s name
        to the appropriate rows.

        Parameters
        ----------
        main_data : pd.DataFrame
            The pandas DataFrame that contains all the raw data from the RITIS
            website (INRIX/NPMRDS speeds)

        Returns
        -------
        main_data : pd.DataFrame
            The same DataFrame as was passed in the input. The only difference 
            is that now, the `time_slot`'s name was applied to the rows that 
            fall within the `time_slot`'s filter.

        '''
        try:
            main_data.loc[self.get_filter,'time_slot'] = self.slot_name
            return main_data
        except:
            main_data['time_slot'] = np.nan
            main_data.loc[self.get_filter,'time_slot'] = self.slot_name
            return main_data


class day_of_year_window():
    '''
    Class that is used to label the observations in the `main_data` DataFrame 
    (that contains all the raw data from the RITIS speeds database) according 
    to the day of the year. For example: some analyses require only data between 
    September and October.
    ''' 
    def __init__(self,start_date,end_date,include_start, include_end, 
                 inside_outside,window_name):
        '''
        Instantiates `day_of_year_window`.
        
        Parameters
        ----------
        start_date : INT
            Integer that indicates the day_of_year for the window's start date
        end_date : INT
            Integer that indicates the day_of_year for the window's end date
        include_start : BOOL
            Indicates whether to use >= or just > for start_date
        include_end : BOOL
            Indicates whether to use <= or just < for end_date
        inside_outside : STR
            Indicates whether the window refers to the days inside or outside
            of the start and end dates. To be more specific:
                If inside_outside=="inside", then the window refers to the 
                time AFTER start_date but BEFORE end_date. 
                If inside_outside=="outside" , then the time slot refers to the 
                time BEFORE start_date but AFTER end_date (e.g.: Before January
                25th and after November 12th). 
        window_name : STR
            Describes the name of the window.

        Returns
        -------
        The newly-created instance of this class.

        '''
        self.start_date     = start_date
        self.end_date       = end_date
        self.include_start  = include_start
        self.include_end    = include_end
        self.inside_outside = inside_outside
        self.window_name    = window_name
        
    
    def get_filter(self, main_data):
        '''
        Gets the filter/mask that indicates which of the INRIX observations belong
        to this specific date window.
        
        Parameters
        ----------
        main_data : pd.DataFrame
            The pandas DataFrame that contains all the raw data from the RITIS
            website (INRIX/NPMRDS speeds)
        
        Returns
        -------
        doy_filter : pd.Series (bool)
            An array of BOOL variables that indicates whether or not each 
            observation belongs to this specific window. The array has length
            equal to the number of rows in main_data.
        '''
        
        try:
            return self.doy_filter
        except: 
            if self.inside_outside == 'inside':
                this_filter = ((self.start_date < main_data['day_of_year']) &
                               (main_data['day_of_year'] < self.end_date))
            elif self.inside_outside == 'outside':
                this_filter = ((main_data['day_of_year'] < self.start_date) | 
                               (self.end_date < main_data['day_of_year']))
            if self.include_start:
                this_filter = (this_filter | 
                               (main_data['day_of_year'] == self.start_date))
            if self.include_end:
                this_filter = (this_filter | 
                               (main_data['day_of_year'] == self.end_date))
            self.doy_filter = this_filter
            return self.doy_filter
        
    def add_window_data_to_main_data(self, main_data):
        '''
        Adds the 'date_window' column to the data and applies the 
        `day_of_year_window`'s name to the appropriate rows.

        Parameters
        ----------
        main_data : pd.DataFrame
            The pandas DataFrame that contains all the raw data from the RITIS
            website (INRIX/NPMRDS speeds)

        Returns
        -------
        main_data : pd.DataFrame
            The same DataFrame as was passed in the input. The only difference 
            is that now, the `day_of_year_window`'s name was applied to the rows 
            that fall within the `day_of_year_window`'s filter.

        '''
        try:
            main_data.loc[self.get_filter,'date_window'] = self.window_name
            return main_data
        except:
            main_data['date_window'] = np.nan
            main_data.loc[self.get_filter,'date_window'] = self.window_name
            return main_data



# Function used to calculate summaries for grouped data
def calc_summaries(grouped_data):
    '''
    Calculates all the important summaries for means and percentiles.

    Parameters
    ----------
    grouped_data : DataFrameGroupBy object
        DataFrame that was filtered down and grouped using the `groupby` function.

    Returns
    -------
    grouped_data_summaries : pd.DataFrame
        Pandas DataFrame containing all of the summary results: means and 
        percentiles for speed and travel time.

    '''
    # Determining which column name to use: minutes or seconds
    if 'travel_time_minutes' in grouped_data.head().columns:
        tt_col = 'travel_time_minutes'
    else:
        tt_col = 'travel_time_seconds'
    
    grouped_data_summaries = grouped_data.agg(
        count_obs = pd.NamedAgg(column='measurement_tstamp', aggfunc='count'),
                                                
        speed_avg = pd.NamedAgg(column='speed', aggfunc=('mean')),
        speed_01p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q =  1))),
        speed_05p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q =  5))),
        speed_15p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 15))),
        speed_20p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 20))),
        speed_50p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 50))),
        speed_80p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 80))),
        speed_85p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 85))),
        speed_95p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 95))),
        speed_99p = pd.NamedAgg(column='speed', aggfunc=(lambda x: np.percentile(x, q = 99))),
        
        ttime_avg = pd.NamedAgg(column=tt_col, aggfunc=('mean')),
        ttime_01p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q =  1))),
        ttime_05p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q =  5))),
        ttime_15p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 15))),
        ttime_20p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 20))),
        ttime_50p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 50))),
        ttime_80p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 80))),
        ttime_85p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 85))),
        ttime_95p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 95))),
        ttime_99p = pd.NamedAgg(column=tt_col, aggfunc=(lambda x: np.percentile(x, q = 99))),
        
        )
    return grouped_data_summaries


def calc_summaries_pipeline(main_data,
                            summary_name,
                            summary_filter,
                            grouping_columns):
    '''
    This function simplifies and standardizes the process of calculating 
    summaries from the raw data. The user needs to tell the function what
    rows are to be kept, what columns will be used to group the data and the 
    name of this particular summary.
    This function then returns the newly-calculated summarized data containing
    a new column called "summary_type".

    Parameters
    ----------
    main_data : pd.DataFrame
        The pandas DataFrame that contains all the raw data from the RITIS
        website (INRIX/NPMRDS speeds).
    summary_name : str
        String that describes this summary type. After the `main_data` is 
        summarized, a new column called "summary_type" will be generated. 
        This new column will contain the text stored in the `summary_name`
        variable.
    summary_filter : np.array
        Array containing only boolean values (False/True). This indicates which
        rows from the `main_data` should be used in the calculation of these 
        summaries.
    grouping_columns : list
        List of column names that will be used to group the `main_data`
        dataset

    Returns
    -------
    summarized_data : pd.DataFrame
        Pandas DataFrame containing all of the summary results: means and 
        percentiles for speed and travel time.

    '''
    grouped_data = main_data.loc[summary_filter].groupby(grouping_columns)
    
    summarized_data = calc_summaries(grouped_data)
    
    summarized_data['summary_type'] = summary_name
    
    return summarized_data




def read_npmrds_geodata(npmrds_geodata_path):
    '''
    Reads in the shapefile associated with the NPMRDS data. Typically, this 
    file is just called "Texas.shp"

    Parameters
    ----------
    npmrds_geodata_path : STR
        String describing the full path to the ".shp" file on disk of where the 
        NPMRDS data can be found. 

    Returns
    -------
    npmrds_geodata : gpd.GeoDataFrame
        The GeoDataFrame with the actual geodata from the NPMRDS shapefile.
    '''
    
    npmrds_geodata = gpd.read_file(npmrds_geodata_path).rename({'Tmc':'tmc_code'},
                                                               axis=1)
    
    return npmrds_geodata




def make_link(df_row):
    '''
    Function that creates a simplified link geometry (straight line) using the 
    start/end long/lat data from the original INRIX main data.

    Parameters
    ----------
    df_row : pd.Series
        One row of the `main_data_summaries` DataFrame

    Returns
    -------
    row_line : shapely.LineString
        Geometric feature created using the lat/lon data from in the input row

    '''
    start_pt = shapely.geometry.Point([df_row['start_longitude'], 
                                       df_row['start_latitude']])
    end_pt = shapely.geometry.Point([df_row['end_longitude'], 
                                     df_row['end_latitude']])
    row_line = shapely.geometry.LineString([start_pt,end_pt])
    return row_line

def add_geometries_to_summaries(summarized_data, 
                                main_tmc_data,
                                npmrds_geodata_path):
    '''
    Adds a column called "geom_final" to the dataset. This new column contains 
    a geometry for each row in the summary dataset. This geometry is generated 
    in one of two different ways, in the following priority:
        -Look in the NPMRDS shapefile to try and find a link with matching TMC
        -If we can't find one, we just draw a straight line from the lat/lon
            data that is found in the TMC_Identification.csv files.
    
    Parameters
    ----------
    summarized_data : pd.DataFrame
        Pandas DataFrame that contains the speed summaries. 
    main_tmc_data : pd.DataFrame
        Pandas DataFrame that contains the associated TMC data for all the 
        TMC segments (i.e., the data from all the "TMC_Identification.csv" files)
    npmrds_geodata_path : STR
        String that identifies where to find the NPMRDS shapefile. Needs to 
        point to the ".shp" file. Typically, this file is just called "Texas.shp"
        
    Returns
    -------
    summarized_data_with_geoms : gpd.GeoDataFrame
        GeoDataFrame containing the geometries associated with each link.
    
    '''
    
    npmrds_geodata = gpd.read_file(npmrds_geodata_path).rename({'Tmc':'tmc_code'},
                                                               axis=1)
    
    # Merging summaries with NPMRDS geometries
    summarized_data_with_geoms = summarized_data.merge(
        npmrds_geodata[['tmc_code','geometry']].to_crs('epsg:4326'), 
        how='left', 
        on='tmc_code')
    
    # Applying the "make_link" function to generate the simplified geometries
    temp_geoms = (summarized_data_with_geoms
                  .drop_duplicates(subset=['data_origin','tmc_code'])
                  .sort_values(by=['data_origin','tmc_code'])
                  .reset_index(drop=True))
    
    temp_geoms = temp_geoms.merge(main_tmc_data[['tmc_code',
                                                 'data_origin',
                                                 'start_latitude',
                                                 'start_longitude',
                                                 'end_latitude',
                                                 'end_longitude']], 
                                  how='left', 
                                  on=['data_origin','tmc_code'])
    
    temp_geoms['geometry_simplified'] = temp_geoms.apply(make_link, axis=1)

    summarized_data_with_geoms = summarized_data_with_geoms.merge(
        temp_geoms[['data_origin','tmc_code','geometry_simplified']], 
        how='left', on=['data_origin','tmc_code'])
    
    
    # Filter that indicates which observations/rows did not have an 
    # associated geometry in the NPMRDS geodata.
    null_geoms_filter = summarized_data_with_geoms['geometry'].isnull()
    
    # Populating the `geom_final` column with either the geometry found in the NPMRDS
    # geodata (preferred), or with the simplified geometry created above (fallback).
    summarized_data_with_geoms['geom_final'] = summarized_data_with_geoms['geometry']
    summarized_data_with_geoms.loc[null_geoms_filter,'geom_final'] = (
        summarized_data_with_geoms.loc[null_geoms_filter,'geometry_simplified'])
    
    # Populating the `geom_final_type` column with a flag that shows what type of 
    # geometry is in the `geom_final` column. 
    # If `geom_final_type`== 'original_tmc_shape', then the geometry in 
    # `geom_final` represents the original NPMRDS geometry. 
    # If `geom_final_type`== 'simplified_tmc_shape', then the geometry in 
    # `geom_final` represents the simplified geometry we created in this script
    # just a few lines above. 
    summarized_data_with_geoms['geom_final_type'] = 'original_tmc_shape'
    summarized_data_with_geoms.loc[null_geoms_filter,
                                   'geom_final_type'] = 'simplified_tmc_shape'
    
    summarized_data_with_geoms = gpd.GeoDataFrame(
        summarized_data_with_geoms.drop(['geometry','geometry_simplified']
                                        ,axis=1),
        crs='epsg:4326',geometry='geom_final')
    
    # Extracting the WKT data. Useful for exporting to CSV. 
    summarized_data_with_geoms['geom_wkt'] = gpd.array.to_wkt(
        summarized_data_with_geoms.geom_final.values)
    
    # Exporting final data to disk
    #summarized_data_with_geoms.to_file(main_data_geoms_filename,driver='GPKG',layer='main')    

    return summarized_data_with_geoms
    
def define_standard_fhwa_timeslots(main_data):
    '''
    Adds timeslot and date window columns to `main_data`. These are the 
    standard timeslots used for the FHWA reliability computations:
        -AM Peak:   Between 06am and 10am
        -Mid-day:   Between 10am and 04pm
        -PM Peak:   Between 04pm and 08pm
        -Overnight: Between 08pm and 06am
    
    Parameters
    ----------
    main_data : pd.DataFrame
        Input DataFrame containing raw speed data for all links.

    Returns
    -------
    main_data : pd.DataFrame
        DataFrame containing raw speed data for all links. 
        After this function is executed, the following columns get added:
            -"time_slot"
            -"date_window"

    '''
    
    # Creating the thresholds for the definitions of peak and off-peak time slots
    # and finding the observations that fall in each category/time slot.
    # Time slots used are:
    #    -AM Peak:   Between 06am and 10am
    #    -Mid-day:   Between 10am and 04pm
    #    -PM Peak:   Between 04pm and 08pm
    #    -Overnight: Between 08pm and 06am
    
    am_peak = time_slot(time_start = pd.to_datetime('06:00 AM').time(),
                        time_end = pd.to_datetime('10:00 AM').time(),
                        include_start = True, 
                        include_end = False,
                        inside_outside = 'inside',
                        slot_name = 'am_peak')
    
    afternoon_offpeak = time_slot(time_start = pd.to_datetime('10:00 AM').time(),
                                  time_end = pd.to_datetime('04:00 PM').time(),
                                  include_start = True, 
                                  include_end = False,
                                  inside_outside = 'inside',
                                  slot_name = 'midday')
    
    pm_peak = time_slot(time_start = pd.to_datetime('04:00 PM').time(),
                        time_end = pd.to_datetime('08:00 PM').time(),
                        include_start = True, 
                        include_end = False,
                        inside_outside = 'inside',
                        slot_name = 'pm_peak')
    
    night = time_slot(time_start = pd.to_datetime('06:00 AM').time(),
                      time_end = pd.to_datetime('08:00 PM').time(),
                      include_start = False, 
                      include_end = True,
                      inside_outside = 'outside',
                      slot_name = 'overnight')
    
    all_time_slots = [am_peak, afternoon_offpeak, pm_peak, night]
    
    # Adding the peak/offpeak/etc category data back into the `main_data` DataFrame
    for this_time_slot in all_time_slots:
        main_data = this_time_slot.add_time_slot_data_to_main_data(main_data)
    
    # Creating the thresholds for the definitions of day-of-year windows and finding
    # the observations that fall in each category/window.
    # The windows used are:
    #    -All days: Between Jan 1, 2019 and Dec 31, 2019
    # Note: Currently, there is only one category that spans the entire year. 
    #       The functionality was built in for future projects, when we might want 
    #       to compare, say, speeds during the four seasons. 
    
    all_days_window = day_of_year_window(start_date=pd.to_datetime('Jan 1, 2019').day_of_year, 
                                         end_date=pd.to_datetime('Dec 31, 2019').day_of_year, 
                                         include_start = True, 
                                         include_end = True,
                                         inside_outside = 'inside',
                                         window_name = 'all_days')
    
    main_data = all_days_window.add_window_data_to_main_data(main_data)
    
    return main_data


def filter_group_summarize_fhwa(main_data):
    '''
    Defines the standard periods and summaries needed for calculating the 
    FHWA reliability values.
    To see the formal definitions of these periods, see CFR 23 490.511 and 
    CFR 23 490.611:
        https://www.ecfr.gov/current/title-23/chapter-I/subchapter-E/part-490/subpart-E/section-490.511
        https://www.ecfr.gov/current/title-23/chapter-I/subchapter-E/part-490/subpart-F/section-490.611
        https://www.law.cornell.edu/cfr/text/23/490.511
        https://www.law.cornell.edu/cfr/text/23/490.611

    Parameters
    ----------
    main_data : pd.DataFrame
        Input dataframe containing raw speed data for every time period. 
        It is expected that this dataframe will contain the following columns:
            -"time_slot"
            -"date_window"
            -"day_of_week"
            -"time"
            -"day_of_year"

    Returns
    -------
    all_summaries_concat : pd.DataFrame
        DataFrame that contains all the standard summary data required for 
        FHWA's reliability calculations.

    '''
    
    #--------------------------------------------------------
    # Step 1: Summarizing data for AM Peaks (only weekdays) -
    #--------------------------------------------------------
    
    # Name for this summary
    summary_name = 'am_peak'
    
    # Filters for Weekdays. Peak and afternoon-off-peak times.
    time_slot_filter   = main_data['time_slot'].isin(['am_peak'])
    day_of_year_filter = main_data['date_window'].isin(['all_days'])
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = main_data['day_of_week'].isin([0,1,2,3,4])
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                      day_of_year_filter & 
                      time_filter & 
                      day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    # Calculating the summaries
    summarized_data_ampeak = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    
    #-------------------------------------------------------
    # Step 2: Summarizing data for Mid-day (only weekdays) -
    #-------------------------------------------------------
    
    # Name for this summary
    summary_name = 'midday'
    
    # Filters for Weekdays. Peak and afternoon-off-peak times.
    time_slot_filter   = main_data['time_slot'].isin(['midday'])
    day_of_year_filter = main_data['date_window'].isin(['all_days'])
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = main_data['day_of_week'].isin([0,1,2,3,4])
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                      day_of_year_filter & 
                      time_filter & 
                      day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    # Calculating the summaries
    summarized_data_midday = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    
    #--------------------------------------------------------
    # Step 3: Summarizing data for PM Peaks (only weekdays) -
    #--------------------------------------------------------
    
    # Name for this summary
    summary_name = 'pm_peak'
    
    # Filters for Weekdays. Peak and afternoon-off-peak times.
    time_slot_filter   = main_data['time_slot'].isin(['pm_peak'])
    day_of_year_filter = main_data['date_window'].isin(['all_days'])
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = main_data['day_of_week'].isin([0,1,2,3,4])
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                      day_of_year_filter & 
                      time_filter & 
                      day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    # Calculating the summaries
    summarized_data_pmpeak = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    
    #----------------------------------------------------
    # Step 4: Summarizing data for overnight - all days -
    #----------------------------------------------------
    
    # Name for this summary
    summary_name = 'overnight'
    
    # Filters for Weekends - only considering 6am to 8pm.
    time_slot_filter   = main_data['time_slot'].isin(['overnight'])
    day_of_year_filter = main_data['date_window'].isin(['all_days'])
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = np.repeat(True, len(main_data))
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                        day_of_year_filter & 
                        time_filter & 
                        day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    # Calculating the summaries
    summarized_data_overnight = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    
    #-----------------------------------------------------
    # Step 5: Summarizing data for weekends - 6am to 8pm -
    #-----------------------------------------------------
    
    # Name for this summary
    summary_name = 'weekends'
    
    # Filters for Weekends - only considering 6am to 8pm.
    time_slot_filter   = main_data['time_slot'].isin(['am_peak','midday','pm_peak'])
    day_of_year_filter = main_data['date_window'].isin(['all_days'])
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = main_data['day_of_week'].isin([5,6])
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                      day_of_year_filter & 
                      time_filter & 
                      day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    # Calculating the summaries
    summarized_data_weekends = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    # Adding extra detail about this summary's time slot
    #summarized_data_weekends['time_slot'] = '6am_to_8pm'
    
    
    #--------------------------------------
    # Step 6: Summarizing data - All-time -
    #--------------------------------------
    
    # Name for this summary
    summary_name = 'alltime'
    
    # Filters for alltime averages
    time_slot_filter   = np.repeat(True, len(main_data))
    day_of_year        = np.repeat(True, len(main_data))
    time_filter        = np.repeat(True, len(main_data))
    day_of_week_filter = np.repeat(True, len(main_data))
    
    # Combining all the filters
    summary_filter = (time_slot_filter & 
                      day_of_year &
                      time_filter & 
                      day_of_week_filter)
    
    # TODO: Need to find a more user-friendly way to define these filters
    
    # Columns used to group data for summaries
    grouping_columns = ['data_origin','tmc_code']
    
    summarized_data_alltime = calc_summaries_pipeline(
        main_data=main_data, 
        summary_name=summary_name,
        summary_filter=summary_filter,
        grouping_columns=grouping_columns)
    
    # Adding extra detail about this summary's time slot
    #summarized_data_alltime['time_slot'] = 'all_hours'
    
    # Creating list with all the summaries from the previous step
    all_summaries = [summarized_data_ampeak.reset_index(),
                     summarized_data_midday.reset_index(),
                     summarized_data_pmpeak.reset_index(),
                     summarized_data_overnight.reset_index(),
                     summarized_data_weekends.reset_index(),
                     summarized_data_alltime.reset_index()]
    
    # Concatenating all of the summaries into one large DataFrame
    all_summaries_concat = pd.concat(all_summaries, ignore_index=True).reset_index(drop=True)
    
    return all_summaries_concat

def calculate_standard_reliabily_mixed_traffic(all_summaries_concat, 
                                               main_tmc_data):
    '''
    Calculates travel time reliability for mixed traffic according to FHWA's 
    standards. 
    Note: See "National Performance Measures for Congestion, Reliability, and 
    Freight, and CMAQ Traffic Congestion":
        https://www.fhwa.dot.gov/tpm/guidance/
        https://www.fhwa.dot.gov/tpm/guidance/hif18040.pdf
    
    Parameters
    ----------
    all_summaries_concat : pd.DataFrame
        Dataframe containing all of the summary data needed for the computation
        of the reliability metrics. 
    main_tmc_data : pd.DataFrame
        Pandas DataFrame that contains the associated TMC data for all the TMC
        segments (i.e., the data from all the "TMC_Identification.csv" files)

    Returns
    -------
    reliability_summaries_all : pd.DataFrame
        Dataframe containing the reliability data for each TMC segment.

    '''
    # For the mixed traffic data (i.e., for observations where 
    # data_origin is in ['inrix', 'npmrds_from_inrix_trucks_and_passveh']):
    #    Calculate 80th percentile divided by 50th percentile for four summary 
    #    groups: am_peak, midday, pm_peak, weekends.
    #    Then, take the highest one of all four. If that value is larger than 
    #    1.5, the segment is deemd "unreliable". Otherwise, the segment is 
    #    deemed "reliable".
    

    # Keeping only the relevant summaries
    mixed_traffic_data = all_summaries_concat.loc[
        (all_summaries_concat['summary_type'].isin(['am_peak', 
                                                    'midday', 
                                                    'pm_peak',
                                                    'weekends'])) 
        & (all_summaries_concat['data_origin'].isin(['inrix',
                                                     'npmrds_from_inrix_trucks_and_passveh']))
        ].reset_index(drop=True)
    
    # Calculating Level of Travel Time Reliability: 80th percentile divided by 
    # 50th percentile (travel times)
    mixed_traffic_data['LOTTR_80p_by_50p'] = (
        mixed_traffic_data['ttime_80p'] / 
        mixed_traffic_data['ttime_50p'])
    
    # Calculating the maximum of the Travel Time Reliability across all 
    # summaries
    reliability_summaries_mixed_traffic = (mixed_traffic_data
        .groupby(['tmc_code','data_origin'])
        .agg(
            RawDataRows = pd.NamedAgg(column='count_obs', 
                                       aggfunc='sum'),
            SummaryCount = pd.NamedAgg(column='LOTTR_80p_by_50p', 
                                       aggfunc='count'),
            Reliability = pd.NamedAgg(column='LOTTR_80p_by_50p', 
                                                   aggfunc='max'))
        .reset_index())
    
    reliability_summaries_mixed_traffic['Reliability_Type'] = 'Mixed_Traffic'
    
    # Dropping rows that didn't have summaries for all periods needed
    reliability_summaries_mixed_traffic = (
        reliability_summaries_mixed_traffic.loc[
            reliability_summaries_mixed_traffic['SummaryCount'] == 4]
        .reset_index(drop=True))
    
    # Adding the binary "Reliable" column. 
    reliability_summaries_mixed_traffic['Reliable'] = (
        reliability_summaries_mixed_traffic['Reliability'] < 1.5)
    
    reliability_summaries_mixed_traffic = (reliability_summaries_mixed_traffic
                                           [['tmc_code', 'data_origin', 
                                             'Reliability_Type', 'RawDataRows', 
                                             'SummaryCount', 'Reliability',
                                             'Reliable']])
    
    return reliability_summaries_mixed_traffic

def calculate_standard_reliability_trucks(all_summaries_concat, main_tmc_data):
    '''
    Calculates truck travel time reliability according to FHWA's standards. 
    Note: See "National Performance Measures for Congestion, Reliability, and 
    Freight, and CMAQ Traffic Congestion":
        https://www.fhwa.dot.gov/tpm/guidance/
        https://www.fhwa.dot.gov/tpm/guidance/hif18040.pdf
        
    Parameters
    ----------
    all_summaries_concat : pd.DataFrame
        Dataframe containing all of the summary data needed for the computation
        of the reliability metrics. 
    main_tmc_data : pd.DataFrame
        Pandas DataFrame that contains the associated TMC data for all the TMC
        segments (i.e., the data from all the "TMC_Identification.csv" files)

    Returns
    -------
    reliability_summaries_all : pd.DataFrame
        Dataframe containing the reliability data for each TMC segment.

    '''
    # For truck traffic (data_origin='npmrds_from_inrix_trucks'):
    #    Calculate 95th percentile divided by 50th percentile for the five 
    #    summary groups: am_peak, midday, pm_peak, overnight and weekends.
    #    Then, just take the maximum of those five values. 
    #
    # Note: Please note that, for trucks, we also have to analyze the 
    #       "overnight" summary, which is absent in the mixed traffic 
    #       summaries.
    
    # Keeping only the relevant summaries
    truck_data = all_summaries_concat.loc[
        (all_summaries_concat['summary_type'].isin(['am_peak', 
                                                    'midday', 
                                                    'pm_peak',
                                                    'overnight',
                                                    'weekends'])) & 
        (all_summaries_concat['data_origin']=='npmrds_from_inrix_trucks')
        ].reset_index(drop=True)
    
    # Calculating Level of Travel Time Reliability: 95th percentile divided by 
    # 50th percentile (travel times)
    truck_data['LOTTR_95p_by_50p'] = (
        truck_data['ttime_95p'] / 
        truck_data['ttime_50p'])
    
    # Calculating the maximum of the Travel Time Reliability across all 
    # summaries
    reliability_summaries_truck_traffic = (truck_data
        .groupby(['tmc_code','data_origin'])
        .agg(
            RawDataRows = pd.NamedAgg(column='count_obs', 
                                       aggfunc='sum'),
            SummaryCount = pd.NamedAgg(column='LOTTR_95p_by_50p', 
                                       aggfunc='count'),
            Reliability = pd.NamedAgg(column='LOTTR_95p_by_50p', 
                                                   aggfunc='max'))
        .reset_index())

    reliability_summaries_truck_traffic['Reliability_Type'] = 'Truck_Traffic'

    # Dropping rows that didn't have summaries for all periods needed
    reliability_summaries_truck_traffic = (
        reliability_summaries_truck_traffic.loc[
            reliability_summaries_truck_traffic['SummaryCount'] == 5]
        .reset_index(drop=True))
    
    # Adding the binary "Reliable" column. 
    reliability_summaries_truck_traffic['Reliable'] = (
        reliability_summaries_truck_traffic['Reliability'] < 1.5)
    
    reliability_summaries_truck_traffic = (reliability_summaries_truck_traffic
                                           [['tmc_code', 'data_origin', 
                                             'Reliability_Type', 'RawDataRows', 
                                             'SummaryCount', 'Reliability',
                                             'Reliable']])
    
    return reliability_summaries_truck_traffic

def calculate_standard_reliabilities(all_summaries_concat, 
                                     main_data, 
                                     main_tmc_data,
                                     calc_mixed_traf_rel=True,
                                     calc_truck_rel=True):
    '''
    Calculates the overall Reliability according to FHWA's standards. 
    Note: See "National Performance Measures for Congestion, Reliability, and 
    Freight, and CMAQ Traffic Congestion":
        https://www.fhwa.dot.gov/tpm/guidance/
        https://www.fhwa.dot.gov/tpm/guidance/hif18040.pdf

    Parameters
    ----------
    all_summaries_concat : pd.DataFrame
        Dataframe containing all of the summary data needed for the computation
        of the reliability metrics. 
    main_data : pd.DataFrame
        Input dataframe containing raw speed data for every time period. 
    main_tmc_data : pd.DataFrame
        Pandas DataFrame that contains the associated TMC data for all the TMC
        segments (i.e., the data from all the "TMC_Identification.csv" files)
    calc_mixed_traf_rel : boolean
        Flag that signals whether or not to calculate reliability for mixed 
        traffic
    calc_truck_rel : boolean
        Flag that signals whether or not to calculate reliability for trucks

    Returns
    -------
    reliability_summaries_all : pd.DataFrame
        Dataframe containing the reliability data for each TMC segment.

    '''
    # Calculating overall Reliability. 
    
    # For the mixed traffic data (data_origin='npmrds_from_inrix_trucks_and_passveh'):
    #    Calculate 80th percentile divided by 50th percentile for four summary 
    #    groups: am_peak, midday, pm_peak, weekends.
    #    Then, take the highest one of all four. If that value is larger than 1.5, 
    #    the segment is deemed "unreliable". Otherwise, the segment is deemed "reliable".
    
    # For truck traffic (data_origin='npmrds_from_inrix_trucks'):
    #    Calculate 95th percentile divided by 50th percentile for the five summary
    #    groups: am_peak, midday, pm_peak, overnight and weekends.
    #    Then, just take the maximum of those five values. 
    #    Note: For trucks, this is a continuous value, while for mixed traffic the 
    #    metric is just a binary "reliable"/"unreliable" variable.
    
    # List that will store all reliability results
    reliability_dfs = []
    
    # Calculating reliability for mixed traffic
    if calc_mixed_traf_rel:
        reliability_summaries_mixed_traffic = (
            calculate_standard_reliabily_mixed_traffic(all_summaries_concat, 
                                                       main_tmc_data))
        
        # Finding missing TMC codes and re-including them
        missing_tmc_codes_mixed_traf = find_missing_tmc_codes(main_data, 
                                                   reliability_summaries_mixed_traffic)
        
        missing_tmc_codes_mixed_traf_df = (
            pd.DataFrame({'tmc_code':missing_tmc_codes_mixed_traf,
                          'data_origin':reliability_summaries_mixed_traffic['data_origin'].values[0]}))
        
        reliability_summaries_mixed_traffic = (
            pd.concat([reliability_summaries_mixed_traffic, 
                       missing_tmc_codes_mixed_traf_df],
                      ignore_index=True).reset_index(drop=True))

        reliability_dfs.append(reliability_summaries_mixed_traffic)
    
    # Calculating reliability for trucks
    if calc_truck_rel:
        reliability_summaries_truck_traffic = (
            calculate_standard_reliability_trucks(all_summaries_concat, 
                                                  main_tmc_data))
        
        missing_tmc_codes_truck = find_missing_tmc_codes(main_data, 
                                                   reliability_summaries_truck_traffic)
        # Finding missing TMC codes and re-including them
        missing_tmc_codestruck_df = (
                    pd.DataFrame({'tmc_code':missing_tmc_codes_truck,
                                  'data_origin':reliability_summaries_truck_traffic['data_origin'].values[0]}))
                
        reliability_summaries_truck_traffic = (
            pd.concat([reliability_summaries_truck_traffic, 
                       missing_tmc_codestruck_df],
                      ignore_index=True).reset_index(drop=True))
        
        reliability_dfs.append(reliability_summaries_truck_traffic)
    
    # Combining mixed traffic and truck reliability data
    reliability_summaries_all = pd.concat(reliability_dfs,
                                          ignore_index=True).reset_index(drop=True)
    
    return reliability_summaries_all


def find_missing_tmc_codes(main_data, ref_data):
    '''
    Finds which TMC codes are missing in the `ref_data` from the original raw
    dataset.

    Parameters
    ----------
    main_data : pd.DataFrame
        Input dataframe containing raw speed data for every time period. 
    ref_data : pd.DataFrame
        Reference data whose TMC codes will be checked against the original
        raw data.

    Returns
    -------
    missing_tmc_codes : LIST
        List of TMC codes that are "missing" from `ref_data` (i.e., they exist
        in `main_data`, but not in `ref_data`).

    '''
    # Getting unique TMC codes from both sets
    main_data_unique = main_data['tmc_code'].unique()
    ref_data_unique  = set(ref_data['tmc_code'].unique())
    
    # List that will hold missing TMC codes
    missing_tmc_codes = []
    
    # Comparing the sets and finding which ones are missing
    trash = pd.Series(main_data_unique).apply(
        lambda x: missing_tmc_codes.append(x) 
            if x not in ref_data_unique 
            else None)
    
    return missing_tmc_codes

def start_to_finish_fhwa_summaries_and_reliability(
        input_data_folder='',
        npmrds_geodata_path='',
        road_str='',
        chunk_size=100000,
        export_raw_speed_data=False,
        output_raw_data_filename_pickle='',
        export_tmc_data=False,
        output_tmc_data_filename_pickle='',
        export_summary_data=False,
        output_summary_data_filename_gpkg='',
        export_reliability_data=False,
        output_reliability_data_filename_gpkg=''):
    '''
    Does everything needed to calculate the FHWA reliability metrics from the 
    zipped raw data files from RITIS. 
    The several bells and whistles in this function's inputs are just controls
    of whether or not to export some of the processed datasets to the local disk.

    Parameters
    ----------
    input_data_folder : STR
        String that indicates the folder to be investigated for the zipfiles 
        containing the raw data
    npmrds_geodata_path : STR
        String that identifies where to find the NPMRDS shapefile. Needs to 
        point to the ".shp" file. Typically, this file is just called "Texas.shp"
    road_str: STR 
        String used to filter road segments based on their names. This is also 
        referred to as "the main searched road" in other places of this script.
        The TMC segments will be filtered based on whether or not the 'road' 
        column contains this string. To get the entire dataset back, just use 
        an empty string ('').
    chunk_size: INT
        Integer used to determine number of rows read at a time by Pandas' 
        read_csv method.
    export_raw_speed_data : bool
        Determines whether or not to export the raw speed data to disk. 
        The default is False.
    output_raw_data_filename_pickle : STR
        Full (absolute) path of the PICKLE file containing the raw data
        read in through this function.
    export_tmc_data : bool
        Determines whether or not to export the TMC Information data to disk. 
        The default is False.
    output_tmc_data_filename_pickle : STR
        Full (absolute) path of the PICKLE file containing the TMC data
        read in through this function.
    export_summary_data : bool
        Determines whether or not to export the summary data to disk. 
        The default is False.
    output_summary_data_filename_gpkg : STR
        String that identifies the path and filename to give to the GeoDataFrame
        that contains the summary data. This needs to be a GeoPackage
        file ('.gpkg' extension).
    export_reliability_data : bool
        Determines whether or not to export the reliability data to disk. 
        The default is False.
    output_reliability_data_filename_gpkg : STR
        String that identifies the path and filename to give to the GeoDataFrame
        that contains the reliability data. This needs to be a GeoPackage
        file ('.gpkg' extension). The default is ''.

    Returns
    -------
    output_dict : DICT
        Dictionary containing four datasets:
            -main_data: pd.DataFrame that contains all the raw data with the 
                extra processing columns 
            -main_tmc_data: pd.DataFrame that contains the TMC information
                about all the links
            -all_summaries_with_geoms: gpd.GeoDataFrame that contains all the
                summary data
            -reliability_summaries_with_geoms: gpd.GeoDataFrame that contains
                all the reliability data
    '''
    
    ############################
    ### STEP 1: READING DATA ###
    ############################
    
    # Actually reading in the whole data and generating the filtered output files
    all_data = read_batch_of_raw_data(input_data_folder, 
                                      road_str, 
                                      chunk_size)
    
    # Fishing out the `main_data` and `main_tmc_data` DataFrames.
    main_data = all_data['main_data']
    main_tmc_data  = all_data['main_tmc_data']

    # Exporting raw data and TMC information
    if export_raw_speed_data:
        main_data.to_pickle(output_raw_data_filename_pickle)
    
    if export_tmc_data:
        main_tmc_data.to_pickle(output_tmc_data_filename_pickle)

    ###################################################
    ### STEP 2: PRE-POCESSING COLUMNS FOR FILTERING ###
    ###################################################
    
    # Fixing datetime information: adding time and day_of_week columns
    main_data = fix_datetime_columns(main_data)
    
    # Adding timeslot and date window columns
    main_data = define_standard_fhwa_timeslots(main_data)
    
    ############################################################
    ### STEP 3: FILTERING, GROUPING AND SUMMARIZING THE DATA ###
    ############################################################
    
    all_summaries_concat = filter_group_summarize_fhwa(main_data)
    
    all_summaries_with_geoms = add_geometries_to_summaries(
                                   summarized_data=all_summaries_concat, 
                                   main_tmc_data=main_tmc_data,
                                   npmrds_geodata_path=npmrds_geodata_path)
    
    if export_summary_data:
        all_summaries_with_geoms.to_file(output_summary_data_filename_gpkg, 
                                         driver='GPKG',
                                         layer='main')
    
    ################################################
    ### STEP 4: CALCULATING RELIABILITY MEASURES ###
    ################################################
    
    reliability_summaries_all = calculate_standard_reliabilities(
        all_summaries_concat=all_summaries_concat, 
        main_data=main_data, 
        main_tmc_data=main_tmc_data,
        calc_mixed_traf_rel=True,
        calc_truck_rel=True)
    
    reliability_summaries_with_geoms = add_geometries_to_summaries(
        summarized_data=reliability_summaries_all, 
        main_tmc_data=main_tmc_data,
        npmrds_geodata_path=npmrds_geodata_path)
    
    if export_reliability_data:
        reliability_summaries_with_geoms.to_file(output_reliability_data_filename_gpkg, 
                                                 driver='GPKG',
                                                 layer='main')
    
    output_dict = {'main_data':main_data,
                   'main_tmc_data':main_tmc_data,
                   'all_summaries_with_geoms':all_summaries_with_geoms,
                   'reliability_summaries_with_geoms':reliability_summaries_with_geoms}
    
    return output_dict

















