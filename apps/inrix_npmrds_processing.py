#-*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sys
import os
import geopandas as gpd
import json
import pathlib

########################################################################
### STEP 0: IMPORTING LIBRARY WITH INRIX/NPMRDS/RITIS ANALYSIS TOOLS ###
########################################################################

# Folder path to the "inrix_and_npmrds_tools.py" file
tool_location = (r'apps')

# Adding the location of the inrix tool
sys.path.append(tool_location)

import inrix_npmrds_ritis_tools as inr

# Location of the "SQL words to avoid" file
#sql_words_loc = "C:/Users/diasf/OneDrive - Jacobs/Random_Stuff"

# Adding the location of the list of SQL words to avoid
#sys.path.append(sql_words_loc)

#import sql_words_to_avoid


###############################################################
### STEP 1: RUNNING THE WHOLE ANALYSIS FROM START TO FINISH ###
###############################################################

def start_to_finish_custom_summaries_and_reliability(
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
    all_data = inr.read_batch_of_raw_data(input_data_folder, 
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
    main_data = inr.fix_datetime_columns(main_data)
    
    # Adding timeslot and date window columns
    main_data = inr.define_standard_fhwa_timeslots(main_data)
    
    ############################################################
    ### STEP 3: FILTERING, GROUPING AND SUMMARIZING THE DATA ###
    ############################################################
    
    all_summaries_concat = inr.filter_group_summarize_fhwa(main_data)
    
    all_summaries_with_geoms = inr.add_geometries_to_summaries(
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
    
    reliability_summaries_all = inr.calculate_standard_reliabilities(
        all_summaries_concat=all_summaries_concat, 
        main_data=main_data, 
        main_tmc_data=main_tmc_data,
        calc_mixed_traf_rel=False,
        calc_truck_rel=True)
    
    reliability_summaries_with_geoms = inr.add_geometries_to_summaries(
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



for ref_year in (2017, 2018, 2021, 2022):
    
    # Skipping 2019, 2020 - There doesn't seem to be enough data for links in 
    # these years
    #ref_year  = 2022 
    
    # Defining the folder to search for zipfiles containing the raw INRIX data
    # input_data_folder = pathlib.Path(r'C:\Users\diasf\Jacobs\CoP_Freight Transport'
    #                                  r'ation Planning - HDOT Freight Plan Update -'
    #                                  r' HDOT Freight Plan Update\Data Analysis\NPM'
    #                                  fr'RDS\raw_data\15 mins Interval\{ref_year}')
    
    # Defining the main searched road and the chunk size for opening the large CSV
    # raw data files
    road_str = ''
    chunk_size = 100000
    
    # Defining the filenames for the output files generated after reading in all of
    # the raw data and filtering it to contain only data related to the main 
    # searched road
    
    # output_folder = pathlib.Path(r'C:\Users\diasf\Jacobs\CoP_Freight Transportatio'
    #                              r'n Planning - HDOT Freight Plan Update - HDOT Fr'
    #                              r'eight Plan Update\Data Analysis\NPMRDS\processe'
    #                              r'd_outputs')
    
    # output_raw_data_filename_pickle = (output_folder / 
    #                                       f'RawSpeeds_{ref_year}_15mins_2023-03-22.pkl')
    # output_tmc_data_filename_pickle = (output_folder /
    #                                       f'TMC_Data_{ref_year}_2023-03-22.pkl')
    
    
    # npmrds_geodata_path = ("zip:///Users/diasf/Jacobs/CoP_Freight Transportation P"
    #                        "lanning - HDOT Freight Plan Update - HDOT Freight Plan"
    #                        " Update/Data Analysis/NPMRDS/raw_data/npmrds_links/NPM"
    #                        f"RDS_Hawaii_{min(ref_year,2021)}.zip")
    
    
    # output_summary_data_filename_gpkg = os.path.join(output_folder,
    #                                         f'FHWA_Summaries_{ref_year}_2023-03-22.gpkg')
    
    # output_reliability_data_filename_gpkg = os.path.join(output_folder,
    #                                             f'FHWA_Reliability_{ref_year}_2023-03-22.gpkg')
    
    
    
    
    
    output_dict = start_to_finish_custom_summaries_and_reliability(
                      input_data_folder=input_data_folder,
                      npmrds_geodata_path=npmrds_geodata_path,
                      road_str=road_str,
                      chunk_size=chunk_size,
                      export_raw_speed_data=False,
                      output_raw_data_filename_pickle=output_raw_data_filename_pickle,
                      export_tmc_data=True,
                      output_tmc_data_filename_pickle=output_tmc_data_filename_pickle,
                      export_summary_data=True,
                      output_summary_data_filename_gpkg=output_summary_data_filename_gpkg,
                      export_reliability_data=True,
                      output_reliability_data_filename_gpkg=output_reliability_data_filename_gpkg)



def get_gpkg(in_file):
    if in_file.stem.find('Summaries') >= 0:
        this_year = in_file.stem[15:19]
    else:
        this_year = in_file.stem[17:21]
    if in_file.stem.find('AllYears') < 0:
        this_gdf = gpd.read_file(in_file).assign(Data_Year=int(this_year))
        return this_gdf

for this_file in output_folder.glob('*Summaries*'):
    pass


summary_data = (pd
                .concat([get_gpkg(this_file) for 
                         this_file in output_folder.glob('*Summaries*')],
                        ignore_index=True)
                .sort_values(by=['Data_Year','tmc_code'])
                .reset_index(drop=True)
                )

reliability_data = (pd
                .concat([get_gpkg(this_file) for 
                         this_file in output_folder.glob('*Reliability*')],
                        ignore_index=True)
                .sort_values(by=['Data_Year','tmc_code'])
                .reset_index(drop=True)
                )



summary_data.to_file(output_folder / 'FHWA_Summaries_AllYears_2023-03-22.gpkg')
reliability_data.to_file(output_folder / 'FHWA_Reliability_AllYears_2023-03-22.gpkg')




