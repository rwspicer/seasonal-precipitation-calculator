"""
subutilities
------------

unit parts for the cli utility 
"""
from multigrids.tools import load_and_create, get_raster_metadata, temporal_grid
import glob, os
import re
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sort import sort_snap_files
import seasonal

SEASONS = {
    "winter": [10, 11, 12, 13, 14, 15],
    "early-winter": [10, 11],
    "summer": [4, 5, 6, 7, 8, 9],
    "late-summer": [8, 9],
}


def get_sort_method(arguments, verbose=False):
    """gets the correct sort method
    """
    if arguments['--sort-method'].lower() == "default":
        sort_method = "SortUsing default python sort function"
        sort_fn = sorted
    elif  arguments['--sort-method'].lower() == 'snap':
        sort_method = "Using SNAP sort function"
        sort_fn = sort_snap_files
    elif os.path.isfile(arguments['--monthly-data']):
        sort_method = "Monthly Data is already TemporalGrid. sort_fn N/A"
        sort_fn = 'N/A'
    else:
        sort_method = (
            "Invalid --sort-method option. "
            "Run utility.py --help to see valid options"
        )
        verbose = True # force verbosity to true to display error message
        sort_fn = None
    
    if verbose:
        print(sort_method)

    return  sort_fn

def get_save_temp_status(test_val):
    if test_val.lower() in ['false', 'f']:
        save_temp = False
    elif test_val.lower() in ['true', 't']:
        save_temp = True
    else: # a path
        save_temp = test_val
    return save_temp

def load_monthly_data (
        directory, dataset_name = 'Monthly Precipitation', sort_func = sorted,
        save_temp = None, verbose=False
    ):
    """Loads monthy data directly from an existing TemporalGrid or by loading
    tiff files to a TemporalGrid
    """
    

    if os.path.isfile(directory):
        if verbose:
            print('Loading File: %s'% directory)
        monthly_data = temporal_grid.TemporalGrid(directory)
        key = list(monthly_data.config['grid_name_map'].keys())[0]
        start = [int(i) for i in key.split('-')][0]
    else:
        if verbose:
            print('Reading tiffs from directory: %s'% directory)
        rasters =  sort_func(glob.glob(os.path.join(directory,'*.tif')))

        ## get first and last year from file names
        match = r'([1-3][0-9]{3})'
        start = os.path.split(rasters[0])[-1]
        start = re.search(match, start).group(0)   
        end = os.path.split(rasters[-1])[-1]
        end = re.search(match, end).group(0)   

        ## create grid names
        grid_names = [] 
        for yr in range(int(start), int(end)+1): 
            for mn in [
                '01','02','03','04','05','06','07','08','09','10','11','12'
            ]: 
                grid_names.append(str(yr) + '-' + str(mn)) 

        load_params = {
            "method": "mp_tiff",
            "directory": directory,
            "sort_func": sort_func,
            "verbose": verbose,
        }
        create_params = {
            "name": dataset_name,
            "grid_names": grid_names,
            "delta_timestep": relativedelta(months=1),
            "start_timestep": datetime(int(start),1,1),
            "verbose": verbose,
        }

        if type(save_temp) is bool:
            if save_temp:
                save_temp = 'spc-temp-monthly-data.grids.yml'
            else:
                save_temp = None

        if not save_temp is None:
            create_params['save_to'] = save_temp
            create_params['name'] = \
                'Seasonal Precip Calculator Temp Monthly Data'

        monthly_data = load_and_create(
            load_params, create_params
        )

        ex_raster = rasters[0]

        monthly_data.config['raster_metadata'] = get_raster_metadata(ex_raster)

        monthly_data.grids[monthly_data.grids < 0] = np.nan

    monthly_data.config['start_year'] = int(start)

    return monthly_data

def load_roots_data(
        directory, dataset_name = 'roots', sort_func = sorted,
        save_temp = None, verbose=False
    ):
    if directory:
        if verbose:
            print('Loading File: %s'% directory)
        roots = temporal_grid.TemporalGrid(directory)
    else:
        if verbose:
            print('Reading tiffs from directory: %s'% directory)
        load_params = {
        "method": "mp_tiff",
        "directory": directory,
        "sort_func": sort_func
        }
        create_params = {
            "name": dataset_name,
            "start_timestep": 0,
        }

        if type(save_temp) is bool:
            if save_temp:
                save_temp = 'spc-temp-roots-data.grids.yml'
            else:
                save_temp = None

        if not save_temp is None:
            create_params['save_to'] = save_temp
            create_params['name'] = \
                'Seasonal Precip Calculator Temp roots data'

        roots = load_and_create(
            load_params, create_params
        )

    return roots

def method_monthly(arguments, monthly,  verbose = False):
    """
    """


    season = arguments['--season']
    if verbose:
        print('Season %s for monhtsL %s' % (season, SEASONS[season]))
    name = season + '-precipitation'

    summed = seasonal.sum_seasonal_by_months(
        monthly, SEASONS[season], name=name
    )

    summed.config['description'] = name + " for months " +\
        str(SEASONS[season]) +\
        "(month '13' is january for the next year, '14' is feb, etc)"
    summed.config['raster_metadata'] = monthly.config['raster_metadata']
    return summed

def method_roots(arguments, monthly, roots,  verbose = False):
    """
    """
    key = list(monthly.config['grid_name_map'].keys())[0]
    year, month = [int(i) for i in key.split('-')]
    start_date = datetime(year, month, 1)

    try:
        season_name, season_length = arguments['--season-length'].split('-')
    except ValueError:
        season_name = arguments['--season-length']
        season_length = None

    if verbose:
        if season_length is None:
            print(
                'Summing %s precipitation from start to end root' % season_name
            ) 
        else:
            print(
                'Summing %s precipitation'
                ' for %s days after start root.' % (season_name, season_length)
            )

    if season_name == 'summer':
        dates = seasonal.root_mg_to_date_mg(roots, start_date) # don't need to skip ends
    elif  season_name == 'winter':
        dates = seasonal.root_mg_to_date_mg(roots, start_date, 1, 1)
    print(dates)

    # try:
    monthly.config['start_timestep'] = 0
    # print(monthly.config['description'])
    summed = seasonal.sum_seasonal_by_roots(
        monthly, dates, season_length, season_name
    ) 
    # except IndexError as e:
    #     print (e)
    #     sys.exit()

    summed.config['start_timestep'] = year
    summed.config['raster_metadata'] = monthly.config['raster_metadata']

    return summed

