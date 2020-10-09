"""
Seasonal Precipitation Calculator
---------------------------------



"""


from multigrids.tools import load_and_create, get_raster_metadata, temporal_grid
import glob, os
import re
import numpy as np

def load_monthly_data (
        directory, dataset_name = 'Monthly Precipitation', sort_func = sorted,
        no_data = -9999
    ):
    rasters =  sort_func(glob.glob(os.path.join(directory,'*.tif')))

    match = r'([1-3][0-9]{3})'
    start = os.path.split(rasters[0])[-1]
    start = re.search(match, start).group(0)   
    end = os.path.split(rasters[-1])[-1]
    end = re.search(match, end).group(0)   

    grid_names = [] 
    for yr in range(int(start), int(end)+1): 
        for mn in ['01','02','03','04','05','06','07','08','09','10','11','12']: 
            grid_names.append(str(yr) + '-' + str(mn)) 

    load_params = {
        "method": "tiff",
        "directory": directory,
        "sort_func": sort_func,
    }
    create_params = {
        "name": dataset_name,
        "grid_names": grid_names,
    }

    monthly_data = load_and_create(
        load_params, create_params
    )

    ex_raster = rasters[0]
    monthly_data.config['raster_metadata'] = get_raster_metadata(ex_raster)

    ## precip is never less than zero
    monthly_data.grids[monthly_data.grids < 0] = np.nan

    monthly_data.config['start_year'] = int(start)

    return monthly_data

def sum_precip(monthly, months, years='all', name="Summed-Precip"):
    """

    Parameters
    ----------
    monthly: multigrids.temporal_gird.TemporalGrid
        monthly precip data

    months: list of ints
        list of months numbers as integers where January is 1, February is 2, 
        and so on. If a season spans two years as winter does, use 13 as 
        the next January and 14 as the next February etc. So for a winter 
        comprising october through march pass [10, 11, 12 ,13 ,14 ,15].
    years: str
        Range of years to calculate average over. 
        'all' or 'start-end' ie '1901-1950'.
    
    Returns
    ------- 
    # winter_precip
    """
    # precip.grids = np.array(precip.grids)
    keys = monthly.config['grid_name_map'].keys()

    find = lambda x: sorted( [k for k in keys if k[-2:]=='{:0>2}'.format(x)])
    monthly_keys = {
        x: find(x) for x in range(1, 13)
        }     

    pad = 1 - max([m//12 for m in months])

    if years != 'all':
        try:
            start, end = years.split('-')
        except AttributeError:
            start, end = years[0], years[1]
        start, end = int(start), int(end)
    else:
        start = min([int(x.split('-')[0])for x in keys]) 
        end = max([int(x.split('-')[0])for x in keys]) 
    

    month_filter = lambda m, y, ks: [k for k in ks[m] if k[:4] == str(y)]
    months_filtered = {}
    for year in range(start, end+pad):
        for mon in months:
            if mon < 1:
                raise IndexError ("Months cannot be less then 1")
            adj_mon = mon
            adj_year = year
            while adj_mon > 12:
                adj_mon -= 12
                adj_year += 1

            try:
                months_filtered[mon] +=  month_filter(
                    adj_mon, adj_year, monthly_keys
                ) 
            except KeyError:
                months_filtered[mon] = month_filter(
                    adj_mon, adj_year, monthly_keys
                )
    
    precip_sum = None
    for mon in months_filtered:
        try:
            precip_sum += monthly.get_grids_at_keys(months_filtered[mon])
        except TypeError:
            precip_sum = monthly.get_grids_at_keys(months_filtered[mon])

    
    
    precip_sum = temporal_grid.TemporalGrid(
        precip_sum.shape[1], # rows
        precip_sum.shape[2], # cols
        precip_sum.shape[0], # years
        data_type = precip_sum.dtype, 
        dataset_name = name,
        description = "summed precip for " + str(months),
        mask = monthly.config['mask'],
        initial_data = precip_sum,
        start_timestep = monthly.config['start_year'],
        )

    return precip_sum
