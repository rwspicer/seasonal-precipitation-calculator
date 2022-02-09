"""
Seasonal Precipitation Calculator
---------------------------------



"""
from multigrids.tools import load_and_create, get_raster_metadata, temporal_grid
import glob, os
import re
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Process, Lock, active_children, cpu_count

def load_monthly_data (
        directory, dataset_name = 'Monthly Precipitation', sort_func = sorted,
        no_data = -9999 
    ):

    if os.path.isfile(directory):
        monthly_data = temporal_grid.TemporalGrid(directory)
        key = list(monthly_data.config['grid_name_map'].keys())[0]
        start = [int(i) for i in key.split('-')][0]
    else:
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

        ## precip is never less than zero TODO is this the cause of the weird issues?
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

def get_start_next_month(date):
    """
    Parameters
    ----------
    date: datetime.datetime

    Returns
    -------
    datetime.datetime
        date of the start of the next month
    """
    if date.month+1 <= 12:
        return datetime(date.year,date.month+1,1)  
    return datetime(date.year+1,1,1) ## its next year

def sum_element_by_date_range(monthly, start, end, row, col):
    """sum a season based on start and end dates. Fractional protion of months
    is calculated.

    Parameters
    ----------
    monthly: TemporalGrid
        monthly precip data
    start: np.datetime64
        start date to create seasonal sum
    end: np.datetime64
        end date to create seasonal sum
    row: int
    col: int
        row and col to process
    
    Returns
    -------
    float
        sum of seasonal precip 
    """
    try:
        start = datetime.fromisoformat(str(start))
    except ValueError:
        return np.nan
    end = datetime.fromisoformat(str(end))

    keys = []
    month = start.month
    year = start.year
    while True:
        keys.append('%i-%02i' % (year,month))

        if year == end.year and month == end.month:
            break

        month += 1
        if month == 13:
            month = 1
            year += 1


    data = monthly.get_grids_at_keys(keys)[:, row,col]
    if len(data) == 1:
        ## do somthing else:
        total_days = (
            get_start_next_month(end) - datetime(end.year,end.month,1)
        ).days
        days = (end.day + 1) - start.day
        fraction = days/total_days
        return data[0] * fraction 

    total_days = (
        get_start_next_month(start) - datetime(start.year,start.month,1)
    ).days
    days = (get_start_next_month(start) - start).days
    fraction_start = days/total_days
    data[0] = data[0] * fraction_start    
    
    total_days = (
        get_start_next_month(end) - datetime(end.year,end.month,1)
    ).days
    days = end.day
    fraction_end = days/total_days
    data[-1] = data[-1] * fraction_end

    return data.sum()


def process_pixel_warper(monthly, bounds, row, col, start_year, grids, lock ):
    """multiprocessing wrapper to process a single year for a single pixel

    Parameters
    ----------
    monthly: TemporalGrid
        monthly data
    bounds: 
    row: int
    col: int
        row, and col to process
    start_year: int  
        year to start processing at
    grid: np.array like
        array to store the results to
    lock: multiprocessing.Lock
        Lock for writing data

    """
    year = start_year

    for bound_pair in bounds:
        start, end = bound_pair[0],bound_pair[1]
        if lock:
            lock.acquire()
        grids[year, row, col] = sum_element_by_date_range(
            monthly, start, end, row, col
        )
        if lock:
            lock.release()
        year += 1


def sum_grids_by_ranges(monthly, dates, n_days=None, num_process = 1, 
        start_row=0, start_col =0, end_row = None,
        summed_data = None
    ):
    """
    calculates seasonal precip based on date ranges

    Parameters
    ----------
    monthly: TemporalGrid
        monthly precip data
    dates: TemporalGrid
        dates grid with alternating freezing then thawing dates
    n_days: int, Optional
        override for season length, if this is used the end date for 
        the seasion is the thawing data my n_days from freezing date.
        Full seaasons are used if this is not provided
    num_processes: int default 
        number of process for multiprocessing
    start_row: int
    start_col: int
        start row and start columns to use
    end_row: int, Optional
        if none number of rows in monthly data is used
    summed_data: np.array or TemporalGrid
        if None data is created, otherwise data is stored in array or 
        TemporalGrid provided
    
    Returns
    -------
    np.array with shape (n_year, row, cols)
    """
    w_lock = Lock()
    if num_process is None:
       num_process = cpu_count()

    grid_shape = monthly.config['grid_shape']
    n_years = dates.config['num_grids']//2
    shape = (n_years,grid_shape[0],grid_shape[1])

    if summed_data is None: ## create resulting data space if none exists
        filename = 'calc_precip_temp.data'
        grids = np.memmap(filename, dtype='float32', mode='w+', shape=shape) 
        grids += np.nan
    else:
        try:
            grids = summed_data.grids
        except AttributeError:
            grids = summed_data
    
    real_shape = dates.config['real_shape']
    bounds_shape = (dates.config['num_timesteps']//2,2)
    disp_bounds = True
    
    if end_row is None:
        rows = range(start_row, grid_shape[0])
        end_row = grid_shape[0]
    else:
        rows = range(start_row, end_row)

    for row in rows:
        print (row, '/', grid_shape[0], ':', row/(end_row-start_row)*100 )
        for col in range(start_col, grid_shape[1]):
            if np.isnan(monthly[0][row,col]):
                continue
            
            bounds = dates.grids.reshape(real_shape)[:,row,col]\
                .reshape(bounds_shape)
            if n_days:
                bounds[:,1] = bounds[:,0] + np.timedelta64(n_days,'D')
            if disp_bounds:
                print(bounds)
                disp_bounds=False

            year_idx = 0

            if num_process > 1: 
                while len(active_children()) >= num_process:
                    continue
                # (monthly, bounds, row, col, start_year, grids, lock ) <- args
                Process(target=process_pixel_warper,
                        name = "Calc precip for all years in r:%i, c:%i" % (row, col),
                        args=(
                            monthly, bounds, row, col, year_idx, grids, w_lock
                        )
                    ).start()
            else: ## no multiprocessing
                process_pixel_warper(
                    monthly, bounds, row, col, year_idx, grids, None
                )

    if num_process > 1:
        print('Waiting on final processes to complete')
        while len(active_children()) > 0 :
            
            continue
        print('Done')

    return grids

def sum_seasonal_2(monthly, dates, n_days=None, season='winter', start_year=0, 
        name="Summed-Precip"
    ):
    """calculates seasonal precip based on freezing/thawing dates

    parameters
    ----------
    monthly: TemporalGrid
        monthly precip data
    dates: TemporalGrid
        dates grid with alternating freezing then thawing dates
    n_days: int or string, optional
        override for season length, if this is used the end date for 
        the seasion is the thawing data my n_days from freezing date

        if a string 'summer' or 'winter' should be provided to calculate the
        season from the roots for the respective season 
    start_year: int, default 0
        start year for labeling 
    name: str
        name used for labeling

    returns
    -------
    TemporalGrid
        the seasonal precip

    """
    precip_sum = sum_grids_by_ranges(monthly, dates, n_days, num_process=8) 

    precip_sum = temporal_grid.TemporalGrid(
        precip_sum.shape[1], # rows
        precip_sum.shape[2], # cols
        precip_sum.shape[0], # years
        data_type = precip_sum.dtype, 
        dataset_name = name,
        description = "seasonal precip summed by date ranges",
        mask = monthly.config['mask'],
        initial_data = precip_sum,
        start_timestep = start_year,
        )

    return precip_sum


def root_mg_to_date_mg(roots, start_date, skip_at_start=0, skip_at_end=None):
    """convert TemporalGrid of roots to TemporalGrid of dates

    Parameters
    ----------
    roots: TemporalGrid
        the roots data, where -numbers represent freezing roots and + numbers
        thawing roots. The absolunte value of each root is the number of 
        days from the start_date
    start_date: datetime.datetime
        day '0' for roots
    skip_at_start: int default 0
        number of roots to skip at start
    skip_at_end: int, default None
        number of roots to skip at end

    Returns
    -------
    TemporalGrid
        type is np.datetime64 
    """

    skip = skip_at_start + skip_at_end if skip_at_end else 0

    dates_mg = temporal_grid.TemporalGrid(
        roots.config['grid_shape'][0], # rows
        roots.config['grid_shape'][1], # cols
        roots.config['num_timesteps'] - skip, # years
        data_type = 'datetime64[D]', 
        start_timestep = 0,
        )

    dates_mg.grids[:] =  np.datetime64(start_date - timedelta(days=1)) 
    dates_mg.grids[:] += abs(roots.grids).astype(np.timedelta64)[
        skip_at_start:( -1 * skip_at_end) if skip_at_end else None
    ] 
    return dates_mg


