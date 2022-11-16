"""
Seasonal Precipitation Calculator
---------------------------------



"""
from multigrids import temporal_grid
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from multiprocessing import Process, Lock, active_children, cpu_count

def sum_seasonal_by_months(monthly, months, years='all', name="Summed-Precip"):
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
            precip_sum += monthly.get_grids(months_filtered[mon]) 
        except TypeError:
            precip_sum = monthly.get_grids(months_filtered[mon])

    
    
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
    return date + relativedelta(months=1)

def sum_element_for_date_range(monthly, start, end, row, col):
    """sum a season based on start and end dates. Fractional portion of months
    is calculated. Preforms operation only on pixel at row,col and for a single
    date_range

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

    date = start
    while True:
        keys.append('%i-%02i' % (date.year,date.month))

        if date.year == end.year and date.month == end.month:
            break
        date = get_start_next_month(date)

    # print(keys, row, col)
    data = monthly[keys, row,col]
    # print('data', len(data), data)
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


def process_pixel_warper(monthly, bounds, row, col, start_year, seasonal, lock ):
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
        # print(bound_pair)
        start, end = bound_pair[0],bound_pair[1]
        if lock:
            lock.acquire()
        sum = sum_element_for_date_range(
            monthly, start, end, row, col
        )
        # print(sum)
        seasonal.grids[year].reshape(
            seasonal.config['grid_shape']
        )[row,col] = sum
        if lock:
            lock.release()
        year += 1

def sum_grids_by_ranges(monthly, dates, n_days=None, num_process = 1, 
        start_row=0, start_col =0, end_row = None,
        summed_data = None, verbose=True
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

    # if summed_data is None: ## create resulting data space if none exists
    #     filename = 'calc_precip_temp.data'
    #     grids = np.memmap(filename, dtype='float32', mode='w+', shape=shape) 
    #     grids += np.nan
    # else:
    #     try:
    #         grids = summed_data.grids
    #     except AttributeError:
    #         grids = summed_data


    # print(grids.filename)
    
    real_shape = dates.config['real_shape']
    bounds_shape = (dates.config['num_timesteps']//2,2)
    disp_bounds = True
    
    # if end_row is None:
    #     rows = range(start_row, grid_shape[0])
    #     end_row = grid_shape[0]
    # else:
    #     rows = range(start_row, end_row)

    print('Calculating valid indices!')

    # indices = range(start, temp_grid.shape[1])
    init = monthly[monthly.config['start_timestep']].flatten()
    indices = ~np.isnan(init)
    recalc_mask = None # TODO impement
    if not recalc_mask is None:
        mask = recalc_mask.flatten()
        indices = np.logical_and(indices, mask)

    indices = np.where(indices)[0]
    shape=monthly.config['grid_shape']
    n_cells = shape[0] * shape[1]
    start = 0 # todo implment
    indices = indices[indices > start]
    import gc
    print("Starting, with %i processes" % num_process)
    for idx in indices: # flatted area grid index
        row, col = np.unravel_index(idx, shape)
        while len(active_children()) >= num_process:
            continue
        [gc.collect(i) for i in range(3)] 
        print(
            'calculating for element ' + str(idx) + \
            '. ~' + '%.2f' % ((idx/n_cells) * 100) + '% complete.'
        )
        # index = row, col
        bounds = dates.grids.reshape(real_shape)[:,row,col]\
            .reshape(bounds_shape)
        year_idx = 0
        if num_process == 1:
            process_pixel_warper(
                monthly, bounds, row, col, year_idx, summed_data, None
            )
        else:
            Process(target=process_pixel_warper,
                name = "Calc precip for all years in r:%i, c:%i" % (row, col),
                args=(
                    monthly, bounds, row, col, year_idx, summed_data, w_lock
                )
            ).start()

    # for row in rows:
    #     if verbose:
    #         print (row, '/', grid_shape[0], ':', row/(end_row-start_row)*100 )
    #     for col in range(start_col, grid_shape[1]):
    #         if np.isnan(monthly[0][row,col]):
    #             continue
    #         print(col)
    #         bounds = dates.grids.reshape(real_shape)[:,row,col]\
    #             .reshape(bounds_shape)
    #         if n_days:
    #             bounds[:,1] = bounds[:,0] + np.timedelta64(n_days,'D')
    #         if disp_bounds:
    #             print(bounds)
    #             disp_bounds=False

    #         year_idx = 0

    #         if num_process > 1: 
    #             while len(active_children()) >= num_process:
    #                 continue
    #             # (monthly, bounds, row, col, start_year, grids, lock ) <- args
    #             Process(target=process_pixel_warper,
    #                     name = "Calc precip for all years in r:%i, c:%i" % (row, col),
    #                     args=(
    #                         monthly, bounds, row, col, year_idx, grids, w_lock
    #                     )
    #                 ).start()
    #         else: ## no multiprocessing
    #             process_pixel_warper(
    #                 monthly, bounds, row, col, year_idx, grids, None
    #             )

    if num_process > 1:
        print('Waiting on final processes to complete')
        while len(active_children()) > 0 :
            
            continue
        print('Done')

    return grids

def sum_seasonal_by_roots(monthly, dates, summed_data=None, n_days=None, verbose=False):
    """calculates seasonal precip based on freezing/thawing dates

    parameters
    ----------
    monthly: TemporalGrid
        monthly precip data
    dates: TemporalGrid
        dates grid with alternating freezing then thawing dates
    n_days: int or string, optional
        override for season length, if this is used the end date for 
        the season is the thawing data my n_days from freezing date

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
    # grid_shape = monthly.config['grid_shape']
    # n_years = dates.config['num_grids']//2

    # precip_sum = temporal_grid.TemporalGrid(
    #     grid_shape[0], # rows
    #     grid_shape[1], # cols
    #     n_years, # years
    #     data_type = precip_sum.dtype, 
    #     dataset_name = name,
    #     description = "seasonal precip summed by date ranges",
    #     mask = monthly.config['mask'],
    #     # initial_data = precip_sum,
    #     start_timestep = start_year,
    #     save_to=
    # )


    sum_grids_by_ranges(
        monthly, dates, n_days, num_process=None,
        summed_data=summed_data, verbose=verbose
    ) 

    # precip_sum = temporal_grid.TemporalGrid(
    #     precip_sum.shape[1], # rows
    #     precip_sum.shape[2], # cols
    #     precip_sum.shape[0], # years
    #     data_type = precip_sum.dtype, 
    #     dataset_name = name,
    #     description = "seasonal precip summed by date ranges",
    #     mask = monthly.config['mask'],
    #     initial_data = precip_sum,
    #     start_timestep = start_year,
    # )

    return summed_data


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
    import os
    if os.path.isfile('temp-dates-grid.yml'):
        dates_mg = temporal_grid.TemporalGrid('temp-dates-grid.yml')
        return dates_mg
    else:
        dates_mg = temporal_grid.TemporalGrid(
            roots.config['grid_shape'][0], # rows
            roots.config['grid_shape'][1], # cols
            roots.config['num_timesteps'] - skip, # years
            data_type = 'datetime64[D]', 
            start_timestep = 0,
            save_to='temp-dates-grid.yml'
        )
    
    # skip_at_start:( -1 * skip_at_end) if skip_at_end else None
    for row in range(dates_mg.grids.shape[0]):
        print('dates_row: %s, roots row: %s, sas: %s' % (
            row, row+skip_at_start, skip_at_start
            )
        )
        dates_mg.grids[row] =  np.datetime64(start_date - timedelta(days=1)) 
        dates_mg.grids[row] += \
            abs(roots.grids[row + skip_at_start]).astype(np.timedelta64)
        print( dates_mg.grids[row][ ~np.isnan( roots.grids[0] ) ][:5] )
    return dates_mg


