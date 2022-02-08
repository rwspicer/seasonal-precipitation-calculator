"""
Seasonal Precipitation Calculator Utility 
-----------------------------------------

Utility for creating seasonal precipitation products by summing monthly
raster files. 

"""
from multigrids.temporal_grid import TemporalGrid
from multigrids import tools
import CLILib 


import os, sys
import numpy as np

import seasonal 

from sort import sort_snap_files

from datetime import datetime

SEASONS = {
    "winter": [10, 11, 12, 13, 14, 15],
    "early-winter": [10, 11],
    "summer": [4, 5, 6, 7, 8, 9],
    "late-summer": [8, 9],
}


def utility ():
    """
    Utility for creating seasonal precipitation products by summing monthly
    raster files. 

    Flags
    -----
        --monthly-data: path
            input monthly raster data
        --out-directory: path
            Directory to save results at
        --method: string
            monthly, or roots
            if monthly, --season must be used also
            if roots,  --roots-directory or  --roots-file must be used also
                --season-length may also be used with roots method
        --season: string
            Season to sum precipitation for. Seasons are defined below 
            with inclusive month ranges
            'winter' for Oct(year N) - Mar(year N+1)
            'early winter' for Oct - Nov
            summer for Apr - Sept

        --roots-directory:
            directory containing roots data in tiff format in files that
            can be sorted correctly using pythons sorted methdod
        --roots-file:
            roots file multigird
        --season-length: int
            length in days of season to use with roots.  
        --out-format:
            tiff, or multogrid
        --sort-method: string, optional
            sort method of input monthly precipitation data
            'default' to use Python's `sorted` method or,
            'snap' use a function to sort SNAP raster files which are
            named '*_MM_YYYY.*'
        
        
    Example
    -------
        python spc/utility.py --monthly-data=./ --out-directory=./ 
            --season=winter --sort-method=snap
    """
    try:
        arguments = CLILib.CLI(
            ['--monthly-data', '--out-directory'],
            [
            '--sort-method', '--season', '--method',
            '--roots-directory', '--roots-file',
            '--season-length', '--out-format', 
            ]   
        )
    except (CLILib.CLILibHelpRequestedError, CLILib.CLILibMandatoryError) as E:
        print (E)
        print(utility.__doc__)
        return

    if arguments['--sort-method']is None:
        sort_method = "Using default python sort function"
        sort_fn = sorted
    elif  arguments['--sort-method'].lower() == 'snap':
        sort_method = "Using SNAP sort function"
        sort_fn = sort_snap_files
    elif not arguments['--sort-method']is None and\
        arguments['--sort-method'].lower() != "default":
        print("invalid --sort-method option")
        print("run utility.py --help to see valid options")
        print("exiting")
        return

    monthly = seasonal.load_monthly_data(
        arguments['--monthly-data'], sort_func=sort_fn
    )

    if arguments['--method'] is None or arguments['--method'] == 'monthly':
        season = arguments['--season']
        name = season + '-precipitation'
        summed = seasonal.sum_precip(monthly, SEASONS[season], name=name)

        summed.config['description'] = name + " for months " +\
            str(SEASONS[season]) +\
            "(month '13' is january for the next year, '14' is feb, etc)"
        summed.config['raster_metadata'] = monthly.config['raster_metadata']
    elif arguments['--method'] == 'roots':

        if arguments['--roots-file']:
            print(' roots file')
            roots = TemporalGrid(arguments['--roots-file'])
        elif arguments['--roots-directory']:
            load_params = {
            "method": "tiff",
            "directory": arguments['--roots-directory'],
            "sort_func": sorted
            }
            create_params = {
                "name": 'roots',
                "start_timestep": 0,
            }

            roots = tools.load_and_create(
                load_params, create_params
            )
        else:
            print ("'--roots-file' or '--roots-directory' must be"
                   " suppied for 'roots' method"
            )

        key = list(monthly.config['grid_name_map'].keys())[0]
        # print(key)
        year, month = [int(i) for i in key.split('-')]
        start_date = datetime(year, month, 1)
        # print(start_date)

        try:
            season_name, season_length = arguments['--season-length'].split('-')
        except ValueError:
            season_name = arguments['--season-length']
            season_length = None

        if season_name == 'summer':
            dates = seasonal.root_mg_to_date_mg(roots, start_date) # don't need to skip ends
        elif  season_name == 'winter':
            dates = seasonal.root_mg_to_date_mg(roots, start_date, 1, 1)
        # print(dates)

        # try:
        monthly.config['start_timestep'] = 0
        print(monthly.config['description'])
        summed = seasonal.sum_seasonal_2(
            monthly, dates, season_length, season_name
        ) 
        # except IndexError as e:
        #     print (e)
        #     sys.exit()

        summed.config['start_timestep'] = year
        summed.config['raster_metadata'] = monthly.config['raster_metadata']

        # import matplotlib.pyplot as plt
        # plt.imshow(summed[1950])
        # plt.colorbar()
        # plt.show()

    else:
        print ("'--method' invalid -> %s" % str(arguments['--method']))




    try:
        os.makedirs(arguments['--out-directory'])
    except:
        pass
    if arguments['--out-format'] is None or \
            arguments['--out-format'] == 'tiff' :
        summed.save_all_as_geotiff(arguments['--out-directory'])
    elif arguments['--out-format'] == 'multigrid':
        summed.config['command-used-to-create'] = ' '.join(sys.argv)
        summed.save(
            os.path.join(
                arguments['--out-directory'], 
                'seasonal-precip.yml'
            )
        )
    


utility()
