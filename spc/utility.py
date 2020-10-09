"""
Seasonal Precipitation Calculator Utility 
-----------------------------------------

Utility for creating seasonal precipitation products by summing monthly
raster files. 

"""
import CLILib 

import glob
import os

import seasonal 

from sort import sort_snap_files

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
        --monthly-data-directory: path
            input monthly raster data
        --out-directory: path
            Directory to save results at
        --season: string
            Season to sum precipitation for. Seasons are defined below 
            with inclusive month ranges
            'winter' for Oct(year N) - Mar(year N+1)
            'early winter' for Oct - Nov
            summer for Apr - Sept
        --sort-method: string, optional
            'default' to use Python's `sorted` method or,
            'snap' use a function to sort SNAP raster files which are
            named '*_MM_YYYY.*'
    
        
    Example
    -------
        python spc/utility.py --monthly-data-directory=./ --out-directory=./ 
            --season=winter --sort-method=snap
    """
    try:
        arguments = CLILib.CLI(
            ['--monthly-data-directory', '--out-directory', '--season'],
            ['--sort-method']   
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
        arguments['--monthly-data-directory'], sort_func=sort_fn
    )

    season = arguments['--season']
    name = season + '-precipitation'
    summed = seasonal.sum_precip(monthly, SEASONS[season], name=name)

    summed.config['description'] = name + " for months " +\
        str(SEASONS[season]) +\
        "(month '13' is january for the next year, '14' is feb, etc)"
    summed.config['raster_metadata'] = monthly.config['raster_metadata']

    try:
        os.makedirs(arguments['--out-directory'])
    except:
        pass
    summed.save_all_as_geotiff(arguments['--out-directory'])


utility()
