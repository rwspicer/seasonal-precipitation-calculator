"""
Seasonal Precipitation Calculator Utility 
-----------------------------------------

Utility for creating seasonal precipitation products by summing monthly
raster files. 

"""
from multigrids.temporal_grid import TemporalGrid
from multigrids import tools
from spicebox import CLILib 
import numpy as np


import os, sys
# import numpy as np

import seasonal 
import subutilities

from sort import sort_snap_files

from datetime import datetime




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
            if roots,  --roots must be used to specify the directory or 
                multigrd file that contains the roots also
                --season-length may also be used with roots method
        --season: string
            Season to sum precipitation for. Seasons are defined below 
            with inclusive month ranges
            'winter' for Oct(year N) - Mar(year N+1)
            'early winter' for Oct - Nov
            summer for Apr - Sept

        --roots:
            roots file multigird, or directort of roots tiff files
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
    flags = {
            '--monthly-data': 
                {'required': True, 'type': str}, 
            '--out-directory': 
                {'required': True, 'type': str},
            "--sort-method": 
                {'required': False, 'type': str, 'default': 'default', 
                'accepted-values':['default','snap']},
            "--season" : 
                {'required': False, 'type': str, 
                'accepted-values':list(subutilities.SEASONS.keys())}, 
            "--method": 
                {
                    'required': False, 
                    'type': str, 
                    'default':'monthly',
                    'accepted-values': ['roots', 'monthly']
                }, 
            "--roots": 
                {'required': False, 'type':str},
            '--season-length': 
                {'required': False, 'type': str},
            '--out-format': 
                {
                    'required': False, 'default': 'tiff', 'type': str, 
                    'accepted-values':['tiff','multigrid']
                },
            '--verbose': 
                {
                    'required': False, 'type': str, 'default': '', 
                    'accepted-values': ['', 'log', 'warn']
                },
            '--save-temp-monthly':
                {'required': False, 'type': str, 'default': 'False' },
            '--save-temp-roots':
                {'required': False, 'type': str, 'default': 'False' },
            '--start-at':
                {'required': False, 'type': int, 'default': 0 },

        }
    try:
        arguments = CLILib.CLI(flags)
    except (CLILib.CLILibHelpRequestedError, CLILib.CLILibMandatoryError) as E:
        print (E)
        print(utility.__doc__)
        return

    verbose = arguments['--verbose'] != ''

    

    sort_fn = subutilities.get_sort_method(arguments, verbose)
    if sort_fn is None:
        print('exiting')
        return 


    
    save_temp = subutilities.get_save_temp_status(
        arguments['--save-temp-monthly']
    )



    if verbose:
        print('Loading Monthly data ...')
    monthly = subutilities.load_monthly_data(
        arguments['--monthly-data'], 
        sort_func=sort_fn, save_temp = save_temp, verbose=verbose
    )

    

    try:
        os.makedirs(arguments['--out-directory'])
    except:
        pass
    # if arguments['--out-format'] is None or \
    #         arguments['--out-format'] == 'tiff' :
    #     ## do this later
    #     # summed.save_all_as_geotiff(arguments['--out-directory'])
    #     pass
    # elif arguments['--out-format'] == 'multigrid':
    #     summed.config['command-used-to-create'] = ' '.join(sys.argv)
    #     summed.save(
    #         os.path.join(
    #             arguments['--out-directory'], 
    #             'seasonal-precip.yml'
    #         )
    #     )

    grid_shape = monthly.config['grid_shape']
    n_years = monthly.config['num_grids']//12

    
    outpath=None
    if arguments['--out-format'] == 'multigrid':
        outpath = os.path.join(
            arguments['--out-directory'], 
            'seasonal-precip.yml'
        )

    ## TODO LOAD ON RE LAUNCH
    summed = TemporalGrid(
        grid_shape[0], # rows
        grid_shape[1], # cols
        n_years, # years
        data_type = monthly.grids.dtype, 
        dataset_name = 'sum-precip-temp-ds-name',
        description = "seasonal precip summed by ?",
        mask = monthly.config['mask'],
        # initial_data = precip_sum,
        start_timestep = 0,
        save_to=outpath
    )

    # ## TODO LOAD ON RE LAUNCH
    # for row in range(summed.grids.shape[0]):
    #     summed.grids[row][:] = np.nan

    

    ## sum data
    if arguments['--method'] is None or arguments['--method'] == 'monthly':
        if verbose:
            print('Using Monthly Sums method')
        summed = subutilities.method_monthly(
            arguments, monthly, summed, verbose
        ) # TODO use SUMMED ARG IN THIS FUNC

    elif arguments['--method'] == 'roots':


        if arguments['--roots']:
            if os.path.isdir(arguments['--roots']):
                save_temp = subutilities.get_save_temp_status(
                    arguments['--save-temp-roots']
                )
            else:
                save_temp=None
            roots = subutilities.load_roots_data(
                arguments['--roots'], 
                sort_func=sort_fn, save_temp = save_temp, verbose=verbose
            )
        else:
            print ("'--roots' must be supplied for 'roots' method"
            )
            print('exiting')
            return


        if verbose:
            print('Using Sum Between roots method')
        summed = subutilities.method_roots(
            arguments, monthly, summed, roots, verbose
        )
        

    else:
        print ("'--method' invalid -> %s" % str(arguments['--method']))




    # try:
    #     os.makedirs(arguments['--out-directory'])
    # except:
    #     pass
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
