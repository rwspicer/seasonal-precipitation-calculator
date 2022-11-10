"""
Seasonal Precipitation Calculator Utility 
-----------------------------------------

Utility for creating seasonal precipitation products by summing monthly
raster files. 

"""
from multigrids.temporal_grid import TemporalGrid
from multigrids import tools
from spicebox import CLILib 


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
            if roots,  --roots-directory or  --roots-file must be used also
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
                {'required': False, 'type': bool, 'default': False },
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

    if arguments['--save-temp-monthly']:
        save_temp = True


    if verbose:
        print('Loading Monthly data ...')
    monthly = subutilities.load_monthly_data(
        arguments['--monthly-data'], 
        sort_func=sort_fn, save_temp = save_temp, verbose=verbose
    )


    ## sum data
    if arguments['--method'] is None or arguments['--method'] == 'monthly':
        if verbose:
            print('Using Monthly Sums method')
        summed = subutilities.method_monthly(arguments, monthly, verbose)

    elif arguments['--method'] == 'roots':

        if arguments['--roots-file']:
            roots = subutilities.load_roots_data(
                arguments['--roots-file'], 
                sort_func=sort_fn, save_temp = save_temp, verbose=verbose
            )
        else:
            print ("'--roots' must be supplied for 'roots' method"
            )
            print('exiting')
            return
        if verbose:
            print('Using Sum Between roots method')
        summed = subutilities.method_roots(arguments, monthly, roots, verbose)
        

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
