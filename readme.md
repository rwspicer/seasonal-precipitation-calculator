# README

Tool for calculating seasonal precipitation 

This project is licensed under the MIT licence. This project includes a copy of 
multigrids, and code based on  `atm.tools.initiation_areas.py` from the 
[atm project](https://github.com/ua-snap/arctic_thermokarst_model) which is 
licensed under the MIT licence. 


## Utility help
```
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
```
