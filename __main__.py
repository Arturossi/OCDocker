#!/usr/lib/python3

# Imports
###############################################################################
import os
import glob
from multiprocessing import cpu_count

from OCDocker.Initialise import *
from OCDocker.UpdateDatabases import update_databases

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Description
###############################################################################

# Classes
###############################################################################

# Functions
###############################################################################

# Main Function
###############################################################################
def main():

    args = initial_args

    if args.update is True:
        print(tw.dedent(f"""
                                         !WARNING!
                      You have chosen to update the local databases.
              ** The root directory for the database files is: """+clrs['y']+ocdb+clrs['n']+"""
              ** The path to local pdb bind mirror is: """+clrs['y']+pdbbind_archive+clrs['n']+"""
              This could take a long time.
              <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              """))
        option = input('Do you confirm the information above? (y/n)')
        if option.lower()  in ['y', 'ye', 'yes']:
            update_databases(args.verbosity)
            print('\n\nDone updating all databases. Exiting.\n')
        else:
            print('\n\nNo positive confirmation, will not update databases.\n')
            exit()
    else:
        pass

# Execute
###############################################################################
if __name__ == "__main__":
    main()
