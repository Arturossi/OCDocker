#!/usr/bin/python3

# Description
###############################################################################
'''
pyAutoBlindDock.py is a python version, written by Artur Rossi, of the 
AutoBlindDock.pl originally written by Yang Liu, et al. If using this script,
please cite their amazing job:

Yang Liu, Maximilian Grimm, et al . CB-Dock: a web server for cavity detection-guided protein–ligand blind docking. Acta Pharmacologica Sinica, 2019.

Yang Cao , Lei Li. Improved protein-ligand binding affinity prediction by using a curvature dependent surface area model, . Bioinformatics, 2014.
'''

import os
import re
import sys
import shutil
import tarfile
import subprocess
from datetime import datetime

def safe_create_dir(dirname):
    '''
    Function to create a dir if not exists
    Input:
     dirname      [string]  - File path to be untarred
    Return:
      0 if success
      1 if folder exists
     -1 if any problem has occurred
     -2 should not appear
    '''

    try:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            return 0
        else:
            print(f"The dir {userDirPath} already exists, aborting")
            #exit(1)
            return 1
    except Exception as e:
        print(f"Error! Exception: {e}")
        exit(-1)
    return -2

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if len(sys.argv) != 5:
      print("***************************************************************");
      print("* Error in input data!                                        *");
      print("* ./pyAutoDock.py [Receptor] [Ligand] [PocketNum] [UserDir]   *");
      print("* Example: ./AutoBlindDock.py receptor.pdb ligand.mol2 5 test *");
      print("***************************************************************")
      exit(0)

protein=sys.argv[1];
ligand=sys.argv[2];
PocketNum=sys.argv[3];
userDirName=sys.argv[4];

sec, mins, hour, day, mon, year = datetime.now().strftime("%S:%M:%H:%d:%m:%Y").split(':')

var = 5; # scale factor for calculating docking box

lig_name = re.split('/|\.', ligand)
pro_name = re.split('/|\.', protein)

mol_name = f"{lig_name[-2]}:{pro_name[-2]}";
userDirPath = f"./dock_file/{userDirName}";

safe_create_dir(userDirPath)

dock_time = f"{year}{mon}{day}{hour}{mins}{sec}"
dock_file = f"{mol_name}:{year}{mon}{day}{hour}{mins}{sec}"
outf      = f"{userDirPath}/{mol_name}{year}{mon}{day}{hour}{mins}{sec}"
logfile   = f"{dock_file}_log.txt"
errfile   = f"{dock_file}_err.txt"
config    = "config.txt"
progPath  = "."

#my %conf;

safe_create_dir(outf)

with open(f'{outf}/{errfile}', 'w') as _:
    pass

with open(f'{outf}/status.txt', 'w') as f:
    f.write(f"{year}-{mon}-{day} {hour}:{mins}:{sec}\n")

with open(f'{outf}/{dock_time}_run.txt', 'w') as f:
    f.write(f"{dock_file}  {PocketNum}  ")

pro_ori = protein.split(".")[0]
pro_format = f"{pro_ori}_format.pdb"

subprocess.run([f"{progPath}/FormatPDB_Simple", protein, pro_format]) # protein format  --error info

if os.path.isfile(pro_format):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("protein format success! ")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("protein format error! ")
    
    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

if lig_name[-1] != "mol2":
    subprocess.run(["babel", f"-i{lig_name[-1]}", f"{ligand}", "-omol2", f"{ligand}.mol2", "-p", "7"]) # ligand format transfer --error info
    ligand = f"{ligand}.mol2";
        
    if os.path.isfile(ligand):
        with open(f'{outf}/{logfile}', 'a') as f:
            f.write("ligand transfer success! ")
    else:
        with open(f'{outf}/{errfile}', 'a') as f:
            f.write("ligand transfer error! ")
        
        with open(f'{outf}/status.txt', 'w') as _:
            pass

        make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
        exit()

shutil.move(f"{pro_format}", f"{outf}/receptor.pdb")
shutil.copy(f"{ligand}", f"{outf}/ligand.mol2")


####################################################################
# curvatureSurface       
###################################################################
grid_filepath = f"{outf}/grid.pdb"

subprocess.run([f"{progPath}/curvatureSurface/bin/curvatureSurface", f"{outf}/receptor.pdb", grid_filepath]) # protein format  --error info

if os.path.isfile(grid_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("curvature calculate success! ")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("curvature calculate error! ")
    
    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

####################################################################
# clusters       
###################################################################
with open(f"{outf}/conf.txt", "w") as f:
    subprocess.run([f"{progPath}/clusters", f"{outf}/grid.pdb", f"{PocketNum}"], stdout=f) # protein format  --error info

####################################################################
# ADT_scripts: prepare_ligand4.py，prepare_receptor4.py, prepare_dpf4.py, eBoxSize.pl;                       
# prepare_ligand4.py——The imported ligands are converted from MOL2 format to pdbqt format；          
# prepare_receptor4.py——Converts the inputted receptor from pdb format to pdbqt format；         
# prepare_dpf4.py——Find the center of the docking pocket, used in Re-Docking；                                 
# eBoxSize.pl——Calculate the size of the ligand;     
####################################################################
subprocess.run([f"{progPath}/ADT_scripts/prepare_ligand4.py", "-l", f"{outf}/ligand.mol2", "-C", "-o", f"{outf}/ligand.pdbqt"]) #change /home/ocean/Softwares/mgltools/bin/python automatic script

lig_filepath = "f{outf}/ligand.pdbqt"

if os.path.isfile(lig_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("ligand pdbqt transfer success! ")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("ligand pdbqt transfer error! ")
    
    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

subprocess.run([f"{progPath}/ADT_scripts/prepare_receptor4.py", "-r", f"{outf}/receptor.pdb", "-o", f"{outf}/receptor.pdbqt", "-A", "hydrogens", "-U", "nphs_lps_waters"])
    
pro_filepath = f"{outf}/receptor.pdbqt"

if os.path.isfile(pro_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("receptor pdbqt transfer success! ")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("receptor pdbqt transfer error! ")
    
    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

with open(f"{outf}/tem_1.txt", "w") as f:
    subprocess.run(["perl", f"{progPath}/ADT_scripts/eBoxSize.pl", f"{outf}/ligand.pdbqt"], stdout=f)
    
print(" ".join(["perl", f"{progPath}/ADT_scripts/eBoxSize.pl", f"{outf}/ligand.pdbqt"]))
