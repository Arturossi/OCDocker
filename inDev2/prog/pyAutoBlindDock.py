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
import glob
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
            print(f"The dir {dirname} already exists, aborting its creation")
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

protein=sys.argv[1]
ligand=sys.argv[2]
PocketNum=sys.argv[3]
userDirName=sys.argv[4]

protein="/home/ligmol/Downloads/docking/receptor.pdb"
ligand="/home/ligmol/Downloads/docking/actives/active1.mol2"
PocketNum="5"
userDirName="aaa"

sec, mins, hour, day, mon, year = datetime.now().strftime("%S:%M:%H:%d:%m:%Y").split(':')

var = 5 # scale factor for calculating docking box

# split in \ and . but not in ..
lig_name = list(filter(None, re.split('/|(?<!\.)\.(?!\.)', ligand)))
pro_name = list(filter(None,re.split('/|(?<!\.)\.(?!\.)', protein)))

#print(f"{ligand} turns into {lig_name}\n{protein} turns into {pro_name}\n") # DEBUG

mol_name = f"{lig_name[-2]}:{pro_name[-2]}"
userDirPath = f"./dock_file/{userDirName}"

safe_create_dir(userDirPath)

dock_time = f"{year}{mon}{day}{hour}{mins}{sec}"
dock_file = f"{mol_name}:{year}{mon}{day}{hour}{mins}{sec}"
outf      = f"{userDirPath}/{mol_name}{year}{mon}{day}{hour}{mins}{sec}"
logfile   = f"{dock_file}_log.txt"
errfile   = f"{dock_file}_err.txt"
config    = "config.txt"
progPath  = "."
temfile   = f"{outf}/tem_1.txt"

conf = {}

safe_create_dir(outf)

with open(f'{outf}/{errfile}', 'w') as _:
    pass

with open(f'{outf}/status.txt', 'w') as f:
    f.write(f"{year}-{mon}-{day} {hour}:{mins}:{sec}\n")

with open(f'{outf}/{dock_time}_run.txt', 'w') as f:
    f.write(f"{dock_file}  {PocketNum}  ")

pro_ori = os.path.join(*pro_name[:-1])
if protein.startswith("/"):
    pro_ori = f"/{pro_ori}"
elif protein.startswith("./"):
    pro_ori = f"./{pro_ori}"
pro_format = f"{pro_ori}_format.pdb"

subprocess.run([f"{progPath}/FormatPDB_Simple", protein, pro_format])

if os.path.isfile(pro_format):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("protein format success!\n")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("protein format error!\n")

    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

if lig_name[-1] != "mol2":
    ligand = f"{ligand}.mol2"
    subprocess.run(["obabel", f"-i{lig_name[-1]}", f"{ligand}", "-omol2", "-O", ligand, "-p", "7"])

    if os.path.isfile(ligand):
        with open(f'{outf}/{logfile}', 'a') as f:
            f.write("ligand transfer success!\n")
    else:
        with open(f'{outf}/{errfile}', 'a') as f:
            f.write("ligand transfer error!\n")

        with open(f'{outf}/status.txt', 'w') as _:
            pass

        make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
        exit()

shutil.move(f"{pro_format}", f"{outf}/receptor.pdb")
shutil.copy(f"{ligand}", f"{outf}/ligand.mol2")

#print(f"Copy from {pro_format} to {outf}/receptor.pdb") # DEBUG
#print(f"Copy from {ligand} to {outf}/ligand.mol2") # DEBUG

####################################################################
# curvatureSurface       
###################################################################
grid_filepath = f"{outf}/grid.pdb"

subprocess.run([f"{progPath}/curvatureSurface/bin/curvatureSurface", f"{outf}/receptor.pdb", grid_filepath])

if os.path.isfile(grid_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("curvature calculate success!\n")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("curvature calculate error!\n")

    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

####################################################################
# clusters       
###################################################################
with open(f"{outf}/conf.txt", "w") as f:
    subprocess.run([f"{progPath}/clusters", f"{outf}/grid.pdb", f"{PocketNum}"], stdout=f)

####################################################################
# ADT_scripts: prepare_ligand4.py, prepare_receptor4.py, prepare_dpf4.py, eBoxSize.pl;                   
# prepare_ligand4.py——The imported ligands are converted from MOL2 format to pdbqt format;          
# prepare_receptor4.py——Converts the inputted receptor from pdb format to pdbqt format;         
# prepare_dpf4.py——Find the center of the docking pocket, used in Re-Docking;                                 
# eBoxSize.pl——Calculate the size of the ligand;     
####################################################################
lig_filepath = f"{outf}/ligand.pdbqt"
subprocess.run([f"{progPath}/ADT_scripts/prepare_ligand4.py", "-l", f"{outf}/ligand.mol2", "-C", "-o", lig_filepath])

if os.path.isfile(lig_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("ligand pdbqt transfer success!\n")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("ligand pdbqt transfer error!\n")

    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

subprocess.run([f"{progPath}/ADT_scripts/prepare_receptor4.py", "-r", f"{outf}/receptor.pdb", "-o", f"{outf}/receptor.pdbqt", "-A", "hydrogens", "-U", "nphs_lps_waters"])
    
pro_filepath = f"{outf}/receptor.pdbqt"

if os.path.isfile(pro_filepath):
    with open(f'{outf}/{logfile}', 'a') as f:
        f.write("receptor pdbqt transfer success!\n")
else:
    with open(f'{outf}/{errfile}', 'a') as f:
        f.write("receptor pdbqt transfer error!\n")

    with open(f'{outf}/status.txt', 'w') as _:
        pass

    make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
    exit()

with open(temfile, "w") as f:
    subprocess.run(["perl", f"{progPath}/ADT_scripts/eBoxSize.pl", f"{outf}/ligand.pdbqt"], stdout=f)

#####################ReDocking and GlobalDocking########################
#（2）get the ligand size;
nm=9
ex=4
sx=0

try:
    with open(temfile, "r") as f:
        sx = float(f.readlines()[-1].strip())
except Exception as e:
    print(f"Error in opening tem_1.txt\nError: {e}")

if os.path.exists(temfile):
  os.remove(temfile)
else:
  print("The {temfile} file does not exist")

sx = sx + 1

####################################################################    
#LocalDocking
with open(f"{outf}/{config}", "w") as f:
    f.write("Cavities  volume  center_x  center_y  center_z  size_x  size_y  size_z  score\n")
    
try:
    with open(f"{outf}/conf.txt", "r") as f:
        for line in f:
            if line.startswith("Cavities"):
                continue
                
            array = line.split()
            
            num = array[0]
            _cx = float(array[1])
            _cy = float(array[2])
            _cz = float(array[3])
            _sx = float(array[4])
            
            if _sx >= sx:
                if _sx - sx <= 2 * var:
                    _sx = sx + 2 * var
                else:
                    _sx = _sx + 5
            else:
                _sx = sx + 2 * var
                
            _sy = float(array[5])
            
            if _sy >= sx:
                if _sy - sx <= 2 * var:
                    _sy = sx + 2 * var
                else:
                    _sy = _sy + 5
            else:
                _sy = sx + 2 * var
                
            _sz = float(array[6])
            
            if _sz >= sx:
                if _sz - sx <= 2 * var:
                    _sz = sx + 2 * var
                else:
                    _sz = _sz + 5
            else:
                _sz =  sx + 2 * var
                
            out_ligand = f"{mol_name}_out_{num}.pdbqt"
            conf[num]=f"{num}  {float(array[7])}  {_cx}  {_cy}  {_cz}  {_sx}  {_sy}  {_sz}  "
            
            with open(f'{outf}/{logfile}', 'a') as fp:
                fp.write(f"Calculate {num} --LocalDocking!\n")
                
            with open(f'{outf}/{logfile}', 'a') as fp:
                lig_outfilepath = f"{outf}/{out_ligand}"
                
                subprocess.run([f"{progPath}/vina", "--receptor", f"{outf}/receptor.pdbqt", "--ligand", f"{outf}/ligand.pdbqt", "--center_x", str(_cx), "--center_y", str(_cy), "--center_z", str(_cz), "--size_x", str(_sx), "--size_y", str(_sy), "--size_z", str(_sz), "--num_modes", str(nm), "--exhaustiveness", str(ex), "--out", lig_outfilepath], stdout=fp)
                
            if os.path.isfile(lig_outfilepath):
                with open(f'{outf}/{logfile}', 'a') as fp:
                    fp.write(f"docking in the {num} th cavity success!\n")
            else:
                with open(f'{outf}/{errfile}', 'a') as fp:
                    fp.write("echo docking error!\n")
                    
                with open(f'{outf}/status.txt', 'w') as _:
                    pass
                    
                make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)
                exit()
except Exception as e:
    print(f"Opening error!:{e}")

#######################################################################
#process the output files
os.remove(f"{outf}/receptor.pdbqt")
os.remove(f"{outf}/ligand.pdbqt")
os.remove(f"{outf}/conf.txt")

pdbqt = glob.glob(f"{outf}/*.pdbqt")

for out_ligand in pdbqt:
    print(out_ligand)
    
    subprocess.run(['obabel', '-ipdbqt', out_ligand, '-omol2', '-O', out_ligand.replace(".pdbqt", ".mol2")])
    #print(" ".join(['obabel', '-ipdbqt', out_ligand, '-omol2', '-O', out_ligand.replace(".pdbqt", ".mol2")]))
    os.remove(out_ligand)

#os.remove(f"{outf}/grid.pdb")

###################################################################
#get the vina score of the first pose
data = []
buff = []
num = 0

try:
    with open(f'{outf}/{logfile}', 'r') as f:
        for line in f:
            if line.startswith("Calculate"):
                temp = line.split()
                buff = []
                buff.append(temp[1])
            if line.startswith("   1"):
                temp = line.split()
                buff.append(temp[1])
                data.append(buff)
                num = num + 1

except Exception as e:
    print(f"Error in opening logfile\nException: {e}\n")

for i in range(num):
    conf[str(i + 1)] = f"{conf[str(i + 1)]}{data[i][1]}"
    with open(f"{outf}/{config}", "w") as f:
        f.write(conf[str(i + 1)])

mol2 = pdbqt

for i in range(len(mol2)):
    mol2[i] = f"{os.path.splitext(mol2[i])[0]}.mol2"
    tmp = list(filter(None, re.split('/|\.|_', mol2[i])))
    for m in range(num):
        if tmp[-2] == data[m][0]:
            out_ligand_score = mol2[i].replace(".mol2", f"{data[m][1]}.mol2")
            os.rename(mol2[i], out_ligand_score)

####################################################################
#Finishing
with open(f'{outf}/status.txt', 'w') as _:
    pass

make_tarfile(f"{userDirPath}/{dock_file}.tar.gz", outf)

print("DONE!")
