#! /bin/sh
if [ ! "$#" -gt 1 ]
  then
    echo "No arguments supplied"
	echo "Argument 1: full path to the python executable of the mgltools"
	echo "E.g.: /mgltools/bin/python"
	echo "Argument 2: full path to the AutoDock Vina executable"
	echo "E.g.: /bin/vina"
	echo "Make sure MGL TOOLS are installed before executing"
	exit 1
fi

cp $2 ./prog/vina

p="\#\!$1"
sed -i "1s:.*:${p}:" ./prog/ADT_scripts/prepare_dpf4.py
sed -i "1s:.*:${p}:" ./prog/ADT_scripts/prepare_ligand4.py
sed -i "1s:.*:${p}:" ./prog/ADT_scripts/prepare_receptor4.py

mkdir -p dock_file
chmod 777 dock_file

