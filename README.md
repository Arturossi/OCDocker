###############################################
OCDocker installation instructions Step-by-step
###############################################

*****
LINUX
*****

Download and install MGLTools
=============================

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

	- Go to the website and download MGLTools Linux GUI installer https://ccsb.scripps.edu/download/
	- Install it

* Option 2 (For those who love to follow each step)

	- Go to the website and download MGLTools Linux tarball installer https://ccsb.scripps.edu/download/
	- Untar it:
		.. code-block:: bash

			$ tar -xvzf mgltools_x86_64Linux2_1.5.X.tar_.gz
	- cd into created dir
		.. code-block:: bash

			$ cd mgltools_x86_64Linux2_1.5.X
	- source the install.sh
		.. code-block:: bash

			$ source ./install.sh

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

	.. code-block:: bash

		$ wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz && mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz && cd mgltools && source ./install.sh

OBS: The scripts used to prepare ligand/receptor will be in the following dir: ``installation_dir/mgltools/MGLToolsPckgs/AutoDockTools``


Download and install Autodock VINA
==================================

To install it, you have 2 options:

* Option 1 (For those who love to follow each step)

	- Go to the website http://vina.scripps.edu/download.html and download the Linux installer (tgz)
	- Untar it:
		.. code-block:: bash

			$ tar -xvzf autodock_vina_1_1_2_linux_x86.tgz

* Option 2 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)
	.. code-block:: bash

		$ wget http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz -O vina.tgz && mkdir -p vina && tar -xvzf vina.tgz -C vina --strip-components=1 && rm vina.tgz

OBS: The vina executable will be in the following dir: ``installation_dir/vina/bin``


Download and install SMINA
==========================

nothing here but us


Download and install PLANTS
===========================

Go to http://www.tcd.uni-konstanz.de/plants_download/ and demand a license
