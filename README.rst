OCDocker installation instructions Step-by-step
===============================================

![OCDocker](./OCDocker.png "OCDocker")

Download and install MGLTools
-----------------------------

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

	.. code-block:: bash
		$ wget https://ccsb.scripps.edu/download/292/ --no-check-certificate -O mgltools_install

* Option 2 (For those who love to follow each step)

	- Download the file

	.. code-block:: bash
		$ wget https://ccsb.scripps.edu/download/532/ --no-check-certificate -O mgltools_install.tar.gz

	- Untar it:

	.. code-block:: bash
		$ tar -xvzf mgltools_install.tar.gz

	- cd into created dir

	.. code-block:: bash
		$ cd mgltools_x86_64Linux2_1.5.X

	- source the install.sh

	.. code-block:: bash
		$ source ./install.sh

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

.. code-block:: bash
	$ wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz --no-check-certificate && mkdir -p mgltools && tar -xvzf mgltools_install.tar.gz -C mgltools --strip-components=1 && rm mgltools_install.tar.gz && cd mgltools && source ./install.sh

OBS: The scripts used to prepare ligand/receptor will be in the following dir: ``<installation_dir>/mgltools/MGLToolsPckgs/AutoDockTools``

> :warning: **Still cannot run MGLTools?**: If you are facing some shady problems such as the numpy one, you might have to compile MGLTools from source. You can download it at https://github.com/genome-vendor/MGLtools (Still not sure about its version... I do not know if it is 1.5.6 or 1.5.4)

Download and install ADFRtools
------------------------------

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

	.. code-block:: bash
		$ wget https://ccsb.scripps.edu/adfr/download/1028/ --no-check-certificate -O adfr_install

* Option 2 (For those who love to follow each step)

	- Download the file

	.. code-block:: bash
		$ wget https://ccsb.scripps.edu/adfr/download/1038/ --no-check-certificate -O adfr_install.tar.gz

	- Untar it and rename it (to look nicer):

	.. code-block:: bash
		$ tar -xvzf adfr_install.tar.gz -C ADFRsuite

	- cd into created dir

	.. code-block:: bash
		$ cd ADFRsuite

	- source the install.sh

	.. code-block:: bash
		$ source ./install.sh

	- export the variable to the path

	.. code-block:: bash
		$ echo "PATH=`pwd`/bin:"'$PATH' >> ~/.bashrc

	- source the bashrc
	
	.. code-block:: bash
		$ source ~/.bashrc

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

.. code-block:: bash
	$ wget https://ccsb.scripps.edu/adfr/download/1028/ --no-check-certificate -O adfr_install && mkdir -p mgltools && tar -xvzf adfr_install.tar.gz -C ADFRsuite --strip-components=1 && rm adfr_install.tar.gz && cd ADFRsuite && source ./install.sh && echo "PATH=`pwd`/bin:"'$PATH' >> ~/.bashrc && source ~/.bashrc


Install DSSP
=========================

To install DSSP in Ubuntu 18.04+:

.. code-block:: bash

	$ sudo apt install dssp

As default, the dssp path will be '/usr/bin/dssp'.


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

		$ mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina && sudo cp vina/vina /usr/bin/vina

OBS: The vina executable will be in the following dir: ``installation_dir/vina/bin``


Download and install SMINA
==========================

First of all make sure that you have all required libs installed (openbabel must be v3+).

.. code-block:: bash

	$ sudo apt install git libboost-all-dev libopenbabel-dev build-essential libeigen3-dev openbabel

Now clone the smina repo then enter it, create a build folder, enter the build folder, perform the cmake using the parent folder as the source and finally use the make with 12 jobs (you can increase/decrease the number of jobs if you want, but 12 is what is written in smina's doc).

.. code-block:: bash

	$ git clone https://git.code.sf.net/p/smina/code smina-code && cd smina-code && mkdir build && cd build && cmake .. && make -j12


Download and install PLANTS
===========================

Go to http://www.tcd.uni-konstanz.de/plants_download/ and demand a license
