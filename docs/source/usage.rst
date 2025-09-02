Usage
=====

This section provides instructions on how to use OCDocker.

Basic Usage
-----------

After installing OCDocker, you can use it to perform various tasks.

To run OCDocker, use the following command:

.. code-block:: bash

   ocdocker [options] [arguments]

Example
-------

Here is an example of how to use OCDocker:

.. code-block:: bash

   ocdocker --help

Common commands
---------------

- Virtual screening for a single entry (receptor/ligand/box):

  .. code-block:: bash

     ocdocker vs --engine vina \
       --receptor path/to/receptor.pdb \
       --ligand path/to/ligand.smi \
       --box path/to/box0.pdb \
       --outdir runs/exp1

- Create a configuration file interactively:

  .. code-block:: bash

     ocdocker init-config

- Run SHAP analysis (delegates to OCScore SHAP CLI):

  .. code-block:: bash

   ocdocker shap --help

- Full pipeline (multi-engine, clustering, rescoring):

  .. code-block:: bash

     ocdocker pipeline \
       --receptor path/to/receptor.pdb \
       --ligand path/to/ligand.smi \
       --box path/to/box0.pdb \
       --engines vina,smina,plants \
       --outdir runs/exp1

- Open interactive console (with tab-completion and history):

  .. code-block:: bash

     ocdocker console --conf OCDocker.cfg

  Inside the console:

  .. code-block:: pycon

     >>> print_args()               # environment overview
     >>> print_args('all')          # all sections
     >>> print_args('vina')         # also: smina, plants, gnina, oddt, db, paths
     >>> debug_all()                # enable global DEBUG level
     >>> debug_modules('vina,smina')# per-module debug prints

This command displays the help message with all available options and arguments.

Advanced Usage
--------------

For more advanced usage, refer to the documentation of specific modules/packages:

- :doc:`OCDocker`
- :doc:`OCDocker.DB`
- :doc:`OCDocker.Docking`
- :doc:`OCDocker.Error`
- :doc:`OCDocker.Initialise`
- :doc:`OCDocker.Ligand`
- :doc:`OCDocker.OCScore`
- :doc:`OCDocker.Processing`
- :doc:`OCDocker.Receptor`
- :doc:`OCDocker.Rescoring`
- :doc:`OCDocker.Toolbox`

Debugging flags
---------------

All subcommands accept debug flags:

.. code-block:: bash

   # Global DEBUG (equivalent to --output-level 5)
   ocdocker vs ... --debug-all

   # Module-specific debug (forces verbose prints for matching modules)
   ocdocker pipeline ... --debug-modules vina,smina

Modules are matched case-insensitively against file/module names. Use a comma-separated list or 'all'.
