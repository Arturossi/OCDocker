OCDockerConsole
===============

The interactive console aggregates common OCDocker imports in a single session and supports:

- Tab-completion for names and attributes
- Command history (stored in ``~/.ocdocker_console_history``)

Launch the console via CLI:

.. code-block:: bash

   ocdocker console --conf OCDocker.cfg

Inside the console, useful helpers include:

.. code-block:: pycon

   >>> print_args()                 # environment overview
   >>> print_args('all')            # all sections
   >>> print_args('vina')           # also: smina, plants, gnina, oddt, db, paths
   >>> debug_all()                  # enable global DEBUG level
   >>> debug_modules('vina,smina')  # per-module debug prints

Reference
---------

.. automodule:: OCDockerConsole
   :members:
   :undoc-members:
   :show-inheritance:
