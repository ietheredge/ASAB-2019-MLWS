===================
Installing on Linux
===================

#. Download the installer:

   * `Miniconda installer for Linux <https://conda.io/miniconda.html>`_.

   * `Anaconda installer for Linux <https://www.anaconda.com/download/>`_.

#. In your terminal window, run:

   * Miniconda:

     .. code::

        bash Miniconda3-latest-Linux-x86_64.sh

   * Anaconda:

     .. code::

        bash Anaconda-latest-Linux-x86_64.sh

#. Follow the prompts on the installer screens.

   If you are unsure about any setting, accept the defaults. You
   can change them later.

#. To make the changes take effect, close and then re-open your
   terminal window.

#. To test your installation, in your terminal window or Anaconda Prompt, run the command ``conda list``.

For a successful installation, a list of installed packages
appears.

.. _install-linux-silent:

Using with fish shell
=========================

To use conda with fish shell, run the following in your terminal: 

  ``conda init fish``

Installing in silent mode
=========================

See the instructions for
:ref:`installing in silent mode on macOS <install-macos-silent>`.


Updating Anaconda or Miniconda
==============================

#. Open a terminal window.

#. Run ``conda update conda``.


Uninstalling Anaconda or Miniconda
==================================

#. Open a terminal window.

#. Remove the entire miniconda install directory with::

     rm -rf ~/miniconda

#. OPTIONAL: Edit ``~/.bash_profile`` to remove the Miniconda
   directory from your PATH environment variable.

#. OPTIONAL: Remove the following hidden file and folders that
   may have been created in the home directory:

   * ``.condarc`` file
   * ``.conda`` directory
   * ``.continuum`` directory

   By running::

     rm -rf ~/.condarc ~/.conda ~/.continuum

