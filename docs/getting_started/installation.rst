Installing bayeso
#################

We recommend it should be installed in **virtualenv**.
You can choose one of three installation options.

Installing from PyPI
====================

It is for user installation.
To install the released version from PyPI repository, command it.

.. code-block:: console

    $ pip install bayeso

Compiling from Source
=====================

It is for developer installation.
To install **bayeso** from source code, command

.. code-block:: console

    $ pip install .

in the **bayeso** root.

Compiling from Source (Editable)
================================

It is for editable development mode.
To use editable development mode, command

.. code-block:: console

    $ pip install -r requirements.txt
    $ python setup.py develop

in the **bayeso** root.

Uninstalling
============

If you would like to uninstall bayeso, command it.

.. code-block:: console

    $ pip uninstall bayeso

Required Packages
=================

Mandatory pacakges are inlcuded in **requirements.txt**.
The following **requirements** files include the package list, the purpose of which is described as follows.

- **requirements-optional.txt**: It is an optional package list, but it needs to be installed to execute some features of **bayeso**.
- **requirements-dev.txt**: It is for developing the **bayeso** package.
- **requirements-examples.txt**: It needs to be installed to execute the examples included in the **bayeso** repository.

