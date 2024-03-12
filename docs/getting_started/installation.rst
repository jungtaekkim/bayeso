Installing BayesO
#################

We recommend installing it with **virtualenv**.
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
To install **bayeso** from source code, command it in the **bayeso** root.

.. code-block:: console

    $ pip install .

Compiling from Source (Editable)
================================

It is for editable development mode.
To use editable development mode, command it in the **bayeso** root.

.. code-block:: console

    $ pip install -e .

If you want to install the packages required for optional features, development, and examples,
you can simply add **[optional]**, **[dev]**, and **[examples]**.
For example, you can command it for installing the packages required for development.

.. code-block:: console
    $ pip install .[dev]

or

.. code-block:: console
    $ pip install -e .[dev]

Uninstalling
============

If you would like to uninstall **bayeso**, command it.

.. code-block:: console

    $ pip uninstall bayeso
