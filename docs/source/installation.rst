Compilation and Installation
============================

.. caution::
     This is draft documentation.
.. note::

     INQ is under active development.

Compiling and running on a local computer
-----------------------------------------

Install dependencies and setup environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example in Ubuntu 22.04:::

    sudo apt install git cmake libblas-dev libboost-filesystem-dev libboost-serialization-dev liblapack-dev libopenmpi-dev pybind11-dev # or libmpich-dev

For example, in Fedora 37, systemwide:::

    sudo dnf install hdf5-devel lapack-devel ...

Clone repository
^^^^^^^^^^^^^^^^
::

    git clone --recursive git@gitlab.com:npneq/inq.git  # or https://gitlab.com/npneq/inq.git without account
    cd inq

For CPU-only system
^^^^^^^^^^^^^^^^^^^

::

    cmake .. --install-prefix=$HOME/.local -DCMAKE_BUILD_TYPE=Release
    make -j 12
    make install
    ctest -j 6 --output-on-failure

