Introduction
============

.. caution::
     This is draft documentation.
.. note::

     INQ is under active development.

INQ is based on density functional theory (DFT).
It can calculate ground state properties in DFT, and also excited states using time-dependent DFT (TDDFT) in real-time and linear response.
It implements different approximations for the exchange and correlation part: semi-local functionals like LDAs, GGAs, and metaGGAs, and hybrid functionals that are implemented through the ACE approach for fast execution.

One key feature of INQ is that is designed from the ground up to work in modern high-performance computing platforms.
It has support for GPU parallelization (through different frameworks), thread parallelization, and distributed memory parallelization using MPI (and Nvidia NCCL).
The thread parallelization is designed so that different tasks within the DFT approach are performed simultaneously, this achieves better scalability with respect to data parallelization alone.
INQ can perform calculations with unpolarized, polarized, and non-colinear spin.

INQ attempts to be as agnostic as possible with the representation used for the states and other quantities that appear in DFT.
It uses plane waves by default, but other representations like real space will be available.

To keep the code simple, INQ is designed as modular as possible, it uses libraries for most of the operations that can be done independently.
Some libraries, like pseudopod (for pseudopotential parsing) or multi (for multidimensional arrays) are developed by INQ authors.
Others are written by third parties, like libxc (for exchange-correlation functionals).
In the future, some other parts of INQ might be split into an independent library.
