Units
=====

here are several units systems that are used at the atomic level, and many times they are mixed.
For users this can be cumbersome, confusing, and many times leads to errors.
To avoid this problem, in inq all the quantities must explicitly include the unit they are in.
With C++ this can be done in a simple and elegant way.

For example, to define an energy magnitude we could do:::

    auto hydrogen_energy = 13.6_electronvolt;

Units supported by inq
----------------------

**Length**

* Bohr: ``_b, _bohr``
* Angstrom: ``_A, _angstrom``
* Metric: ``_nm, _nanometer, _pm, _picometer``

**Energy**

* Hartree: ``_ha, _Ha, _hartree``
* Rydberg: ``_ry, _Ry, _rydberg``
* Electronvolt: ``_ev, _eV, _electronvolt``
* Kelvin: ``_K, _kelvin``

**Time**

* Atomic: ``_atomictime``
* Metric: ``_as, _attosecond, _fs, _femtosecond, _ps, _picosecond, _ns, _nanosecond``
* From energy: a scalar divided by an energy magnitude results in a magnitude of time.
