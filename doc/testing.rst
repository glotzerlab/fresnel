Testing
=======

All code in **fresnel** must be tested to ensure that it operates correctly.

**Unit tests** check that basic functionality works, one class at a time. Unit tests assume internal
knowledge about how classes work and may use unpublished APIs to stress test all possible input and
outputs of a given class in order to exercise all code paths.

Running tests
-------------

Execute the following commands to run the tests:

* ``python3 -m pytest fresnel``

When you run pytest_ outside of the build directory, it will the test ``fresnel`` package that
Python imports (which may not be the version just built).

.. seealso::

    See the pytest_ documentation for information on how to control output, select specific tests,
    and more.

.. _pytest: https://docs.pytest.org/

Implementing tests
------------------

Add ``test_*.py`` files that use pytest_ to test new functionality, following the patterns in the
existing tests. **fresnel** produces images as output. Many of the tests provide a reference image
and compare the output image to the reference. Reference images should be relatively low resolution
and path tracing should use only a moderate number of samples to reduce the time needed to run
tests.

.. note::

    Add any new ``test_*.py`` files to the list in the corresponding ``CMakeLists.txt`` file.
