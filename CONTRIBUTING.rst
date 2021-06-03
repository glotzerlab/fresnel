.. Copyright (c) 2016-2021 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Contributing
============

Contributions are welcomed via `pull requests on
GitHub <https://github.com/glotzerlab/fresnel/pulls>`__. Contact the **fresnel** developers before
starting work to ensure it meshes well with the planned development direction and standards set for
the project.

Features
--------

Implement functionality in a general and flexible fashion
_________________________________________________________

New features should be applicable to a variety of use-cases. The **fresnel** developers can assist
you in designing flexible interfaces.

Maintain performance of existing code paths
___________________________________________

Expensive code paths should only execute when requested.

Version control
---------------

Base your work off the correct branch
_____________________________________

All pull requests should be based off of ``master``.

Propose a minimal set of related changes
________________________________________

All changes in a pull request should be closely related. Multiple change sets that are loosely
coupled should be proposed in separate pull requests.

Agree to the contributor agreement
__________________________________

All contributors must agree to the Contributor Agreement before their pull request can be merged.

Source code
-----------

Use a consistent style
______________________

The **Code style** section of the documentation sets the style guidelines for **fresnel** code.

Document code with comments
___________________________

Use C++ comments for classes, functions, etc… Also comment complex sections of code so that other
developers can understand them.

Compile without warnings
________________________

Your changes should compile without warnings.

Tests
-----

Write unit tests
________________

Add unit tests for all new functionality.

Documentation
-------------

User documentation
__________________

Document public-facing API with Python docstrings in Google style.

Add developer to the credits
____________________________

Update the credits documentation to list the name and affiliation of each individual that has
contributed to the code.

Propose a change log entry
__________________________

Propose a short concise entry describing the change in the pull request description.
