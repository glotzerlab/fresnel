.. Copyright (c) 2016-2020 The Regents of the University of Michigan
.. This file is part of the Fresnel project, released under the BSD 3-Clause
.. License.

Code style
==========

All code in fresnel must follow a consistent style to ensure readability.
We provide configuration files for linters (specified below) so that developers
can automatically validate and format files.

Python
------

Python code in GSD should follow `PEP8
<https://www.python.org/dev/peps/pep-0008>`_ with the formatting performed by
`yapf <https://github.com/google/yapf>`_ (configuration in ``setup.cfg``).
Code should pass all **flake8** tests and formatted by **yapf**.

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_

  * With these plugins:

    * `pep8-naming <https://github.com/PyCQA/pep8-naming>`_
    * `flake8-docstrings <https://gitlab.com/pycqa/flake8-docstrings>`_
    * `flake8-rst-docstrings <https://github.com/peterjc/flake8-rst-docstrings>`_

  * Run: ``flake8`` to see a list of linter violations.

* Autoformatter: `yapf <https://github.com/google/yapf>`_

  * Run: ``yapf -d -r .`` to see needed style changes.
  * Run: ``yapf -i file.py`` to apply style changes to a whole file, or use
    your IDE to apply **yapf** to a selection.

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``doc/``. Docstrings should follow `Google style
<https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_
formatting for use in `Napoleon
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

C++/CUDA
--------

* Style is set by clang-format >= 10

  * Whitesmith's indentation style.
  * 100 character line width.
  * Indent only with spaces.
  * 4 spaces per indent level.
  * See :file:`.clang-format` for the full **clang-format** configuration.

* Naming conventions:

    * Namespaces: All lowercase ``somenamespace``
    * Class names: ``UpperCamelCase``
    * Methods: ``lowerCamelCase``
    * Member variables: ``m_`` prefix followed by lowercase with words
      separated by underscores ``m_member_variable``
    * Constants: all upper-case with words separated by underscores
      ``SOME_CONSTANT``
    * Functions: ``lowerCamelCase``

Tools
^^^^^

* Autoformatter: `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.

  * Run: ``./run-clang-format.py -r .`` to see needed changes.
  * Run: ``clang-format -i file.c`` to apply the changes.

Documentation
^^^^^^^^^^^^^

Documentation comments should be in Javadoc format and precede the item they
document for compatibility with Doxygen and most source code editors. Multi-line
documentation comment blocks start with ``/**`` and single line ones start with
``///``.

Other file types
----------------

Use your best judgment and follow existing patterns when styling CMake and other
files types. The following general guidelines apply:

* 100 character line width.
* 4 spaces per indent level.
* 4 space indent.

Editor configuration
--------------------

`Visual Studio Code <https://code.visualstudio.com/>`_ users: Open the provided
workspace file (``fresnel.code-workspace``) which provides configuration
settings for these style guidelines.
