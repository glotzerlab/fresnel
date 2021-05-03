.. Copyright (c) 2016-2021 The Regents of the University of Michigan
.. This file is part of the Fresnel project, released under the BSD 3-Clause
.. License.

Code style
==========

All code in fresnel must follow a consistent style to ensure readability. We
provide configuration files for linters and autoformatters (specified below) so
that developers can automatically validate and format files.

These tools are configured for use with `pre-commit`_ in
``.pre-commit-config.yaml``. You can install pre-commit hooks to validate your
code. Checks will run on pull requests. Run checks manually with::

    pre-commit run --all-files

.. _pre-commit: https://pre-commit.com/

Python
------

Python code in GSD should follow `PEP8`_ with the formatting performed by
`yapf`_ (configuration in ``setup.cfg``). Code should pass all **flake8** tests
and formatted by **yapf**.

.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _yapf: https://github.com/google/yapf

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_

  * With these plugins:

    * `pep8-naming <https://github.com/PyCQA/pep8-naming>`_
    * `flake8-docstrings <https://gitlab.com/pycqa/flake8-docstrings>`_
    * `flake8-rst-docstrings <https://github.com/peterjc/flake8-rst-docstrings>`_

  * Configure flake8 in your editor to see violations on save.

* Autoformatter: `yapf <https://github.com/google/yapf>`_

  * Run: ``pre-commit run --all-files`` to apply style changes to the whole
    repository.

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``doc/``. Docstrings should follow `Google style`_
formatting for use in `Napoleon`_.

.. _Google Style: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
.. _Napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

C++/CUDA
--------

* Style is set by clang-format-10

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

  * If you have clang-format-10 installed, run:
    ``pre-commit run --all-files --hook-stage manual`` to apply changes to the
    whole repository.

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
