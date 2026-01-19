.. Copyright (c) 2016-2026 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Code style
==========

All code in fresnel must follow a consistent style to ensure readability. We
provide configuration files for linters and autoformatters (specified below) so
that developers can automatically validate and format files.

These tools are configured for use with `prek`_ and checks will run on pull requests.
Run checks manually with::

    prek run --all-files

.. _prek: https://prek.j178.dev/

Python
------

Python code in GSD should follow `PEP8`_ with the formatting performed by
`ruff`_ (configuration in ``ruff.toml``). Code should pass all **ruff** linter checks.

.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _ruff: https://github.com/astral-sh/ruff

Tools
^^^^^

* Linter: `ruff`_

  * With these extensions:

    * `D <https://docs.astral.sh/ruff/rules/#pydocstyle-d>`_
    * `E, W <https://docs.astral.sh/ruff/rules/#pycodestyle-e-w>`_
    * `F <https://docs.astral.sh/ruff/rules/#pyflakes-f>`_
    * `N <https://docs.astral.sh/ruff/rules/#pep8-naming-n>`_
    * `NPY <https://docs.astral.sh/ruff/rules/#numpy-specific-rules-npy>`_
    * `RUF200 <https://docs.astral.sh/ruff/rules/invalid-pyproject-toml>`_

  * The ``ruff.toml`` included with the package automatically configures these rules.

* Autoformatter: `ruff <https://github.com/astral-sh/ruff>`_

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``doc/``. Docstrings should follow `Google style`_
formatting for use in `Napoleon`_.

.. _Google Style: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
.. _Napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

C++/CUDA
--------

* Style is set by clang-format

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
