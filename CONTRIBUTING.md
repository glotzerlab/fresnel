Contributions are welcomed via [pull requests on GitHub](https://github.com/glotzerlab/fresnel/pulls). Contact
the **fresnel** developers before starting work to ensure it meshes well with the planned development direction and
standards set for the project.

# Features

## Implement functionality in a general and flexible fashion

New features should be applicable to a variety of use-cases. The **fresnel** developers can assist you in designing
flexible interfaces.

## Maintain performance of existing code paths

Expensive code paths should only execute when requested.

# Version control

## Base your work off the correct branch

All pull requests should be based off of `master`.

## Propose a minimal set of related changes

All changes in a pull request should be closely related. Multiple change sets that
are loosely coupled should be proposed in separate pull requests.

## Agree to the contributor agreement

All contributors must agree to the Contributor Agreement ([ContributorAgreement.md](ContributorAgreement.md)) before
their pull request can be merged.

# Source code

## Use a consistent style

[SourceConventions.md](SourceConventions.md) defines the style guidelines for **fresnel** code.

## Document code with comments

Use C++ comments for classes, functions, etc... Also comment complex sections of code so that other
developers can understand them.

## Compile without warnings

Your changes should compile without warnings.

# Tests

## Write unit tests

Add unit tests for all new functionality.

# Documentation

## User documentation

Document public facing API with python docstrings in the napoleon format.

## Example notebooks

Add demonstrations of new functionality to [fresnel-examples](https://github.com/glotzerlab/fresnel-examples).

## Add developer to the credits

Update the credits documentation to reference what each developer contributed to the code.

## Propose a change log entry

Propose a short concise entry describing the change in the pull request description.
