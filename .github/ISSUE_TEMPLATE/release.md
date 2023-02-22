---
name: Release checklist
about: '[for maintainer use]'
title: 'Release v0.XX.Y'
labels: ''
assignees: 'joaander'

---

Release checklist:

- [ ] Run *bumpversion*.
- [ ] Update change log.
  - ``git log --format=oneline --first-parent `git log -n 1 --pretty=format:%H -- CHANGELOG.rst`...``
  - [milestone](https://github.com/glotzerlab/fresnel/milestones)
- [ ] Check readthedocs build, especially change log formatting.
  - [Build status](https://readthedocs.org/projects/fresnel/builds/)
  - [Output](https://fresnel.readthedocs.io/en/latest/)
- [ ] Tag and push.
- [ ] Update conda-forge recipe.
