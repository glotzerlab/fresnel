---
name: Release checklist
about: '[for maintainer use]'
title: 'Release v0.XX.Y'
labels: ''
assignees: 'joaander'

---

Release checklist:

- [ ] Update actions versions.
  - See current actions usage with: `rg --no-filename --hidden uses: | awk '{$1=$1;print}' | sort | uniq `
  - Use global search and replace to update them to the latest tags
- [ ] Run *bumpversion*.
- [ ] Update change log.
  - ``git log --format=oneline --first-parent `git log -n 1 --pretty=format:%H -- CHANGELOG.rst`...``
  - [milestone](https://github.com/glotzerlab/fresnel/milestones)
- [ ] Check readthedocs build, especially change log formatting.
  - [Build status](https://readthedocs.org/projects/fresnel/builds/)
  - [Output](https://fresnel.readthedocs.io/en/latest/)
- [ ] Tag and push.
- [ ] Build tarball.
- [ ] Update conda-forge recipe.
