#!/usr/bin/env python3
# Copyright (c) 2016-2022 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Generate GitHub Actions workflows from the templates."""

import jinja2
import yaml
import os

# device options needed to access the GPU devices on the runners
# because the nvidia container toolkit is built without cgroups
# support:
# https://aur.archlinux.org/packages/nvidia-container-toolkit
optix_docker_options = "--mount type=bind,source=/usr/lib/libnvidia-rtcore.so,"\
                       "target=/usr/lib/libnvidia-rtcore.so --mount type=bind,"\
                       "source=/usr/lib/libnvoptix.so,"\
                       "target=/usr/lib/libnvoptix.so " \
                       "--device /dev/nvidia0 " \
                       "--device /dev/nvidia1 " \
                       "--device /dev/nvidia-uvm " \
                       "--device /dev/nvidia-uvm-tools " \
                       "--device /dev/nvidiactl " \
                       "--gpus=all"

if __name__ == '__main__':
    # change to the directory of the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'),
                             block_start_string='<%',
                             block_end_string='%>',
                             variable_start_string='<<',
                             variable_end_string='>>',
                             comment_start_string='<#',
                             comment_end_string='#>',
                             trim_blocks=True,
                             lstrip_blocks=True)
    template = env.get_template('unit_test.yml')
    with open('templates/configurations.yml', 'r') as f:
        configurations = yaml.safe_load(f)

    # preprocess configurations and fill out additional fields needed by
    # `unit_test.yml` to configure the matrix jobs
    for name, configuration in configurations.items():
        for entry in configuration:
            if entry['config'].startswith('[optix'):
                entry['runner'] = "[self-hosted,GPU]"
                entry['docker_options'] = optix_docker_options
                entry['repository'] = "joaander"
            else:
                entry['runner'] = "ubuntu-latest"
                entry['docker_options'] = ""
                entry['repository'] = "glotzerlab"

    with open('unit_test.yml', 'w') as f:
        f.write(template.render(configurations))

    template = env.get_template('release.yml')
    with open('release.yml', 'w') as f:
        f.write(template.render())

    template = env.get_template('stale.yml')
    with open('stale.yml', 'w') as f:
        f.write(template.render())
