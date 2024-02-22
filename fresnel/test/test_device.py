# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the Device class."""

import fresnel


def test_cpu():
    """Test the CPU device."""
    if 'cpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='cpu')


def test_cpu_limit():
    """Test the cpu n argument."""
    if 'cpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='cpu', n=2)


def test_gpu():
    """Test the GPU device."""
    if 'gpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='gpu')


def test_gpu_limit():
    """Test the GPU n argument."""
    if 'gpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='gpu', n=1)
