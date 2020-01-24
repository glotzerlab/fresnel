import fresnel


def test_cpu():
    if 'cpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='cpu')


def test_cpu_limit():
    if 'cpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='cpu', n=2)


def test_gpu():
    if 'gpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='gpu')


def test_gpu_limit():
    if 'gpu' in fresnel.Device.available_modes:
        fresnel.Device(mode='gpu', n=1)
