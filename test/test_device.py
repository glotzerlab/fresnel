import fresnel

def test_cpu():
    if 'cpu' in fresnel.Device.available_modes:
        dev = fresnel.Device(mode='cpu')

def test_cpu_limit():
    if 'cpu' in fresnel.Device.available_modes:
        dev = fresnel.Device(mode='cpu', limit=2)

def test_gpu():
    if 'gpu' in fresnel.Device.available_modes:
        dev = fresnel.Device(mode='gpu')
