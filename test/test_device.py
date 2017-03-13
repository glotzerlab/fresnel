import fresnel

def test_cpu():
    dev = fresnel.Device(mode='cpu')

def test_cpu_limit():
    dev = fresnel.Device(mode='cpu', limit=2)

def test_gpu():
    if fresnel._gpu is not None:
        dev = fresnel.Device(mode='gpu')
