import fresnel

def test_cpu():
    if fresnel._cpu is not None:
        dev = fresnel.Device(mode='cpu')

def test_cpu_limit():
    if fresnel._cpu is not None:
        dev = fresnel.Device(mode='cpu', limit=2)

def test_gpu():
    if fresnel._gpu is not None:
        dev = fresnel.Device(mode='gpu')
