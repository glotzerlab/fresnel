import fresnel

def test_cpu():
    dev = fresnel.Device(mode='cpu')

def test_cpu_limit():
    dev = fresnel.Device(mode='cpu', limit=2)

def test_gpu():
    dev = fresnel.Device(mode='gpu')
