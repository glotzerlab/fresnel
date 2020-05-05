# placeholder to take place of _common.so when building sphinx docs from source
print("**** You are importing fresnel from the source directory.")
print("**** This will not work, compile and import from the build directory")


class Material:
    def __init__(self, *args, **kwargs):
        pass


class RGBf:
    def __init__(self, *args, **kwargs):
        self.r = args[0]
        self.g = args[1]
        self.b = args[2]


class UserCamera:
    def __init__(self, *args, **kwargs):
        pass


class vec3f: # noqa
    def __init__(self, *args, **kwargs):
        pass


def cpu_built():
    return False


def gpu_built():
    return False

class CameraModel():
    orthographic=1
