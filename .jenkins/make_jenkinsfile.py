import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template('Jenkinsfile.jinja')

tests = []

tests.append(dict(name='cuda9-optix51',
                  agent='gpu-short',
                  CC = 'gcc',
                  CXX = 'g++',
                  PYVER = '3.6',
                  ENABLE_OPTIX = 'ON',
                  ENABLE_EMBREE = 'OFF',
                  CONTAINER = 'ci-2019.06-cuda9-optix51.simg',
                  timeout=2))

tests.append(dict(name='cuda10-optix51',
                  agent='gpu-short',
                  CC = 'gcc',
                  CXX = 'g++',
                  PYVER = '3.6',
                  ENABLE_OPTIX = 'ON',
                  ENABLE_EMBREE = 'OFF',
                  CONTAINER = 'ci-2019.06-cuda10-optix51.simg',
                  timeout=2))

print(template.render(tests=tests))
