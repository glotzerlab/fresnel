import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template('Jenkinsfile.jinja')

tests = []

tests.append(dict(name='cuda9-optix50',
                  agent='gpu',
                  CC = 'gcc',
                  CXX = 'g++',
                  PYVER = '3.6',
                  PYTEST = 'py.test-3',
                  CMAKE = 'cmake',
                  ENABLE_OPTIX = 'ON',
                  ENABLE_EMBREE = 'OFF',
                  CONTAINER = 'ci-optix-2018.06.simg',
                  timeout=2))

print(template.render(tests=tests))
