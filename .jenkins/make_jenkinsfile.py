import jinja2

env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
template = env.get_template('Jenkinsfile.jinja')

tests = []

tests.append(dict(name='gcc6-py36-cd90-optx50',
                  agent='gpu',
                  CC = '/usr/sbin/gcc-6',
                  CXX = '/usr/sbin/g++-6',
                  PYVER = '3.6',
                  PYTEST = '/usr/bin/pytest',
                  CMAKE_BIN = '/usr/sbin',
                  ENABLE_OPTIX = 'ON',
                  CONTAINER = 'ci-optix-2018.02.simg',
                  timeout=2))

print(template.render(tests=tests))
