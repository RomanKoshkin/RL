import sys, pathlib

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
print(f'Python version: {sys.version_info[0]}')

ROOT = pathlib.Path(__file__).parent.parent.resolve()
print(f'ROOT: {ROOT}')
