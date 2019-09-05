from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='orbittools',
      version='0.2',
      description='basic python tools for working with orbital dynamics',
      url='https://github.com/logan-pearce/orbittools',
      author='Logan Pearce',
      author_email='loganpearce55@gmail.com',
      license='',
      packages=['orbittools'],
      install_requires=['numpy'],
      #dependency_links=['https://github.com/logan-pearce/myastrotools/tarball/master#egg=package-1.0'],
      #package_data={'': ['myastrotools/table_u0_g_col.txt','myastrotools/table_u0_g.txt']},
      #include_package_data=True,
      zip_safe=False)
