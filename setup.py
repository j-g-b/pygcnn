from setuptools import setup
from setuptools import find_packages

setup(name='pygcnn',
      version='1.0',
      description='Convolutions on Graphs',
      author='Jordan Bryan',
      author_email='jbryan@broadinstitute.org',
      url='tbd',
      download_url='tbd',
      license='none',
      install_requires=['numpy',
                        'tensorflow',
                        'networkx',
                        'scipy',
                        'pandas',
                        'progressbar'
                        ],
      package_data={'pygcnn': ['README.md']},
      packages=find_packages())