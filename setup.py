"""
Simple setup.py file.
"""

import io
from glob import glob
from os.path import basename, dirname, join, splitext, abspath

from setuptools import find_packages
from setuptools import setup


THIS_DIRECTORY = abspath(dirname(__file__))
with io.open(join(THIS_DIRECTORY, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='diffraction_utils',
    version='0.0.1',
    license='MIT',
    description='A package designed to simplify working with diffraction data.',
    author='Richard Brearton',
    author_email='richardbrearton@gmail.com',
    long_description=LONG_DESCRIPTION,
    long_decription_content_type='text/markdown',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    setup_requires=[
        'numpy', 'scipy', 'nexusformat'
    ],
    install_requires=[
        'numpy', 'scipy', 'nexusformat'
    ],
)
