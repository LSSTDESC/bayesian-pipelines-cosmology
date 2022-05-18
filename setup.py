from setuptools import find_packages
from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='BPCosmo',
    description='Cosmology-level Bayesian Pipeline forward model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='LSST DESC',
    url='https://github.com/LSSTDESC/bayesian-pipelines-cosmology',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpyro', 'jax', 'lenstools', 'dm-haiku', 'jaxpm', 'jax-cosmo'],
    dependency_links=[
        'https://github.com/DifferentiableUniverseInitiative/JaxPM/tarball/master#egg=jaxpm-0.0.1'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    keywords='cosmology',
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)
