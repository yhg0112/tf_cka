"""Install tf_cka."""

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tf_cka',
    version='0.1.0',
    description='Centered Kerenel Alignment implementation with tensorflowo 2.0',
    long_description=long_description,
    author='hyeongu yun',
    author_email='youaredead@snu.ac.kr',
    url='https://github.com/yhg0112/tf_cka',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.2.0'],
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning',
)

