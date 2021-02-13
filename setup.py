from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='heart_health',
    packages=find_packages(),
    version='0.1.0',
    description='Pipeline for heart disease classification',
    author='Maxim Blumental',
    license='MIT',
    install_requires=required,
    entry_points={
        'console_scripts': ['train_pipeline=heart_health.train_pipeline:train_pipeline_command'],
    },
)
