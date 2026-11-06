from setuptools import setup, find_packages

setup(
    name='assarm_rl_env',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy'],
    description='Gymnasium environment for ASSARM RL training.',
)
