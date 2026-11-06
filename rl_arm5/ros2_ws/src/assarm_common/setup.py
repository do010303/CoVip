from setuptools import setup, find_packages

setup(
    name='assarm_common',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy'],
    description='Shared logic for ASSARM robot (FK, obs, perception, grasp script). NumPy only.',
    python_requires='>=3.8',
)
