from setuptools import setup, find_packages

package_name = 'assarm_rl_train'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name] if False else []),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'stable-baselines3', 'gymnasium', 'numpy'],
    zip_safe=True,
    maintainer='quan',
    maintainer_email='user@todo.todo',
    description='Training pipeline and callbacks for ASSARM RL (SAC).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
