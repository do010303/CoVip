from setuptools import setup, find_packages

package_name = 'assarm_deploy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='quan',
    maintainer_email='user@todo.todo',
    description='Raspberry Pi 4 deployment node, ONNX inference, UART bridge, and safety layer.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gazebo_player = assarm_deploy.gazebo_player:main',
        ],
    },
)
