from setuptools import setup, find_packages

setup(
    name='pyEW',
    version='1.0.0',
    url='https://github.com/madamow/pyEW',
    license='MIT',
    author='Monika Adamow',
    author_email='madamow@icloud.com',
    description='Equivalent width calculator',
    entry_points = {
        'console_scripts': ['pyew = pyew.__main__:main']
        },
    packages=find_packages()
)
