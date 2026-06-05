from setuptools import setup, find_packages

setup(
    name='adaptive_trilemma',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy','scipy','pandas','matplotlib','cvxpy','deap','torch','requests','stable-baselines3','gym','osqp'
    ],
    entry_points={
        'console_scripts': [
            'run-sim=main:main'
        ],
    },
)
