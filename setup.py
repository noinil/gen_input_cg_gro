from setuptools import setup, find_packages

setup(
    name='dna_3spn_gro',
    version='0.1',

    description='Generate 3SPN DNA model files for MD simulations in GRO format.',

    url='https://github.com/noinil/dna_3spn_gro',
    download_url = 'https://github.com/noinil/dna_3spn_gro/archive/v0.1.zip',

    author='Cheng Tan',
    author_email='ctan.info@gmail.com',

    license='GPL3',

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='molecular dynamics, simulation, DNA model',

    packages = ['dna_3spn_gro'],

    install_requires=['MDAnalysis'],

    entry_points={
    },
)
