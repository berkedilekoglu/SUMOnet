from setuptools import setup, find_packages


setup(
    name='sumonet',
    version='0.0.3',
    license='Academic Free License ("AFL") v. 3.0',
    author="Berke Dilekoglu",
    author_email='dilekogluberke@gmail.com',
    packages=find_packages('sumonet'),
    package_dir={'': 'sumonet'},
    url='https://github.com/berkedilekoglu/SUMOnet',
    download_url = 'https://github.com/berkedilekoglu/SUMOnet/archive/refs/tags/v0.0.3tar.gz', 
    keywords='sumoylation machine-learning deep-learning',
    install_requires=[
        'numpy',
        'scikit-learn',
        'joblib',
        'pandas',
        'epitopepredict',
        'tensorflow',
        'keras',
      ],

)