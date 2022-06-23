from setuptools import setup, find_packages


setup(
    name='sumonet',
    version='0.1',
    license='Academic Free License ("AFL") v. 3.0',
    author="Berke Dilekoglu",
    author_email='dilekogluberke@gmail.com',
    packages=find_packages('sumonet'),
    package_dir={'': 'sumonet'},
    url='https://github.com/berkedilekoglu/SUMOnet',
    keywords='sumoylation machine-learning deep-learning',
    install_requires=[
        python,
        numpy,
        scikit-learn,
        joblib,
        pandas,
        epitopepredict,
        tensorflow,
        keras,
      ],

)