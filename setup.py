from setuptools import setup, find_packages


setup(
    name='sumonet',
    version='0.1.1',
    license='Academic Free License ("AFL") v. 3.0',
    author="Berke Dilekoglu",
    author_email='dilekogluberke@gmail.com',
    packages=find_packages(),
    package_data={
    # If any package or subpackage contains *.txt or *.rst files, include
    # them:
    "": ["*.fasta", "*.gz","*.h5"]
    
  },
    url='https://github.com/berkedilekoglu/SUMOnet',
    download_url = 'https://github.com/berkedilekoglu/SUMOnet/archive/refs/tags/v0.0.1tar.gz', 
    keywords='sumoylation machine-learning deep-learning',
    install_requires=[
        'numpy',
        'scikit-learn',
        'joblib',
        'pandas',
        'epitopepredict',
        'tensorflow',
        'keras',
        'requests',
        'biopython',
        '',

      ],

)