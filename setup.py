from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='sumonet',
    version='0.1.5',
    license='Academic Free License ("AFL") v. 3.0',
    description='Package Description',
    long_description=long_description,  # Use contents of README.md
    long_description_content_type='text/markdown',  # Specify content type
    author="Berke Dilekoglu",
    author_email='berkedilekoglu@sabanciuniv.edu',
    packages=find_packages(),
    package_data={
    # If any package or subpackage contains *.fasta or *.gz or *.h5 files, include
    # them:
    "": ["*.fasta", "*.gz","*.h5","*.md"]
    
  },
    url='https://github.com/berkedilekoglu/SUMOnet',
    download_url = 'https://github.com/berkedilekoglu/SUMOnet/archive/refs/tags/v0.1.5tar.gz', 
    keywords='sumoylation machine-learning deep-learning bioinformatics',
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
        'loguru',

      ],

)