from setuptools import setup, find_packages

setup(
    name='object-locator',
    version='1.1.0',
    description='Object Location using PyTorch (inference/testing only).',

    # The project's main homepage.
    url='https://viperlab.org',

    # Author details
    author='Javier Ribera, Yuhao Chen, and Edward Delp',
    author_email='ace@ecn.purdue.edu',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='~=3.6',
    # What does your project relate to?
    keywords='object localization purdue',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    
    package_data={'object-locator': ['models/*.ckpt']},

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['matplotlib', 'numpy',
                      'scikit-image', 'tqdm', 'argparse', 'parse',
                      'scikit-learn', 'pandas'],
)
