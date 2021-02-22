from setuptools import setup, find_packages

subfolder = "src/python"
setup(name='caremit',
      version='0.1.0',
      description='Python package for factoring out helpers which we reused in our caremit project.',
      author='Stephan Sahm, Moriz Münst, Marcel Blistein',
      author_email='stephan.sahm@gmx.de, moses-github@geekbox.com, marcel.blistein@gmail.com',
      packages=find_packages(subfolder),
      package_dir={"": subfolder},
      install_requires=[
            'Flask>=1,<2',
            'flask-session~=0.3.2',
            'ipykernel',
            'jedi==0.17.0',
            'keras',
            'msal>=1.7,<2',
            'numpy',
            'pandas',
            'requests>=2,<3',
            'scipy',
            'sklearn',
            'tensorflow',
            'toolz',
            'werkzeug>=1,<2',
            'wfdb',
      ])