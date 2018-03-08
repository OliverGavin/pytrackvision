from setuptools import setup

setup(name='pytrackvision',
      version='0.1',
      description='A Python library for face and hand tracking and gesture recognition.',
      url='http://github.com/OliverGavin/pytrackvision',
      author='Oliver Gavin',
      author_email='oliver@gavin.ie',
      license='MIT',
      packages=['pytrackvision'],
      test_suite='nose2.collector.collector',
      tests_require=['nose2'],
      zip_safe=False)
