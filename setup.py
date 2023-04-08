from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'auto_tabnet',
    packages = ['auto_tabnet'],
    version = 'v0.0.1', 
    description = 'Automated implementation of Google TabNet.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Ersilia',
    author_email = 'hello@ersilia.io',
    url = 'https://github.com/ersilia-os/auto-tabnet',
    download_url = 'https://github.com/ersilia-os/auto-tabnet/archive/refs/tags/v1.0.0.tar.gz',
    keywords = ['automl', 'tabular'],
    classifiers = [],
)