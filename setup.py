import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_tabnet",
    version = "0.1",
    author="Ersilia",
    author_email="hello@ersilia.io",
    description="Auto Tabnet",
    keywords="automl, hyperparameter optimisation, tabular",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ersilia-os/auto-tabnet",
    project_urls={
        "Documentation": "",
        "Source Code": "https://github.com/ersilia-os/auto-tabnet",
    },
    package_dir={"": "auto_tabnet"},
    packages=setuptools.find_packages(where="auto_tabnet"),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.7",
)