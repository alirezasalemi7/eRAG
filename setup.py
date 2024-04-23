import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erag",
    version="0.0.1",
    author="Alireza Salemi",
    author_email="asalemi@cs.umass.edu",
    description="the implementation of the eRAG score.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alirezasalemi7/eRAG",
    packages=setuptools.find_packages(
        include=['erag*'],  # ['*'] by default
        exclude=['erag.tests', 'erag.eval']
    ),
    install_requires=['pytrec_eval == 0.5'],
)