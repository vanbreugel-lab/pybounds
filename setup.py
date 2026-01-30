import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pybounds",
    version="0.1.0",
    author="Ben Cellini, Burak Boyacioglu, Floris van Breugel",
    author_email="bcellini00@gmail.com",
    description="Bounding Observability for Uncertain Nonlinear Dynamics Systems (BOUNDS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/pybounds/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
