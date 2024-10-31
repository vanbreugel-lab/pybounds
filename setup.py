import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pybounds", # Replace with your own username
    version="0.0.1",
    author="Ben Cellini, Burak Boyacioglu, Floris van Breugel",
    author_email="bcellini00@gmail.com",
    description="Bounding Observability for Uncertain Nonlinear Dynamics Systems (BOUNDS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vanbreugel-lab/pybounds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)