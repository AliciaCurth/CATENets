import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='catenets',
    version='0.1.0',
    author="Alicia Curth",
    author_email="aliciacurth@gmail.com",
    description="Conditional Average Treatment Effect Estimation Using Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AliciaCurth/CATENets",
    packages=['catenets'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=['jax', 'scikit-learn', 'numpy', 'pandas']
)
