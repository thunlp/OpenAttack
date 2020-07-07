import setuptools

"""
Guide: https://setuptools.readthedocs.io/en/latest/setuptools.html
"""

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
with  open("requirements.txt") as freq:
    for line in freq.readlines():
        requirements.append( line.strip() )

setuptools.setup(
    name="TAADToolbox",  # Replace with your own username
    version="0.0.1",
    author="THUNLP",
    author_email="thunlp@gmail.com",
    description="TAADToolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fanchao-Qi/TAADToolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements
)
