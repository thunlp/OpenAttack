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
    name="OpenAttack",  # Replace with your own username
    version="1.1.1",
    author="THUNLP",
    author_email="thunlp@gmail.com",
    description="OpenAttack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thunlp/OpenAttack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements
)
