import setuptools

VERSION = "test"
with open("OpenAttack/version.py", "r") as fver:
    VERSION = fver.read().replace("VERSION", "").replace("=", "").replace("\"", "").strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = []
with  open("requirements.txt") as freq:
    for line in freq.readlines():
        requirements.append( line.strip() )

setuptools.setup(
    name="OpenAttack",  # Replace with your own username
    version=VERSION,
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
