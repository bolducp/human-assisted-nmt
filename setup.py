import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="human-assisted-nmt-pkg-BOLDUCP",
    version="0.0.1",
    author="Paige Finkelstein",
    author_email="paigelfink@gmail.com",
    description="A package for human-assisted NMT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bolducp/human-assisted-nmt",
    packages=['hnmt'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)