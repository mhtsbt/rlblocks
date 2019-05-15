import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlblocks",
    version="0.0.1",
    author="Matthias Hutsebaut",
    author_email="matthias.hutsebaut@gmail.co,",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mhtsbt/rlblocks",
    packages=["rlblocks.memory", "rlblocks.models", "rlblocks.utils"],
    install_requires=["gym", "torch", "opencv-python>=4.1.0.25,<5.0.0.0"]
)
