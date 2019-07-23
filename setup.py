import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlblocks",
    version="0.0.1",
    author="Matthias Hutsebaut",
    author_email="matthias.hutsebaut@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mhtsbt/rlblocks",
    packages=["rlblocks.memory", "rlblocks.envs", "rlblocks.models", "rlblocks.policy", "rlblocks.trainers", "rlblocks.utils"],
    install_requires=["gym", "torch>=1.1.0,<2.0.0", "opencv-python>=4.1.0.25,<5.0.0.0", "interface"]
)
