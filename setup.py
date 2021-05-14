import pathlib
import setuptools

from nd_mlp_mixer.version import version

setuptools.setup(
    name="nd_mlp_mixer",
    version=version,
    author="Sidney Radcliffe",
    author_email="sidneyradcliffe@gmail.com",
    description="MLP-Mixer for TensorFlow.",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/sradc/nd_mlp_mixer",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    install_requires=["tensorflow", "einops"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
