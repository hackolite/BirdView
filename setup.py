from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="birdview",
    version="0.1.0",
    description="Pipeline BEV auto-calibré : vue caméra → heatmap / tracking vue de haut",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="hackolite",
    url="https://github.com/hackolite/BirdView",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "birdview=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)
