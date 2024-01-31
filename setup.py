from setuptools import find_packages, setup

version = "0.0.1"


def read_requirements(path):
    with open(path, "r") as f:
        reqs = f.read().splitlines()
    return [req for req in reqs if req and not req.startswith('#')]


requirements = read_requirements("requirements.txt")

# Read your long description from the README.md file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="three_gen_subnet",
    version=version,
    description="Bittensensor subnet 32",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/404-Repo/three-gen-subnet",
    author="Maxim Bladyko",
    author_email="max@404.xyz",
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
