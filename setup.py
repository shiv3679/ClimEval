from setuptools import setup, find_packages

setup(
    name="ClimEval",  # PyPI package name
    use_scm_version=True,  # Auto-manages versions via Git tags
    setup_requires=["setuptools_scm"],  # Ensures setuptools_scm is used

    author="Shiv Shankar Singh",
    author_email="shivshankarsingh.py@gmail.com",
    description="Climate Model vs. Observation Verification Tool",
    
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    
    url="https://github.com/shiv3679/ClimEval",
    packages=find_packages(include=["climeval", "climeval.*"]),

    install_requires=[
        "numpy",
        "xarray",
        "dask",
        "scipy",
        "matplotlib",
        "cartopy",
        "pytest"
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    python_requires=">=3.7",

    entry_points={
        "console_scripts": [
            "climeval=climeval.cli:main"  # Optional CLI support
        ]
    },

    include_package_data=True,  # Ensures non-Python files (like README.md) are included
    zip_safe=False,  # Ensures compatibility with different environments
)
