from setuptools import setup, find_packages

setup(
    name="research_agent",
    version="0.1.0",
    package_dir={
        "research_agent": "src/research_agent"
    },
    packages=find_packages(where="src", include=["research_agent*"]),
    include_package_data=True,
)
