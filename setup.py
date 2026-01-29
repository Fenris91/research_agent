from pathlib import Path
from setuptools import setup, find_packages


def _parse_requirements(path: str) -> list[str]:
    req_path = Path(path)
    if not req_path.exists():
        return []
    lines = req_path.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


setup(
    name="research_agent",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["research_agent*"]),
    include_package_data=True,
    install_requires=_parse_requirements("requirements-runtime.txt"),
)
