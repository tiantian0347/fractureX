import sys
import os
import pathlib
from setuptools import setup, find_packages

# Add current directory to Python Path (if needed for local imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

__version__ = "1.1.0"  # Update this to your module's version

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

def load_requirements(path_dir=here, comment_char="#"):
    """Load requirements from requirements.txt, ignoring comments."""
    requirements = []
    requirements_path = os.path.join(path_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            # Filter out comments
            if comment_char in line:
                line = line[:line.index(comment_char)].strip()
            if line:  # if requirement is not empty
                requirements.append(line)
    return requirements

setup(
    name="fracturex",
    version=__version__,
    description="FractureX: A Python module for fracture analysis",  # Update description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tiantian0347/fracturex",  # Update URL
    author="Tian Tian",  # Update author
    author_email="tiantian0347@126.com",  # Update email
    license="GNU",  # Or your preferred license
    packages=find_packages(),
    install_requires=load_requirements(),
    zip_safe=False,
    extras_require={
        "doc": ["sphinx", "sphinx-rtd-theme"],  # Documentation dependencies
        "dev": ["pytest", "pytest-cov"],  # Development dependencies
        # Add any optional dependencies here
    },
    include_package_data=True,
    python_requires=">=3.8",  # Adjust Python version requirement as needed
)
