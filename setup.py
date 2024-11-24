from setuptools import find_packages, setup

setup(
    name="docker_kb",
    version="0.1",
    description="A knowledgebase for docker basics",
    author="Simão Gonçalves",
    author_email="simao.campos.goncalves@gmail.com",
    packages=find_packages(),
    include_package_data=True,
)