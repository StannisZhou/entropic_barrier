from setuptools import setup, find_packages

setup(
    name='diffusion_git_repo',
    author='Guangyao Zhou',
    author_email='guangyao_zhou@brown.edu',
    license='MIT',
    # Package info
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
)
