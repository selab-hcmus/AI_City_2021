import platform
from setuptools import find_packages, setup, find_namespace_packages

setup(
    name="AI_City_Challenge_2021",
    version="v0.0.1",
    description="NLP-Based Vehicle Track Retrieval",
    author="Tien-Phat Nguyen, Ba-Thinh Tran-Le",
    url="https://github.com/selab-hcmus/AI_City_2021",
    packages=find_namespace_packages(
<<<<<<< HEAD
        exclude=["assets", "dataset", "visualize_tool", "*/results", "object_tracking/results_exp"]
=======
        exclude=["assets", "dataset", "visualize_tool", "object_tracking/resuls", "object_tracking/resuls_exp"]
>>>>>>> 3d6bf6e0a9ca6cb1905d8b01ee44702456207b18
    ),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=["numpy", "Pillow", "opencv-python", "tqdm", "yacs",],
)