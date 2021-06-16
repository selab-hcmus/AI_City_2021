import platform
from setuptools import find_packages, setup, find_namespace_packages


setup(
    name="aic21",
    version="v0.0.1",
    description="NLP-Based Vehicle Track Retrieval",
    author="Tien-Phat Nguyen, Ba-Thinh Tran-Le",
    url="https://github.com/selab-hcmus/AI_City_2021",
    packages=find_packages(
                include=[
                    'detector', 'classifier', 'object_tracking', 
                    'retrieval_model','refinement', 'utils', 'relation_graph', 'srl',
                    # 'srl_extraction', 'srl_handler', 
                ],
                exclude=[
                    "results", "dataset", "assets", "visualize_tool", 
                    "*.data", "*.data.*", "data.*", 
                    "*.results", "*.results.*", "results.*",
                    # "dataset.train", "dataset.Track2", "dataset.validation", "dataset.test_boxes",
                    "*.notebooks", "*.notebooks.*", "*.test", 
                    "srl_handler.models", "srl_handler.models.*", 
                    "object_tracking.reid.torchreid","object_tracking.reid.torchreid.*",
                    "retrieval_model.config", "retrieval_model.config.*"
                ]
            ),
    include_package_data=False,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=["numpy", "Pillow", "opencv-python", "tqdm",],
)