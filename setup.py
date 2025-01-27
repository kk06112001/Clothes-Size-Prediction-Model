from setuptools import setup,find_packages
setup(
    name="clothes-size-prediction-etl",  
    version="0.1",   
    author="Kunal",  
    description="An ETL pipeline to predict clothing sizes based on weight, age, and height",
    packages=find_packages(),  
   
    install_requires=[  
        "pandas",
        "numpy",
        "scikit-learn",
        "flask",
        "matplotlib",
        "seaborn",
    ],
    
)