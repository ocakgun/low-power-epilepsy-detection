low-power-epilepsy-detection
==============================

Project which uses epilepsy data to create a ANN which detects seizures and ports this to a low level integration 

## Current status
Updated: 16-03-2020
 - Currently working to make this completely work on Google Colab, for this the main problem is that the complete dataset is to big, solution is to download the dataset in multiple stages and process raw data to (much smaller) processed data which can be used as input for the ANN 
 - Some profiling code has to be ported from an earlier version of this project, used to find correlation ed, will be usefull in a later stage to compare performance 
 - Also some work has been done in audioprocessing ANNs with detecting if an audio sample was music or not (was an easy extension from a training exercise and is quite comparable to epilepsy detection in some sense). After this runs well on Colab I will port that code to the model folder
 - Reason why this runs on Github is due to the Google Colab integration, other solution may be possible 

## How to Run this project

 (Currently broken, but steps should be:)
 - Import main ipynb (python notbebook) in Google Colab via Git
 - Run command to copy  (``!git clone https://github.com/jmsvanrijn/low-power-epilepsy-detection.git``) to project
 - Run command to download environment (``!make create_environment``)
 - Run command to download data and process to model input (``!make data``) -> Currently not possible due to the amount of data the raw dataset it, working on a solution as mentioned in "Current-status"
 - Run command to train ANN (``!make model``)

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py



--------
