### Modeling small-scale turbulence in Large Eddy Simulations using Deep Learning

## How to navigate this repositroy:

------------
    │
    ├── Archive
    |    |-- Contains outdated, experimental notebooks
    |
    ├── EDA
    │    |-- Contains two notebooks on eploratory data visualization of our input and output datasets
    |
    ├── Saved Models
    |    |-- Multi Output
    |         |-- Contains model weights and architectures for our multi-output models, which predict all &tau and h simultaneously   
    |
    ├──  model_training_final.ipynb
    |    |--This notebook contains code to train the model that we will finally use. It also preprocesses images.
    |
    ├── lane_keep_testing_final.ipynb
    |    |--This notebook uses a pretrained model( .h5 and .json) files to make online predictions on the GoPiGo.
    |
    ├── final_trained_model.h5, final_trained_model.json
    |    |--Trained Models
--------

