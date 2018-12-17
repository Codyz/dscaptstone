### Modeling small-scale turbulence in Large Eddy Simulations using Deep Learning

## How to navigate this repositroy:


      │
      ├── Archive
      |    |-- Contains outdated, experimental notebooks
      |
      ├── EDA
      │    |-- Contains two notebooks on eploratory data visualization of our input and output datasets
      |
      ├── Saved Models
      |    |-- Multi Output
      |         |-- Contains multi-output models, which predict all sheer stresses and heat flux simultaneously
      |    |-- Sinle Output
      |         |-- Base
      |              |-- Contains sinlge-output models for sheer stress and heat flux
      |         |-- Combined
      |              |-- Contains mutli-input single-output models with differing initial conditions
      |         |-- Combined_Time_Dependencies
      |              |-- Contains mutli-input single-output models with differing initial conditions and time steps
      |
      ├──  model_training_final.ipynb
      |    |--This notebook contains code to train the model that we will finally use. It also preprocesses images.
      |
      ├── lane_keep_testing_final.ipynb
      |    |--This notebook uses a pretrained model( .h5 and .json) files to make online predictions on the GoPiGo.
      |
      ├── final_trained_model.h5, final_trained_model.json
      |    |--Trained Models



