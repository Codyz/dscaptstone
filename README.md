### Modeling small-scale turbulence in Large Eddy Simulations using Deep Learning
      
#### Problem Statement:
Turbulence modeling remains one of the biggest challenges in engineering today, and has impacts ranging from aerodynamics to climate modeling. A plane cannot fly without jet propulsion and your car consumes the amount of fuel it does partly due to its aerodynamic properties. The image below shows fluid turbulence including several eddys (swirls).

<p align="center">
<img src="Images/Turbulence.png" style="display: block; margin: auto;" height="150" width="250" />

Given the difficulty and high cost of observing turbulence in a natural environment, most turbulence models include some form of simulation. Among the most popular methods are Direct Numerical Simulations (DNS), and Large-Eddy Simulations, first proposed in 1963 by Joseph Smagorinsky [1] and first explored in 1970 by Deardorff [2].

The governing equations of these types of simulations are the Navier-Stokes equations, which represent Newton's second law applied to the x, y and z-direction of Newtonian fluids in motion:

<p align="center">
<img src="Images/Navier_Stokes.png" style="display: block; margin: auto;" height="125" width="375" />

Both simulation-based approaches mentioned above use these equations as follows:

* Direct Numerical Simulations (DNS) analytically solve the Navier-Stokes equation across all temporal and spatial scales. The main issue arises because current computing power is not sufficient to conduct a DNS on large scales.
* Large Eddy Simulations (LES) are more computationally efficient, by attempting to solve the Navier-Stokes equations on large scales only and subsequently  model the turbulence behavior on smaller scales. This is achieved via low-passed spacial filtering (i.e. a convolution kernel <bdi>G_<sub>&Delta</sub>(x)</bdi> of size <bdi>&Delta</bdi> is applied), a method used to separate large (resolved scale) and small (sub-grid scale) eddys. Applying the kernel will remove sub-grid scale information and transforms the Navier-Stokes equations by adding a stress term <bdi>&tau<sup>&Delta</sup></bdi> that needs to be modeled. 

#### How to navigate this repository:

      │
      ├── Archive
      |    |-- Contains outdated, experimental notebooks
      |
      ├── EDA
      │    |-- Contains two notebooks on eploratory data visualization of our input and output datasets
      |
      ├── Saved Models
      |    |-- Multi Output
      |    |    |-- Contains multi-output models, which predict all sheer stresses and heat flux simultaneously
      |    |-- Sinle Output
      |         |-- Base
      |         |    |-- Contains sinlge-output models for sheer stress and heat flux
      |         |-- Combined
      |         |    |-- Contains mutli-input single-output models with differing initial conditions
      |         |-- Combined_Time_Dependencies
      |              |-- Contains mutli-input single-output models with differing initial conditions and time steps
      |
      ├── Code
      |    |-- CNN
      |    |    |-- Contains experimental notebooks with Convolutional Neural Networks
      |    |-- LSTM
      |    |    |-- Contains experimental notebook with single-output LSTM for sheer stress
      |    |-- DNN
      |         |-- Final Models
      |         |    |-- Multi Output
      |         |         |-- Contains multi-output DNN models, which predict all sheer stresses and heat flux
      |         |         |-- Inputs: base input, differing conditions and differing conditions and time steps
      |         |    |-- Single Output
      |         |         |-- Contains single-output DNN models for heat flux and sheer stresses
      |         |         |-- Inputs: base input, differing conditions and differing conditions and time steps
      |         |-- Hyperparameter Checks
      |         |    |-- Includes checks for box size, input data, loss function and data scaling technique
      |         |-- Capstone_DNN_Correlation_by_layer.ipynb
      |         |    |-- Explores the true vs. pred. correlation evolution for each layer along the z-axis
      |         |-- Capstone_DNN_TauPredictions_by_layer.ipynb
      |         |    |-- Explores true vs. pred. distributions for several layers along the z-axis
      |         |-- Predictions
      |              |-- Multi Output
      |              |    |-- Contains predictions of multi-output models for heat flux and sheer stress
      |              |    |-- Inputs: base input, differing conditions and differing conditions and time steps
      |              |-- Single Output
      |              |    |-- Taus
      |              |         |-- Contains predictions of single-output models for sheer stresses
      |              |         |-- Inputs: base input, differing conditions and differing conditions and time steps
      |              |    |-- Heat
      |              |         |-- Contains predictions of single-output models for heat flux
      |              |         |-- Inputs: base input, differing conditions and differing conditions and time steps
--------------------
[1] J. Smagorinsky. General circulation experiments with the primitive equation. i. the basic experiment. Monthly Weather Review, 91, 1963.<br/>
[2] J. B. Deardorff. A numerical study of three-dimensional turbulent channel flow at large reynolds numbers. J. Fluid Mech., 41, 05 1970.
