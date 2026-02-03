# Gaussian processes and spatial-temporal autoregressive models, PCA, pPCA in Julia/Turing


## (Probabilistic, Latent) Principle Components Analysis Bayesian Householder Rotation invariant form.

Copied and adapted from numerous sources. All credit goes to these authors. Notably useful sources include:

[Rajbir S. Nirwan and Nils Bertschinger, 2019. Proceedings of the 36 th International Conference on Machine Learning, Long Beach, California, PMLR 97](https://github.com/jae0/HouseholderBPCA/blob/master/py_stan_code/ppca_house_improved.stan)
 
And the [main Julia/Turing pPCA tutorial](https://turing.ml/v0.21/tutorials/11-probabilistic-pca).

What is here are only minor changes and adaptation for my own needs ([species composition in marine environments and feeding networks](https://github.com/jae0/aegis)).

Example work flow: see [./docs/pca.md](./docs/pca.md) 

Or to use directly from R (you will need to install julia and required libs on your own): [./example_call_from_R.md](./example_call_from_R.md)


## Conditional AutoRegressive ("CAR") models 

CAR models are just about the simplest possible way of accounting for spatial autorcorrelation. It is essentially the analogue of the temporal ARIMA type models but in space. It is arguably even simpler as the units of space are discrete areal units (arbitrary polygons). Originally it was formulated in lattice form from the Ising models in physics. Its use is most prevalent in epidemiology made accessible by the original Besag-York-Mollie paper, Leroux and Riebler, and many others.

For the R implementation (front-end to INLA): see https://github.com/jae0/carstm .

- [https://github.com/jae0/carstm/inst/scripts/example_temperature_carstm.md]( https://github.com/jae0/carstm/inst/scripts/example_temperature_carstm.md)




## Conditional AutoRegressive Space-Time Models (in Julia) and other approaches: 

- CAR/ICAR/BYM areal unit models (space)
- GP models in time or space or covariates ("features")
- Temporal (dynamical) models  
- Spacetime models that combine all of the above using Julia/Turing

Most didactic are the examples that use STAN (see info copied from various sources in "docs/example_car_stan.md"). 

The INLA/R, BRMS/R, Turing/Julia (subdirectory examples) implementations are confirming parameter estimates.

Ultimately, we use Julia/Turing due to simplicity and speed and attempt to develop a useful statistical and dynamical model of snow crab spatiotemporal processes.

It is similar in scope to the github.com/jae0/carstm, however, with no reliance upon INLA for computation. 

See examples and progression of model building in [./docs/](./docs/):
 
- [carstm_julia.md](./docs/carstm_julia.md)
    
- [spatiotemporal_processes.md](./docs/spatiotemporal_processes.md)
       


## Installation 

Clone the repository or copy/download the Julia files (*.jl)


## Other useful references

Mitzi Morris: https://mc-stan.org/users/documentation/case-studies/icar_stan.html (very thorough)

Max Joseph: Exact sparse CAR models in Stan: https://github.com/mbjoseph/CARstan 

https://github.com/ConnorDonegan/Stan-IAR
https://github.com/ConnorDonegan/survey-HBM#car-models-in-stan

https://www.mdpi.com/1660-4601/18/13/6856

Besag, Julian, Jeremy York, and Annie Mollié. "Bayesian image restoration, with two applications in spatial statistics." Annals of the institute of statistical mathematics 43.1 (1991): 1-20.

Gelfand, Alan E., and Penelope Vounatsou. "Proper multivariate conditional autoregressive models for spatial data analysis." Biostatistics 4.1 (2003): 11-15.

Jin, Xiaoping, Bradley P. Carlin, and Sudipto Banerjee. "Generalized hierarchical multivariate CAR models for areal data." Biometrics 61.4 (2005): 950-961.

