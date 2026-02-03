
## Species composition

A PyMC example


```python

# Get species composition data 
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc import HalfCauchy, Model, Normal, sample

print(f"Running on PyMC v{pm.__version__}")

data_folder = Path("projects/model_covariance/data/")

set = pd.read_pickle(data_folder / "set.pickle") 
m = pd.read_pickle(data_folder / "m.pickle") 
obs = pd.read_pickle(data_folder / "obs.pickle") 
preds = pd.read_pickle(data_folder / "preds.pickle") 
taxa = pd.read_pickle(data_folder / "taxa.pickle") 
ids = pd.read_pickle(data_folder / "ids.pickle") 
nb = pd.read_pickle(data_folder / "nb.pickle") 

```

### Warmup: simple GLM

```python

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

x0 = np.log10(obs.z)
y = obs.pca1

with Model() as model1:  
    sigma = HalfCauchy("sigma", beta=10)
    intercept = Normal("Intercept", 0, sigma=20)
    slope = Normal("slope", 0, sigma=20)
    likelihood = Normal("y", mu=intercept + slope * x0, sigma=sigma, observed=y)
    idata = sample(3000)

az.plot_trace(idata, figsize=(10, 7));
plt.show()

idata.posterior["y_model"] = idata.posterior["Intercept"] + idata.posterior["slope"] * xr.DataArray(x)

_, ax = plt.subplots(figsize=(7, 7))
az.plot_lm(idata=idata, y="y", num_samples=100, axes=ax, y_model="y_model")
ax.set_title("Posterior predictive regression lines")
ax.set_xlabel("x");
plt.show()


x0 = np.log10(obs.z)
x1 = obs.t
y = obs.pca1

with Model() as model2:  
    sigma = HalfCauchy("sigma", beta=10)
    intercept = Normal("Intercept", 0, sigma=20)
    slope = Normal("slope", 0, sigma=20, shape=2)
    mu = intercept + slope[0] * x0 +  slope[1] * x1
    llik = Normal("y", mu=mu , sigma=sigma, observed=y)
    idata2 = sample(3000)

az.plot_forest(idata2, var_names="slope");
plt.show()

az.plot_trace(idata2 );
plt.show()

# diagnostics
with model:
    pm.compute_log_likelihood(idata)

with model2:
    pm.compute_log_likelihood(idata2)

pooled_loo2 = az.loo(idata2)
pooled_loo = az.loo(idata)

df_comp_loo = az.compare({"model": idata, "model2": idata2})
df_comp_loo

az.waic(idata)
az.waic(idata2)

# for more details, see:
# https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html

# numpyro

x = np.log10(obs.z)
y = obs.pca1

with Model() as model_gpu:
    sigma = HalfCauchy("sigma", beta=10)
    intercept = Normal("Intercept", 0, sigma=20)
    slope = Normal("slope", 0, sigma=20)
    mu = intercept + slope * x
    lik = Normal("y", mu=mu , sigma=sigma, observed=y)
   

jax.local_device_count() 

numpyro.set_host_device_count(4)

with model_gpu:
    res = sample(nuts_sampler="numpyro")  # others too blackjax, etc ..

# hierarchical_trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
# print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))

```

