# Heisenberg Model MPS Practice

Code located under `src/`.  

## Reproducing Development Environments

Development environments could be saved and reproduced using package managers. 
The following package managers are supported:  

1. Conda  
2. Nix  

### Conda

Before continuing, please make sure Conda (at least the `conda` cli) is installed properly first.  

The development environment is stored as a conda environment named `hmm`.  
Environment specifications of `hmm` is written in the `environment.yml` file.  

For first time environment creation run

```
$ conda env create -f environment.yml -n hmm 
```

Or after remote changes, one could update local `hmm` environment from the updated `environment.yml` (pulling changes from others):  

```
$ conda env update --file environment.yml  --prune
```

Environments could now be activated and development could be done under the activated `hmm` environment:  

```
$ conda activate hmm
```

After development, environment changes should be exported to `environment.yml` (where we remove the `prefix:` attribute for cross-machine support):  

```
$ conda env export --from-history | grep -v "^prefix: " > environment.yml
```
