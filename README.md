# Detection of the Possible Occurrence of Depressive Episodes in Patients with Multiple Sclerosis from Motor Activity Data

> Project to obtain the Master's Degree in Data Science

## Data preparation

### Dataset
Depresjon dataset proposed by Garcia Ceja et. al. (2018), available at [Simula Datasets](https://datasets.simula.no/depresjon/)

### Install dependencies

```bash
$ pip install -r python-code/requirements.txt
```

### Generating datasets for classification

```bash
$ scripts/generate_classification_datasets.sh
```

### Spliting datasets

```bash
$ scripts/split_classification_datasets.sh
```

### Feature extraction

```bash
$ scripts/feature_selection_all.sh -m rlo -v loo
```

or pass several models separated by comma

```bash
$ scripts/feature_selection_all.sh  -m nearcent,rlo,svm,sgd,rf,adaboost -v loo 
```

param *-v* can be either ***loo*** for *Leave One Out* validation or an integer specifying the ***number of folds*** in order to use *k-fold* cross validation

> *Note:* feature generation is a very slow process and can be slower when using *Leave one out (loo)* validation

### Training and testing models

```bash
$ scripts/build_classification_models.sh
```

## Forecasting activity

```bash
$ scripts/build_forecasting_models.sh
```

## Results analysis

### Classification models results

```bash
$ jupyter notebook python-code/notebooks/classification_models_results_analysis.ipynb
```

### forecasting models
```bash
$ jupyter notebook python-code/notebooks/forecasting_models.ipynb
```

### Forecasting Classification results

```bash
$ jupyter notebook python-code/notebooks/deployment_results_analysis_mape.ipynb
$ jupyter notebook python-code/notebooks/deployment_results_analysis_rmse.ipynb
```

## References
- Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T., Oedegaard, K. J., & Fasmer, O. B. (2018). Depresjon: A motor activity database of depression episodes in unipolar and bipolar patients. Proceedings of the 9th ACM Multimedia Systems Conference, MMSys 2018, 472–477. [https://doi.org/10.1145/3204949.3208125](https://doi.org/10.1145/3204949.3208125)