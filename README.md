# Detection of the Possible Occurrence of Depressive Episodes in Patients with Multiple Sclerosis from Motor Activity Data

> Project to obtain Master Degree in Data Science

## Data preparation
### Dataset
Depresjon dataset proposed by Garcia Ceja et. al. (2018), available at [Simula Datasets](https://datasets.simula.no/depresjon/)

### Generating datasets for classification

```bash
$ scripts/generate_datasets.sh
```
### spliting datasets

```bash
$ scripts/split_datasets.sh
```

### Feature extraction

```bash
$ scripts/feature_selection.sh
```

### Training and testing models

## Forecasting activity


## References
- Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T., Oedegaard, K. J., & Fasmer, O. B. (2018). Depresjon: A motor activity database of depression episodes in unipolar and bipolar patients. Proceedings of the 9th ACM Multimedia Systems Conference, MMSys 2018, 472–477. [https://doi.org/10.1145/3204949.3208125](https://doi.org/10.1145/3204949.3208125)