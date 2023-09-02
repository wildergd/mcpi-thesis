# Detection of the Possible Occurrence of Depressive Episodes in Patients with Multiple Sclerosis from Motor Activity Data

> Project to obtain Master Degree in Data Science

## Data preparation
### Dataset
Depresjon dataset proposed by Garcia Ceja et. al. (2018), available at [Simula Datasets](https://datasets.simula.no/depresjon/)

### Generating datasets for classification

```bash
$ tools/generate_classification_datasets.py -f 1H -sm sum -st zscore -sto no -cf all
$ tools/generate_classification_datasets.py -f 1H -sm sum -st zscore -sto yes -cf all
$ tools/generate_classification_datasets.py -f 1H -sm sum -st zscore-robust -sto no -cf all
$ tools/generate_classification_datasets.py -f 1H -sm sum -st zscore-robust -sto yes -cf all
$ tools/generate_classification_datasets.py -f 1H -sm mean -st zscore -sto no -cf all
$ tools/generate_classification_datasets.py -f 1H -sm mean -st zscore -sto yes -cf all
$ tools/generate_classification_datasets.py -f 1H -sm mean -st zscore-robust -sto no -cf all
$ tools/generate_classification_datasets.py -f 1H -sm mean -st zscore-robust -sto yes -cf all
```
### spliting datasets

```bash
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv

$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv

$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes

$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes
$ tools/split_dataset.py -f ../dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes
```

### Feature extraction

### Training and testing models

## Forecasting activity


## References
- Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T., Oedegaard, K. J., & Fasmer, O. B. (2018). Depresjon: A motor activity database of depression episodes in unipolar and bipolar patients. Proceedings of the 9th ACM Multimedia Systems Conference, MMSys 2018, 472–477. [https://doi.org/10.1145/3204949.3208125](https://doi.org/10.1145/3204949.3208125)