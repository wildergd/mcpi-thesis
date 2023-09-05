from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

def get_estimator(
    method: str = 'nearcent',
    random_state: int = None,
    max_features: int = 20
):
    if method == 'rlo':
        return LogisticRegressionCV(
            solver='liblinear',
            class_weight = 'balanced',
            cv = 5,
            tol = 0.05,
            random_state = random_state
        )

    if method == 'rf':
        return RandomForestClassifier(
            class_weight = 'balanced',
            bootstrap = False,
            n_jobs = -1,
            max_features = max_features,
            random_state = random_state
        )
        
    if method == 'svm':
        return SVC(
            class_weight = 'balanced',
            probability = True,
            random_state = random_state
        )

    if method == 'sdg':
        return SGDClassifier(
            n_jobs = -1,
            class_weight = 'balanced',
            random_state = random_state
        )

    if method == 'adaboost':
        return AdaBoostClassifier(
            random_state = random_state
        )

    return NearestCentroid()
