from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

def get_estimator(
    method: str = 'nearcent',
    random_state: int = None,
    max_features: int = 20,
    cv_method: int = 5,
    **kwargs
):
    if method == 'rlo' and cv_method is None:
        return LogisticRegression(
            solver='liblinear',
            class_weight = 'balanced',
            tol = 0.0001,
            random_state = random_state,
            **kwargs
        )

    if method == 'rlo':
        return LogisticRegressionCV(
            solver='liblinear',
            class_weight = 'balanced',
            tol = 0.0001,
            cv = cv_method,
            random_state = random_state,
            **kwargs
        )
        
    if method == 'rf' :
        return RandomForestClassifier(
            class_weight = 'balanced',
            bootstrap = False,
            n_jobs = -1,
            max_features = max_features,
            random_state = random_state,
            **kwargs
        )
        
    if method == 'svm':
        return SVC(
            class_weight = 'balanced',
            probability = True,
            random_state = random_state,
            **kwargs
        )

    if method == 'sdg':
        return SGDClassifier(
            n_jobs = -1,
            class_weight = 'balanced',
            random_state = random_state,
            **kwargs
        )

    if method == 'adaboost':
        return AdaBoostClassifier(
            random_state = random_state,
            **kwargs
        )

    return NearestCentroid()
