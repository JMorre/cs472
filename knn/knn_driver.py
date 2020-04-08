from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from warnings import filterwarnings
from utils.metrics import *
from collections import Counter
import numpy as np

filterwarnings('ignore')

features = [
    'position',
    'height',
    'weight',
    'field_goals',
    'field_goal_attempts',
    'field_goal_percentage',
    'three_pointers',
    'three_point_attempts',
    'three_point_percentage',
    'three_pointers',
    'three_point_attempts',
    'three_point_percentage',
    'effective_field_goal_percentage',
    'free_throws',
    'free_throw_attempts',
    'free_throw_percentage',
    'offensive_rebounds',
    'defensive_rebounds',
    'total_rebounds',
    'assists',
    'steals',
    'blocks',
    'turnovers',
    'personal_fouls',
    'points',
    'true_shooting_percentage',
    'three_point_attempt_rate',
    'free_throw_attempt_rate',
    'offensive_rebound_percentage',
    'defensive_rebound_percentage',
    'total_rebound_percentage',
    'assist_percentage',
    'steal_percentage',
    'block_percentage',
    'turnover_percentage',
    'usage_percentage',
]

n_neighbors = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size = [5, 10, 20, 30, 40, 50, 100, 200]
p = [3, 4, 5, 6, 7, 8, 9, 10]
metric = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']


all_param_options = {
    'k': n_neighbors,
    'w': weights,
    'a': algorithm,
    's': leaf_size,
    'p': p,
    'm': metric
}


def init_params():
    params = {}
    for key in all_param_options:
        params[key] = all_param_options[key][0]
    return params


def init_knn(params):
    return KNeighborsClassifier(
        n_neighbors = params['k'],
        weights = params['w'],
        algorithm = params['a'],
        leaf_size = params['s'],
        p = params['p']
    )


def find_best_params(X_train, y_train, X_test, y_test):
    params = init_params()
    best_overall_results = (0, 0, 0)
    best_overall_params = None
    for param_key in all_param_options:
        param_options = all_param_options[param_key]
        best_param_option_results = (0, 0, 0)
        best_param_option_idx = 0
        for option_idx in range(len(param_options)):
            params[param_key] = param_options[option_idx]
            if params['m'] != 'minkowski': params['p'] = 2
            KClass = init_knn(params)
            KClass.fit(X_train, y_train)
            y_pred = KClass.predict(X_test)
            current_results = precision_recall_score(y_test, y_pred)
            if total_quality(current_results) > total_quality(best_param_option_results):
                best_param_option_results = current_results
                best_param_option_idx = option_idx
            if total_quality(current_results) > total_quality(best_overall_results):
                best_overall_results = current_results
                best_overall_params = params.copy()
                print('New Best', best_overall_results)
        params[param_key] = param_options[best_param_option_idx]

    print('**********************')
    precision, recall, score = best_overall_results
    print(f"best overall params: {best_overall_params}")
    print(f"Yielded precision: {precision}, recall: {recall}, score: {score}")
    print(f"Total quality rating: {total_quality(best_overall_results)}")


def reduce_PCA(X, y):
    pca = PCA(n_components=min(len(X), len(X[0])) - 2, svd_solver="full")
    newX = pca.fit_transform(X, y)
    return newX


def reduce_wrapper(X, y):
    reducedX = np.copy(X)
    kept_features = features[:]
    index = None
    keepGoing = True
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    original = knn.score(X_test, y_test)
    overall_best = original
    while keepGoing and len(kept_features) > 7:
        best = 0
        keepGoing = False
        for i in range(len(reducedX[0])):
            tempX = np.delete(reducedX, i, 1)
            X_train, X_test, y_train, y_test = train_test_split(tempX, y, test_size=0.25)

            knn.fit(X_train, y_train)
            tempBest = knn.score(X_test, y_test)
            if tempBest > best:
                index = i
                best = tempBest
        if original - best <= .02 and overall_best - best <= .04:
            reducedX = np.delete(reducedX, index, 1)
            kept_features.pop(index)
            keepGoing = True
            index = None
            overall_best = max(overall_best, best)
    print(f"Kept features: {kept_features}")
    return reducedX


def initial(X, y):
    print('Initial...')
    print('class distribution:', Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    find_best_params(X_train, y_train, X_test, y_test)
    print('\n\n')


def smote(X, y):
    print('SMOTE...')
    oversample = SMOTE(k_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    new_X, new_y = oversample.fit_resample(X_train, y_train)
    print('Oversampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def undersampling(X, y):
    print('Under-sampling...')
    undersample = RandomUnderSampler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    new_X, new_y = undersample.fit_resample(X_train, y_train)
    print('Undersampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def smote_undersampling(X, y):
    print('SMOTE and under-sampling...')
    oversample = SMOTE(k_neighbors=5)
    undersample = RandomUnderSampler()

    steps = [('o', oversample), ('u', undersample)]
    combo = Pipeline(steps=steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    new_X, new_y = combo.fit_resample(X_train, y_train)
    print('Combo class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def run_PCA(X, y):
    print('PCA...')
    print('Pre PCA shape: ', X.shape)
    reducedX = reduce_PCA(X, y)
    print('Post PCA shape: ', reducedX.shape)
    print('PCA class distribution:')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    find_best_params(X_train, y_train, X_test, y_test)
    print('\n\n')


def run_Wrapper(X, y):
    print('Backwards wrapper...')
    print('Pre wrapper shape: ', X.shape)
    reducedX = reduce_wrapper(X, y)
    print('Post wrapper shape: ', reducedX.shape)
    print('Backwards wrapper class distribution:')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    find_best_params(X_train, y_train, X_test, y_test)
    print('\n\n')


def pca_smote(X, y):
    print('PCA and SMOTE...')
    print('Pre PCA shape: ', X.shape)
    reducedX = reduce_PCA(X, y)
    print('Post PCA shape: ', reducedX.shape)
    oversample = SMOTE(k_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = oversample.fit_resample(X_train, y_train)
    print('Oversampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def pca_undersampling(X, y):
    print('PCA and Under-sampling...')
    print('Pre PCA shape: ', X.shape)
    reducedX = reduce_PCA(X, y)
    print('Post PCA shape: ', reducedX.shape)
    undersample = RandomUnderSampler()
    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = undersample.fit_resample(X_train, y_train)
    print('Undersampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def pca_smote_undersampling(X, y):
    print('PCA, SMOTE, and under-sampling...')
    oversample = SMOTE(k_neighbors=5)
    undersample = RandomUnderSampler()
    print('Pre PCA shape: ', X.shape)
    reducedX = reduce_PCA(X, y)
    print('Post PCA shape: ', reducedX.shape)

    steps = [('o', oversample), ('u', undersample)]
    combo = Pipeline(steps=steps)

    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = combo.fit_resample(X_train, y_train)
    print('Combo class distribution:', Counter(new_y))
    # X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.25)
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def wrapper_smote(X, y):
    print('Backwards wrapper and SMOTE...')
    print('Pre Wrapper shape: ', X.shape)
    reducedX = reduce_wrapper(X, y)
    print('Post Wrapper shape: ', reducedX.shape)
    oversample = SMOTE(k_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = oversample.fit_resample(X_train, y_train)
    print('Oversampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def wrapper_undersampling(X, y):
    print('Backwards Wrapper and Under-sampling...')
    print('Pre Wrapper shape: ', X.shape)
    reducedX = reduce_wrapper(X, y)
    print('Post Wrapper shape: ', reducedX.shape)
    undersample = RandomUnderSampler()
    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = undersample.fit_resample(X_train, y_train)
    print('Undersampled class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def wrapper_smote_undersampling(X, y):
    print('Backwards wrapper, SMOTE, and under-sampling...')
    oversample = SMOTE(k_neighbors=5)
    undersample = RandomUnderSampler()
    print('Pre Wrapper shape: ', X.shape)
    reducedX = reduce_wrapper(X, y)
    print('Post Wrapper shape: ', reducedX.shape)

    steps = [('o', oversample), ('u', undersample)]
    combo = Pipeline(steps=steps)

    X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size=0.25)
    new_X, new_y = combo.fit_resample(X_train, y_train)
    print('Combo class distribution:', Counter(new_y))
    find_best_params(new_X, new_y, X_test, y_test)
    print('\n\n')


def run(X, y):
    initial(X, y)
    run_PCA(X, y)
    run_Wrapper(X, y)
    smote(X, y)
    pca_smote(X, y)
    wrapper_smote(X, y)
    undersampling(X, y)
    pca_undersampling(X, y)
    wrapper_undersampling(X, y)
    smote_undersampling(X, y)
    pca_smote_undersampling(X, y)
    wrapper_smote_undersampling(X, y)
