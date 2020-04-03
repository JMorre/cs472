from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
from utils.metrics import *
from collections import Counter

filterwarnings('ignore')

hidden_layer_sizes = [
    (100,),
    (100, 100),
    (100, 100, 100),
    (200,),
    (200, 200),
    (200, 200, 200),
    (300,),
    (300, 300),
    (300, 300, 300),
]
activations = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['lbfgs', 'sgd', 'adam']
alphas = [.001, .0001, .00001]
learning_rates = ['constant', 'invscaling', 'adaptive']
learning_rate_inits = [.01, .001, .0001]
max_iters = [200, 300, 400, 500]
momentums = [0, .3, .6, .9]
nesterovs_momentums = [True, False]
early_stoppings = [True, False]
tols = [.0001, .001, .00001]

# batch size = ['auto']
# power_ts = [.5]
# shuffle = [True]
# beta_1s = [] used for adam
# beta_2s = [] used for adam
# epsilons = [] used for adam
# n_iter_no_changes = [8, 10, 15] used for sgd | adam
# max_funs = [15000] used for lbfgs


all_param_options = {
    'hl': hidden_layer_sizes,
    'mi': max_iters,
    'ac': activations,
    's': solvers,
    'al': alphas,
    'lr': learning_rates,
    'lri': learning_rate_inits,
    'm': momentums,
    'nm': nesterovs_momentums,
    'es': early_stoppings,
    't': tols
}


def init_params():
    params = {}
    for key in all_param_options:
        params[key] = all_param_options[key][0]
    return params


def init_mlp(params):
    return MLPClassifier(
        hidden_layer_sizes=params['hl'],
        activation=params['ac'],
        solver=params['s'],
        alpha=params['al'],
        learning_rate=params['lr'],
        learning_rate_init=params['lri'],
        max_iter=params['mi'],
        momentum=params['m'],
        nesterovs_momentum=params['nm'],
        early_stopping=params['es'],
        tol=params['t'],
        validation_fraction=0.25
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
            MClass = init_mlp(params)
            MClass.fit(X_train, y_train)
            y_pred = MClass.predict(X_test)
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


def run(X, y):
    initial(X, y)
    smote(X, y)
    undersampling(X, y)
    smote_undersampling(X, y)
