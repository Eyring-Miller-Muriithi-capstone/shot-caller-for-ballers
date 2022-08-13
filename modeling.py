import pandas as pd
import json
import sklearn.metrics as metrics
from itertools import product

from explore import find_elites
from wrangle import wrangle_prep_player

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------------------------
# Main Functions
# ------------------------------------------------------------------------------------------------

BASELINE_ACCURACY = 0.6424

RAND_SEED = 123

def column_dropper(X_train, X_validate, X_test, drop_list):
    '''
    Drops the features listed in the drop_list parameter from modeling.
    '''
    X_train = X_train.drop(columns = drop_list)
    X_validate = X_validate.drop(columns = drop_list)
    X_test = X_test.drop(columns = drop_list)

    return X_train, X_validate, X_test


def model_maker(X_train, y_train, X_validate, y_validate, drop_list, baseline_acc = BASELINE_ACCURACY):
    '''
    Makes a mass of models and returns a dataframe with accuracy metric
    '''
    X_train, X_validate = column_dropper(X_train, X_validate,drop_list)
    outputs = []
    # Make logistic regression model
    outputs.append(make_log_reg_model(X_train, y_train, X_validate, y_validate, baseline_acc))
    # Make knn and decision trees
    for i in range(1, 25):
        outputs.append(make_knn_model(X_train, y_train, X_validate, y_validate, i, baseline_acc))
        outputs.append(make_decision_tree_model(X_train, y_train, X_validate, y_validate, i, baseline_acc))
    rand_forest_params = return_product(3, 5, 4)
    # Make random forest and extra tree models
    for prod in rand_forest_params:
        outputs.append(make_random_forest_model(X_train, y_train, X_validate, y_validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
        outputs.append(make_extra_trees_model(X_train, y_train, X_validate, y_validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
    estimators = [
        {'model':LogisticRegression(), 'name':'LogisticRegression'},
        {'model':KNeighborsClassifier(), 'name':'KNeighborsClassifier'},
        {'model':DecisionTreeClassifier(), 'name':'DecisionTreeClassifier'},
        {'model':ExtraTreesClassifier(), 'name':'ExtraTreesClassifier'}
    ]
    # Make ensemble model
    for estimator in estimators:
        outputs.append(make_bagging_classifier(X_train, y_train, X_validate, y_validate, estimator['model'], estimator['name'], baseline_acc = baseline_acc))
    return pd.DataFrame(outputs)


def baseline_model_maker(y_train, y_validate):
    '''
    Creates a baseline model and returns metrics on train and validate sets
    '''
    # Get the relevant columns
    baseline_model = pd.DataFrame(y_train)
    baseline_model_val = pd.DataFrame(y_validate)
    # Prediction is the mode of the langauges
    baseline_model['predicted'] = y_train.mode().to_list()[0]
    baseline_model_val['predicted'] = y_train.mode().to_list()[0]
    # Get the metrics
    metrics_dict = metrics.classification_report(y_train, baseline_model['predicted'], output_dict=True, zero_division = True)
    metrics_dict_val = metrics.classification_report(y_validate, baseline_model_val['predicted'], output_dict=True, zero_division = True)
    output = {
        'model':'Baseline Model',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
    }
    # Return the metrics as a dataframe and the baseline accuracy for the model
    return pd.DataFrame([output]), metrics_dict['accuracy']


def test_model(X_train, y_train, X_validate, y_validate, X_test, y_test, drop_list, baseline_acc = BASELINE_ACCURACY):
    '''
    Final model trained, and then run on unseen test data.  Dataframe of metrics returned.
    This is hardcoded in based on the results of the model maker analysis above.
    '''
    # Drop those from the drop_list
    X_train, X_validate, X_test= column_dropper(X_train, X_validate, X_test, drop_list)

     # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    # Make and train the model - this must be manually hardcoded (future change is to automate this)
    dt = DecisionTreeClassifier(max_depth = 6, random_state=RAND_SEED)
    dt = dt.fit(X_train, y_train['shot_result'])

    # Get predictions
    y_train['predicted'] = dt.predict(X_train)
    y_validate['predicted'] = dt.predict(X_validate)
    y_test['predicted'] = dt.predict(X_test)
    
    # Get the metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    metrics_dict_test = metrics.classification_report(y_test['shot_result'], y_test['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Decision Tree',
        'attributes':'max_depth = 6',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'test_accuracy':metrics_dict_test['accuracy'],
        'better_than_baseline':metrics_dict_test['accuracy'] > baseline_acc,
        'beats_baseline_by':metrics_dict_test['accuracy'] - baseline_acc
    }
   
    return pd.DataFrame([output])


# ------------------------------------------------------------------------------------------------
# Elite Players Analysis Function
# ------------------------------------------------------------------------------------------------


def best_model_elites(df, X_train, y_train, X_validate, y_validate):
    '''
    This function creates an list of 'elite' three point shooters (as measured by Jem-metrics, aka tm_v2)
    It returns a mdoel based on their shots alone for the season    
    '''
    elites_list = find_elites(df)

    df_p = df[df.player.isin(elites_list)]

    # Create data sctructure containing both player name and player_id
    player_id_list = df_p.player_id.unique()
    player_name_list = df_p.player.unique()
    elites_tuple = list(zip(player_id_list, player_name_list))

    # Create dataframe to hold information
    best_models = pd.DataFrame()
    # Loops through elite players
    for player in elites_tuple:
        print('>',player[1])
        # Pulls up player specific df
        df, df_outlier_3pt, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle_prep_player(player[0])
        # Models player's baseline
        baseline_model_maker(y_train, y_validate)[0]
        BASELINE_ACCURACY = baseline_model_maker(y_train, y_validate)[1]
        # Use multi classificaiotn and ensemble model analysis on individual player
        models = model_maker(X_train, y_train, X_validate, y_validate, ['abs_time','play_time'], baseline_acc = BASELINE_ACCURACY)
        # Take the best model, append the name, baseline and calucalte validate difference from baseline
        best_model = models[models.better_than_baseline == True].sort_values('validate_accuracy', ascending = False).head(1)
        best_model['baseline'] = BASELINE_ACCURACY
        best_model['player'] = player[1]
        best_model['validate_improvement_over_baseline'] = best_model.validate_accuracy - best_model.baseline
        # Merge it with existing dataframe tracking top models for elite players
        best_models = pd.concat([best_models, best_model])

    # Set index to player and slice out relevant columns for dataframe
    best_models = best_models.set_index('player')
    best_models = best_models[['model','attributes','baseline','train_accuracy','validate_accuracy', 'val_improvement_over_baseline']]

    return best_models


# ------------------------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------------------------


def splitter(df, target = 'None', train_split_1 = .8, train_split_2 = .7, random_state = 123):
    '''
    Splits a dataset into train, validate and test dataframes.
    Optional target, with default splits of 56% 'Train' (80% * 70%), 20% 'Test', 24% Validate (80% * 30%)
    Default random seed/state of 123
    '''
    if target == 'None':
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state)
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state)
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        print('You did not stratify.  If looking to stratify, ensure to add argument: "target = variable to stratify on".')
        return train, validate, test
    else: 
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state, stratify = df[target])
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state, stratify = train[target])
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        return train, validate, test  


def return_product(l, d, t):
    '''
    Makes a itertools object iterable for the random forest and extra trees models
    '''
    # Make the range sets
    leaf_vals = range(1,l)
    depth_vals = range(2,d)
    # Make tree values starting at 100 and going up in steps of 50
    tree_values = range(100, t*100, 50)
    # Make the cartesian product
    product_output = product(leaf_vals, depth_vals, tree_values)

    return product_output


# ------------------------------------------------------------------------------------------------
# Modeling Functions
# ------------------------------------------------------------------------------------------------


def make_bagging_classifier(X_train, y_train, X_validate, y_validate, estimator, estimator_name, baseline_acc = BASELINE_ACCURACY):
    '''
    Makes a bagging classifier based on passed estimator
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Fit model
    bc = BaggingClassifier(base_estimator=estimator, n_estimators = 20, max_samples = 0.5, max_features = 0.5, bootstrap=False, random_state = RAND_SEED).fit(X_train, y_train['shot_result'])
    # Predict
    y_train['predicted'] = bc.predict(X_train)
    y_validate['predicted'] = bc.predict(X_validate)
    #g Get metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'BaggingClassifier',
        'attributes':f'estimator = {estimator_name}',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }

    return output


def make_log_reg_model(X_train, y_train, X_validate, y_validate, baseline_acc = BASELINE_ACCURACY):
    '''
    makes a logistic regression model
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Fit the model
    lm = LogisticRegression().fit(X_train, y_train['shot_result'])
    # Make predictions
    y_train['predicted'] = lm.predict(X_train)
    y_validate['predicted'] = lm.predict(X_validate)
    # Get metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'LogisticRegression',
        'attributes':'None',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc}

    return output


def make_knn_model(X_train, y_train, X_validate, y_validate, neighbors, baseline_acc = BASELINE_ACCURACY):
    '''
    Make a knn model
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Fit the model
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn = knn.fit(X_train, y_train['shot_result'])
    # Make predictions
    y_train['predicted'] = knn.predict(X_train)
    y_validate['predicted'] = knn.predict(X_validate)
    # Get metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'KNNeighbors',
        'attributes':f'n_neighbors = {neighbors}',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }

    return output


def make_decision_tree_model(X_train, y_train, X_validate, y_validate, depth, baseline_acc = BASELINE_ACCURACY):
    '''
    Makes a decision tree model and returns a dictionary containing calculated accuracy metrics
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Make and fit the model
    dt = DecisionTreeClassifier(max_depth = depth, random_state=RAND_SEED)
    dt = dt.fit(X_train, y_train['shot_result'])
    # Make predictions
    y_train['predicted'] = dt.predict(X_train)
    y_validate['predicted'] = dt.predict(X_validate)
    # Calculate metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Decision Tree Classifier',
        'attributes': f"max_depth = {depth}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
   
    return output


def make_random_forest_model(X_train, y_train, X_validate, y_validate, leaf, depth, trees, baseline_acc = BASELINE_ACCURACY):
    '''
    Makes a random forest model
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Make and fit the model
    rf = RandomForestClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    rf = rf.fit(X_train, y_train['shot_result'])
    # Make predictions
    y_train['predicted'] = rf.predict(X_train)
    y_validate['predicted'] = rf.predict(X_validate)
    # Calculate metric`s
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Random Forest Classifier',
        'attributes': f"leafs = {leaf} : depth = {depth} : trees = {trees}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    
    return output


def make_extra_trees_model(X_train, y_train, X_validate, y_validate, leaf, depth, trees, baseline_acc = BASELINE_ACCURACY):
    '''
    Makes an extra trees model
    '''
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # Make and fit the model
    et = ExtraTreesClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    et = et.fit(X_train, y_train['shot_result'])
    # Make predictions
    y_train['predicted'] = et.predict(X_train)
    y_validate['predicted'] = et.predict(X_validate)
    # Calculate metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'Extra Trees Classifier',
        'attributes': f"leafs = {leaf} : depth = {depth} : trees = {trees}",
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    
    return output


# ------------------------------------------------------------------------------------------------
# Predictions file creation for league_wide analysis
# ------------------------------------------------------------------------------------------------


def predictions_generator(df, X_train, y_train, X_test, y_test):
    '''
    Generates a prediction csv comparing predicted miss/made vs actual.
    Also includes classification percentages.  Only for the 
    '''
    # Create and fit a classifier object on train based on the best model
    tree = DecisionTreeClassifier(max_depth=6, random_state=123)
    tree = tree.fit(X_train, y_train)

    # Use object to predict for y_predicted and y_probability of prediction
    y_tree_predict = tree.predict(X_test)
    y_tree_proba = tree.predict_proba(X_test)
    # Turn into dataframe and concatenate
    proba_df = pd.DataFrame(y_tree_proba, columns=tree.classes_.tolist()).round(4)
    reset_test = (pd.concat([X_test, y_test], axis = 1).reset_index())
    test_proba_df = pd.concat([reset_test, proba_df], axis=1)
    test_proba_df = test_proba_df.merge(df, how = 'inner', left_on = 'index', right_index = True)
    test_proba_df['predicted'] = y_tree_predict
    csv_df = test_proba_df[['player','index','Made Shot', 'Missed Shot', 'predicted','shot_result_x']]
    csv_df = csv_df.rename(columns = {'shot_result_x':'actual'})
    notes = csv_df.predicted.value_counts()
    print(f'Predicted missed shots: {notes[0]}\nPredicted made shots: {notes[1]}')
    csv_df.to_csv('predictions.csv')
    print("Predictions saved to 'predictions.csv'")
    
    return csv_df