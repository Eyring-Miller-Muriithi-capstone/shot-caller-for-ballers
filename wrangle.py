import pandas as pd
import numpy as np
from re import T
import json
import sklearn.metrics as metrics
from itertools import product

from acquire import tome_prep
from acquire import acquire_shots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler


# -----
# Pre-Wrangle Functions
# -----

def game_shots(df):
    # Initialize game_id with the first player-game indexed in the dataframe
    game_id = 22100014

    # Need this to reset games counter
    player_id = 203992

    # Create lists to hold the running counts per player-game - these will become the columns
    count_hold_3pm = []
    count_hold_3pa = []
    # games counter will change only when game changes
    games_counter = []

    # Initialize counters for player-game 3pa and 3pm
    counter_3pm = 0
    counter_3pa = 0
    counter_games_played = 0

    # Loop through each df row.  Since it goes through all of a player's games first, if the game_id is new it is a new player-game
    for row in df.index:
        # If game is new, reset the counters
        if df.game_id[row] != game_id:
            counter_3pm = 0
            counter_3pa = 0
            # If player is new, reset this counter
            if df.player_id[row] != player_id:
                counter_games_played = 0
                player_id = df.player_id[row]
            else:
                counter_games_played += 1
        counter_3pa += 1
        if df['shot_result'][row] == 'Made Shot':
            counter_3pm += 1
        games_counter.append(counter_games_played)
        count_hold_3pm.append(counter_3pm)
        count_hold_3pa.append(counter_3pa)
        game_id = df.game_id[row]

    df['games_played'] = games_counter
    df['game_3pa'] = count_hold_3pa
    df['game_3pm'] = count_hold_3pm 
    df['game_3miss'] = df['game_3pa'] - df['game_3pm']

    return df

def season_shots(df):
    # Initialize game_id with the first player-game indexed in the dataframe
    player_id = 203992

    # Create lists to hold the running counts per player-game - these will become the columns
    to_date_season_3pm_hold = []
    to_date_season_3pa_hold = []

    # Initialize counters for player-game 3pa and 3pm
    counter_3pm = 0
    counter_3pa = 0

    # Loop through each df row.  Since it goes through all of a player's games first, if the game_id is new it is a new player-game
    for row in df.index:
        # If game is new, reset the counters
        if df.player_id[row] != player_id:
            counter_3pm = 0
            counter_3pa = 0
        counter_3pa += 1
        if df['shot_result'][row] == 'Made Shot':
            counter_3pm += 1
        to_date_season_3pm_hold.append(counter_3pm)
        to_date_season_3pa_hold.append(counter_3pa)
        player_id = df.player_id[row]
        
    df['cum_3pa'] = to_date_season_3pa_hold
    df['cum_3pm'] = to_date_season_3pm_hold 
    df['cum_3miss'] = df['cum_3pa'] - df['cum_3pm']

    return df

def create_metrics(df):

    # Simple 3pt percentage
    df['cum_3pct'] = df.cum_3pm/df.cum_3pa

    # Three metric v1 is made^2 / attempts
    df['tm_v1'] = (df.cum_3pm**2 / df.cum_3pa)/(df.games_played + 1)

    # Three metric v2 is made * [(1 - (made/attempts)) / 2 + (made / attempts)
    df['tm_v2'] = (df.cum_3pm * (( 1 - (df.cum_3pm/df.cum_3pa))/2 + (df.cum_3pm/df.cum_3pa))) / (df.games_played + 1)

    # Three metric v3 is made^2/miss
    df['tm_v3'] = (df.cum_3pm**2 / df.cum_3miss) / (df.games_played + 1)
    df['tm_v3'] = np.where(df['tm_v3'] == np.inf, 0, df['tm_v3'])

    return df

def create_location(df):
    '''
    '''
    df['location'] = ((df.loc_x/10)**2 + (df.loc_y/10)**2)**(1/2)

    return df

def create_game_event(df):
    df_shots, df_outlier_3pt = acquire_shots()
    game_events = df_shots[['GAME_ID','abs_time','GAME_EVENT_ID']]

    df = df.merge(game_events, how = 'inner', left_on = ('game_id','abs_time'), right_on = ('GAME_ID','abs_time'))

    df.drop(columns = 'GAME_ID', inplace = True)

    df.rename(columns = {'GAME_EVENT_ID':'game_event_id'}, inplace = True)

    return df, df_outlier_3pt

def encoder(df):
    '''
    '''
    # Encode target
    df['shot_made_flag'] = np.where(df.shot_result == 'Made Shot',1,0)

    # Encode these columns
    encode_cols = ['home','zone','shot_type']

    df_encoded = pd.get_dummies(df, columns = encode_cols)

    return df_encoded

def pre_wrangle_prep():
    df = tome_prep()
    df = game_shots(df)
    df = season_shots(df)
    df = create_metrics(df)
    df = create_location(df)
    df , df_outlier_3pt = create_game_event(df)
    df = encoder(df)
    train, validate, test = splitter(df, target = 'shot_result')
    X_train_exp = train
    train_scaled, validate_scaled, test_scaled = scaling_minmax(train, validate, test)
    X_train, y_train, X_validate, y_validate, X_test, y_test = seperate_X_y(train_scaled, validate_scaled, test_scaled) 

    return df, df_outlier_3pt, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

# -----
# Supporting Functions
# -----

def scaling_minmax(train, validate, test):

    '''
    This function takes in a data set that is split , makes a copy and uses the min max scaler to scale all three data sets. additionally it adds the columns names on the scaled data and returns trainedscaled data, validate scaled data and test scale
    '''
    # Columns to scale - only those with values that range greater than 0-10ish
    columns_to_scale = ['abs_time', 'play_time', 'since_rest', 'loc_x', 'loc_y', 'score_margin','points','cum_3pa', 'cum_3pm', 'cum_3miss','location']
    #copying the dataframes for distinguishing between scaled and unscaled data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # defining the minmax scaler 
    scaler = MinMaxScaler()
    
    #scaling the trained data and giving the scaled data column names 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    #scaling the validate data and giving the scaled data column names 
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    
    #scaling the test data and giving the scaled data column names 
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #returns three dataframes; train_scaled, validate_scaled, test_scaled
    return train_scaled, validate_scaled, test_scaled

def seperate_X_y(train_scaled, validate_scaled, test_scaled):
    '''
    '''
    
    target = 'shot_result'

    X_drop_columns_list = ['player', 'player_id', 'team', 'team_id', 'game_id','loc_x', 'loc_y','shot_result',
                        'games_played', 'game_3pa', 'game_3pm', 'game_3miss', 'cum_3pa', 'cum_3pm', 'cum_3miss',
                        'location', 'game_event_id', 'shot_made_flag','tm_v1','tm_v3']

    X_train = train_scaled.drop(columns = X_drop_columns_list)
    y_train = train_scaled[target]

    X_validate = validate_scaled.drop(columns = X_drop_columns_list)
    y_validate = validate_scaled[target]

    X_test = test_scaled.drop(columns = X_drop_columns_list)
    y_test = test_scaled[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# ------------MODELING BELOW -------------
#baseline accuracy
BASELINE_ACCURACY = 0.6424

#taxicab number
RAND_SEED = 1802

### FLOW CONTROL FUNCTIONS

def splitter(df, target = 'None', train_split_1 = .8, train_split_2 = .7, random_state = 123):
    '''
    Splits a dataset into train, validate and test dataframes.
    Optional target, with default splits of 56% 'Train' (80% * 70%), 20% 'Test', 24% Validate (80% * 30%)
    Defailt random seed/state of 123
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

def model_maker(X_train, y_train, X_validate, y_validate, baseline_acc = BASELINE_ACCURACY):
    """
    Makes a mass of models and returns a dataframe with accuracy metric
    """
    outputs = []
    #make logistic regression model
    outputs.append(make_log_reg_model(X_train, y_train, X_validate, y_validate, baseline_acc))
    #make knn and decision trees
    for i in range(1, 25):
        outputs.append(make_knn_model(X_train, y_train, X_validate, y_validate, i, baseline_acc))
        outputs.append(make_decision_tree_model(X_train, y_train, X_validate, y_validate, i, baseline_acc))
    rand_forest_params = return_product(3, 5, 4)
    print('starting rf and et')
    #make random forest and extra tree models
    for prod in rand_forest_params:
        #print('making rf and et')
        outputs.append(make_random_forest_model(X_train, y_train, X_validate, y_validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
        outputs.append(make_extra_trees_model(X_train, y_train, X_validate, y_validate, leaf=prod[0], depth=prod[1], trees = prod[2], baseline_acc = baseline_acc))
    print('finished rf and et')
    estimators = [
        {'model':LogisticRegression(), 'name':'LogisticRegression'},
        {'model':KNeighborsClassifier(), 'name':'KNeighborsClassifier'},
        {'model':DecisionTreeClassifier(), 'name':'DecisionTreeClassifier'},
        {'model':ExtraTreesClassifier(), 'name':'ExtraTreesClassifier'}
    ]
    #make ensemble model
    for estimator in estimators:
        outputs.append(make_bagging_classifier(X_train, y_train, X_validate, y_validate, estimator['model'], estimator['name'], baseline_acc = baseline_acc))
    return pd.DataFrame(outputs)

def baseline_model_maker(y_train, y_validate):
    """
    Creates a baseline model and returns metrics on train and validate sets
    """
    #get the relevant columns
    baseline_model = pd.DataFrame(y_train)
    baseline_model_val = pd.DataFrame(y_validate)
    #prediction is the mode of the langauges
    baseline_model['predicted'] = y_train.mode().to_list()[0]
    baseline_model_val['predicted'] = y_train.mode().to_list()[0]
    #get the metrics
    metrics_dict = metrics.classification_report(y_train, baseline_model['predicted'], output_dict=True, zero_division = True)
    metrics_dict_val = metrics.classification_report(y_validate, baseline_model_val['predicted'], output_dict=True, zero_division = True)
    output = {
        'model':'Baseline Model',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
    }
    #return the metrics as a dataframe and the baseline accuracy for the model
    return pd.DataFrame([output]), metrics_dict['accuracy']


def test_model(X_train, y_train, X_validate, y_validate, X_test, y_test, baseline_acc = BASELINE_ACCURACY):
    """
    Final model trained, and then run on unseen test data.  Dataframe of metrics returned
    """
    #make and train the model
    bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators = 20, max_samples = 0.5, max_features = 0.5, bootstrap=False, random_state = RAND_SEED).fit(X_train, y_train['language'])
    
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    #get predictions
    y_train['predicted'] = bc.predict(X_train)
    y_validate['predicted'] = bc.predict(X_validate)
    y_test['predicted'] = bc.predict(X_test)
    #get the metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    metrics_dict_test = metrics.classification_report(y_test['shot_result'], y_test['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'BaggingClassifier',
        'attributes':f'estimator = DecisionTreeClassifier',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'test_accuracy':metrics_dict_test['accuracy'],
        'better_than_baseline':metrics_dict_test['accuracy'] > baseline_acc
    }
    #return the dataframe
    return pd.DataFrame([output])

### UTILITY FUNCTIONS

# SPLITTER

def return_product(l, d, t):
    """
    makes a itertools object iterable for the random forest and extra trees models
    """
    #make the range sets
    leaf_vals = range(1,l)
    depth_vals = range(2,d)
    #make tree values starting at 100 and going up in steps of 50
    tree_values = range(100, t*100, 50)
    #make the cartesian product
    product_output = product(leaf_vals, depth_vals, tree_values)
    return product_output


### MODEL MAKERS

def make_bagging_classifier(X_train, y_train, X_validate, y_validate, estimator, estimator_name, baseline_acc = BASELINE_ACCURACY):
    """
    Makes a bagging classifier based on passed estimator
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    #fit model
    bc = BaggingClassifier(base_estimator=estimator, n_estimators = 20, max_samples = 0.5, max_features = 0.5, bootstrap=False, random_state = RAND_SEED).fit(X_train, y_train['language'])
    #predict
    y_train['predicted'] = bc.predict(X_train)
    y_validate['predicted'] = bc.predict(X_validate)
    #get metrics
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
    """
    makes a logistic regression model
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    #REMOVE BELOW----X
    X_train = X_train.drop(columns = ['zone','shot_type'])
    X_validate = X_validate.drop(columns = ['zone','shot_type'])
    #REMOVE ABOVE----X

    #fit the model
    lm = LogisticRegression().fit(X_train, y_train['shot_result'])
    #make predictions
    y_train['predicted'] = lm.predict(X_train)
    y_validate['predicted'] = lm.predict(X_validate)
    # get metrics
    metrics_dict = metrics.classification_report(y_train['shot_result'], y_train['predicted'], output_dict=True, zero_division=True)
    metrics_dict_val = metrics.classification_report(y_validate['shot_result'], y_validate['predicted'], output_dict=True, zero_division=True)
    output = {
        'model':'LogisticRegression',
        'attributes':'None',
        'train_accuracy': metrics_dict['accuracy'],
        'validate_accuracy': metrics_dict_val['accuracy'],
        'better_than_baseline':metrics_dict['accuracy'] > baseline_acc and metrics_dict_val['accuracy'] > baseline_acc
    }
    return output

def make_knn_model(X_train, y_train, X_validate, y_validate, neighbors, baseline_acc = BASELINE_ACCURACY):
    """
    Make a knn model
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    #fit the model
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn = knn.fit(X_train, y_train['shot_result'])
    #make predictions
    y_train['predicted'] = knn.predict(X_train)
    y_validate['predicted'] = knn.predict(X_validate)
    #get metrics
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
    """
    Makes a decision tree model and returns a dictionary containing calculated accuracy metrics
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    #make and fit the model
    dt = DecisionTreeClassifier(max_depth = depth, random_state=RAND_SEED)
    dt = dt.fit(X_train, y_train['shot_result'])
    #make predictions
    y_train['predicted'] = dt.predict(X_train)
    y_validate['predicted'] = dt.predict(X_validate)
    # calculate metrics
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
    """
    Makes a random forest model
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    #make and fit the model
    rf = RandomForestClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    rf = rf.fit(X_train, y_train['shot_result'])
    #make predictions
    y_train['predicted'] = rf.predict(X_train)
    y_validate['predicted'] = rf.predict(X_validate)
    # calculate metrics
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
    """
    makes an extra trees model
    """
    # Turn y Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    #make and fit the model
    et = ExtraTreesClassifier(min_samples_leaf = leaf, max_depth=depth, n_estimators=trees, random_state=RAND_SEED)
    et = et.fit(X_train, y_train['shot_result'])
    #make predictions
    y_train['predicted'] = et.predict(X_train)
    y_validate['predicted'] = et.predict(X_validate)
    # calculate metrics
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