import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None,
                             max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

# global variables
def prep_data():
    missing_indicator = -1000
    train = pd.read_csv("../1RawData/train.csv")
    test = pd.read_csv("../1TestData/test.csv")
    # combine train and test
    all_data = train.append(test)
    # create any new variables    
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)
    all_data.fillna(missing_indicator, inplace=True)
    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)
    # split train and test
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()
    return train, test


def train_offset(x0, y, train_preds):
    '''
    Finding offsets
    '''
    res = digit(x0, train_preds)
    return -quadratic_weighted_kappa(y, res)


def digit(x0, train_preds):
    '''
    Digitize train list
    '''
    res = []
    for y in list(train_preds):
        limit = True
        for index, value in enumerate(x0):
            if y < value:
                res.append(index + 1)
                limit = False
                break
        if limit:
            res.append(index + 2)
    return res


def feature_importance_plot(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 25))
    sns.barplot(feature_importance.values, feature_importance.index,
                orient='h', ax=ax).tick_params(labelsize=8)
    plt.show()
