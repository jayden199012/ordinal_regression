import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
pd.set_option("display.max_columns", 100)


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


def prep_data(pca=False, pca_scale=False, inputation=False,
              strategy='median', remove_low_variance=False,
              variacne_thresh=0.0001):
    train = pd.read_csv("../1RawData/train.csv")
    test = pd.read_csv("../1TestData/test.csv")
    # combine train and test
    all_data = train.append(test)

    # set the right data_type
    description = all_data.describe(include='all').append(
                    [all_data.isnull().sum().rename('null_vals'),
                     all_data.dtypes.rename('data_types')])
    categorical_des = description.loc[:, all_data.dtypes == 'int64']
    categorical_variables = list(categorical_des.columns)
    all_data.loc[:, categorical_variables] =\
        all_data.loc[:, categorical_variables].astype(object)
#    all_data.dtypes.value_counts()

    # create any new variables
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] =\
        pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] =\
        pd.factorize(all_data['Product_Info_2_num'])[0]
    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
    med_keyword_columns =\
        all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)
    all_data['Med_Keywords_avg'] = all_data[med_keyword_columns].mean(axis=1)
    all_data['count_na'] = all_data.isnull().sum(axis=1)

    # fix the dtype on the label column
    all_data['Response'].fillna(-1, inplace=True)
    all_data['Response'] = all_data['Response'].astype(int)

    # deal with cat and numeric variables
    numeric_des = description.loc[:, (all_data.dtypes == 'float64') |
                                     (all_data.dtypes == 'int64')]
    numeric_variables = list(numeric_des.columns)
#    for item in categorical_variables:
#        print(f"{item}: {len(all_data[item].value_counts())}")

    # from numeric description
#    num_suspicious_list = [ 'Medical_History_1', 'Medical_History_10',
#                           'Medical_History_15', 'Medical_History_24',
#                           'Medical_History_32',]
    cat_suspicious_list = ['Medical_History_2']
#    for item in num_suspicious_list:
#            print(f"{item}: {all_data[item].value_counts()}")
#    categorical_des[cat_suspicious_list]

    # variable type adjustment
    categorical_variables = list(set(categorical_variables) -
                                 set(['Medical_History_2', 'Id']))
    numeric_variables.append(cat_suspicious_list[0])
    
    # seperate numeric and categorical data and handle them seperatly
    all_cat_data = all_data.loc[:, categorical_variables]
    all_numeric_data = all_data.loc[:, numeric_variables]

    # Inspect missing values for numeric variables
    # There is no missing values in categorical variables
#    na_col = all_numeric_data.columns[all_numeric_data.isnull().sum() > 0]
#    na_des = description[na_col]
    
    if inputation:
        inputer = SimpleImputer(strategy=strategy)
        all_numeric_data[:] = inputer.fit_transform(all_numeric_data)
    else:
        # type one variable has min max 0 and 1 
        type_1_variable = ['Employment_Info_1', 'Employment_Info_4',
                           'Employment_Info_6', 'Family_Hist_2', 'Family_Hist_3',
                           'Family_Hist_4', 'Family_Hist_5', 'Insurance_History_5']
        # type two variable has min max 0 and 240
        type_2_variable = ['Medical_History_1', 'Medical_History_10',
                           'Medical_History_15', 'Medical_History_24',
                           'Medical_History_32']
        type_1_inpute = -1
        type_2_inpute = -240

        # we cant use df.fillna(replace=True) as all_numeric_data is a chain
        # assignment
        all_numeric_data[type_1_variable] = all_numeric_data[
                                         type_1_variable].fillna(type_1_inpute)
        all_numeric_data[type_2_variable] = all_numeric_data[
                                         type_2_variable].fillna(type_2_inpute)

    # one hot encode cat variables
    all_cat_data = pd.get_dummies(all_cat_data, columns=categorical_variables)
#    all_cat_data.shape
#    all_cat_data.dtypes
    # pca cat variables
    if pca:
        pca = PCA().fit(all_cat_data)
        explain = np.cumsum(pca.explained_variance_ratio_)
        pca_dimenssion_k = np.where(explain > 0.99)[0][0] + 1
        pca_k = PCA(n_components=pca_dimenssion_k)
        cat_pca = pca_k.fit_transform(all_cat_data)
        cat_pca = pd.DataFrame(cat_pca)
#        cat_pca.describe()

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_numeric_data[:] = scaler.fit_transform(all_numeric_data)
#    all_numeric_data.isnull().sum()
    if pca_scale:
        cat_pca = scaler.fit_transform(cat_pca)
        
    # concat all data
    if pca:
        cat_pca.index = all_numeric_data.index
        all_data_transformed = pd.concat([cat_pca, all_numeric_data,
                                          all_data.Response], axis=1)
    else:
        all_data_transformed = pd.concat([all_cat_data, all_numeric_data,
                                          all_data.Response], axis=1)
    if not pca and remove_low_variance:
        variabel_variance = all_data_transformed.var(axis=0)
        low_var_col = list(variabel_variance[
                        variabel_variance < variacne_thresh].index)
        all_data_transformed = all_data_transformed.drop(low_var_col, axis=1)

    # split train and test
    train = all_data_transformed[all_data_transformed['Response'] > 0].copy()
    test = all_data_transformed[all_data_transformed['Response'] < 1].copy()
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
