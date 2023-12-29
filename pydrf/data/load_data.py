from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
import time
from os.path import dirname, join

def load_abalone(preprocess=False):
    '''
    Load abalone data.
    
    Parameters
    ----------
    preprocess: bool, optional (default: False)
        Perform data preprocessing.

    Returns
    -------
    (X, y): tuple
    '''
    module_path = dirname(__file__)
    data_path = join(module_path, 'abalone_data','data_abalone.csv')
    abalone = pd.read_csv(data_path)
    X = abalone.drop(' Rings', axis=1)
    y = abalone[' Rings']
    if preprocess:
        X['Sex'] = pd.Categorical(X['Sex'])
        return (X, y)
    else:
        return (X, y)

def load_adult(preprocess=False):
    module_path = dirname(__file__)
    data_path = join(module_path, 'adult_data','adult.data')
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    adult = pd.read_csv(data_path, header=None, names=names)
    X = adult.drop('income', axis = 1)
    y = adult['income']
    if preprocess:
        adult[adult==" ?"]=np.nan
        adult = adult.dropna()
        X = adult.drop('income', axis = 1)
        y = adult['income']
        categories = X.columns[X.dtypes=='object']
        for i in categories:
            X[i] = pd.Categorical(X[i])
    return (X, y)
    
def load_carEvaluation(preprocess=False):
    '''
    Load car evaluation data.
    
    Parameters
    ----------
    preprocess: bool, optional (default: False)
        Perform data preprocessing.

    Returns
    -------
    (X, y): tuple
    '''
    module_path = dirname(__file__)
    data_path = join(module_path, 'car_evaluation_data','car.data')
    car_eval = pd.read_csv(data_path, names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability'])
    X = car_eval.drop('acceptability', axis=1)
    y = car_eval['acceptability']
    if preprocess:
        X.loc[X['doors']=="5more", 'doors']=5
        X.loc[X['persons']=="more", 'persons']=6
        
        X['buying'] = pd.Categorical(X['buying'])
        X['maint'] = pd.Categorical(X['maint'])
        X['doors'] = pd.to_numeric(X['doors'])
        X['persons'] = pd.to_numeric(X['persons'])
        X['lug_boot'] = pd.Categorical(X['lug_boot'])
        X['safety'] = pd.Categorical(X['safety'])
    return (X, y)

def load_diamonds(preprocess=False):
    '''
    Load diamonds data.
    
    Parameters
    ----------
    preprocess: bool, optional (default: False)
        Perform data preprocessing.
    
    Returns
    -------
    (X, y): tuple
    '''
    module_path = dirname(__file__)
    data_path = join(module_path, 'diamonds_data', 'diamonds.csv')
    diamonds = pd.read_csv(data_path, index_col=0)
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    if preprocess:
        categories = X.columns[X.dtypes=='object']
        for i in categories:
            X[i] = pd.Categorical(X[i])
    return (X, y)

def load_lendingClub(preprocess=False, balanced=False):
    '''
    Load lending club data.
    
    Parameters
    ----------
    preprocess: bool, optional (default: False)
        Perform data preprocessing.
        
    balanced: bool, optional (default: False)
        Resample to balanced the outcome.

    Returns
    -------
    (X, y): tuple
    '''
    module_path = dirname(__file__)
    data_path = join(module_path, 'loan_data', 'data_LoanStats3a.csv')
    loan = pd.read_csv(data_path, nrows = 39786, header = 1)
    if preprocess:
        vars_to_keep = ['loan_amnt','term','int_rate','installment','grade',
        'emp_length','home_ownership','annual_inc','verification_status',
        'purpose','dti', "total_acc", 'loan_status']
        loan_clean = loan.loc[:,vars_to_keep]
        loan_clean['loan_status'] = pd.Categorical(loan_clean['loan_status'])
        loan_clean['term'] = pd.Categorical(loan_clean['term'])
        loan_clean['int_rate'] = pd.to_numeric(loan_clean['int_rate'].str.replace("%",""))/100
        loan_clean['grade'] = pd.Categorical(loan_clean['grade'])
        loan_clean['emp_length'] = loan_clean['emp_length'].replace(np.nan, 'Na', regex=True)
        loan_clean['emp_length'] = pd.Categorical(loan_clean['emp_length'], categories=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years","6 years", "7 years", "8 years", "9 years", "10+ years","Na"], ordered=True)
        loan_clean.loc[loan_clean['home_ownership']=="NONE", 'home_ownership']="OTHER"
        loan_clean['home_ownership'] = pd.Categorical(loan_clean['home_ownership'])
        loan_clean = loan_clean.loc[loan_clean['annual_inc']< 5000000]# Remove outlier
        loan_clean['verification_status'] = pd.Categorical(loan_clean['verification_status'])
        loan_clean['purpose'] = pd.Categorical(loan_clean['purpose'], categories=["debt_consolidation", "car", "credit_card", "educational", "home_improvement", "house", "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"])
        
        if(balanced):
            loanStat_balance = pd.concat([loan_clean[loan_clean['loan_status']=='Fully Paid'].sample(replace=False, n = 5670), loan_clean[loan_clean['loan_status']=='Charged Off']])
            
#         onehot = OneHotEncoder(categorical_features = [1,4,5,6,8,9])
#         onehot = OneHotEncoder(categories="auto")
#         enc = pd.DataFrame(onehot.fit_transform(loanStat_balance.iloc[:,:-1], loanStat_balance['loan_status']).toarray())
        
        #result = pd.concat([enc, loanStat_balance['loan_status']], axis=1)
            return (loanStat_balance.drop("loan_status", axis=1), loanStat_balance['loan_status'])
        else:
            return (loan_clean.drop("loan_status", axis=1), loan_clean['loan_status'])
    return (loan.drop("loan_status", axis=1), loan['loan_status'])


def load_poker(preprocess=False):
    '''
    Load poker data.
    
    Parameters
    ----------
    preprocess: bool, optional (default: False)

    Returns
    -------
    (X_train, X_test, y_train, y_test): tuple
    '''
    module_path = dirname(__file__)
    train_data_path = join(module_path, 'poker_data', 'poker-hand-training-true.data')
    test_data_path = join(module_path, 'poker_data', 'poker-hand-testing.data')
    train = pd.read_csv(train_data_path, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])
    test = pd.read_csv(test_data_path, names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])
    X_train, y_train = (train.iloc[:,0:10], train.iloc[:,10])
    X_test, y_test= (test.iloc[:, 0:10], test.iloc[:, 10])
    if preprocess:
        onehot = OneHotEncoder(categories='auto')
        enc_train = pd.DataFrame(onehot.fit_transform(X_train, y_train).toarray())
        enc_test = pd.DataFrame(onehot.transform(X_test).toarray())
        return (enc_train, enc_test, y_train, y_test)
    else:
        return (X_train, X_test, y_train, y_test)
