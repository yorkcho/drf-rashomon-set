import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from collections.abc import Iterable


class CategoryOrderEncoder(BaseEstimator):
    '''
    An encoder that encodes categorical variables into oredered integers. The order depends on the PCA scores of the covariance matrix of the contingency table between the predictor and the outcome.
    '''
    def __init__(self, n_jobs = 1):
        self.map_dict = None
        self.categorical_features = None
        self.n_jobs = n_jobs
        self.inv_dict = None
        
    def fit(self, X, y, categorical_features):
        '''
        Fit an encoder.
        
        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features], required
            The training data.

        y: array-like, shape = [n_samples], required
            The target value (class for classification or real number for regression).
            
        categorical_features: a list of column names.
            It indicates columns which should be transformed into integers.

        Returns
        -------
        self: object
        '''
        if isinstance(categorical_features, Iterable):
            self.categorical_features = categorical_features
            y = np.array(y)
            # convert data into ndarray
            if isinstance(X, pd.DataFrame):
                # Get the columns names
                column_names = X.columns
                column_index = X.columns.isin(categorical_features)

                X = np.array(X)
                column_index = np.array(range(0, X.shape[1]))[column_index]
            else:
                column_names = np.array(range(0, X.shape[1]))
                column_index = categorical_features
            # If outcome is numeric
            if np.issubdtype(y.dtype, np.number):
                map_dict = self._reorder_by_mean(X, y, column_index, column_names)
            elif len(np.unique(y)==2):
                map_dict = self._reorder_by_freq(X, y, column_index, column_names)
            # If outcome is multi-class
            else:
                map_dict = self._reorder_by_PC(X, y, column_index, column_names)
            self.map_dict = map_dict
            return self
        
        else:
            raise ValueError("categorical_features should be Iterable object, got {}".format(type(categorical_features)))

    def _level_mean(self, l, x, y):
        return y[x==l].mean()
            
    def _reorder_by_mean(self, X, y, column_index, column_names):
        '''
        Reorder for regression outcome.
        '''
        map_dict = dict()
        # For each column of categorical variable
        for var in column_index:
            x = X[:, var].reshape((len(y),))
            # Get levels
            levels = np.unique(x)
            mean_list = list()
            
            # For each levels, compute the mean of y
            mean_list = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._level_mean)(level, x, y) for level in levels))
            mean_df = pd.DataFrame({'mean': mean_list}, index=levels)
            
            # Sort the levels by mean
            mean_order = mean_df.sort_values(ascending=True, by='mean')
            cat_order = pd.Series(range(1, len(levels)+1), index=mean_order.index)
            # Build the dictionary and map the categories to integer
            order_dict = dict(cat_order)
            map_dict[column_names[var]] = order_dict
        return map_dict

    def _level_freq(self, l, x, y):
        uni, counts = np.unique(y[x==l], return_counts=True)
        return counts[0]
    
    def _reorder_by_freq(self, X, y, column_index, column_names):
        '''
        Reorder for binary class outcome.
        '''
        map_dict = dict()
        # For each column of categorical variable
        for var in column_index:
            x = X[:, var].reshape((len(y),))
            # Get levels
            levels = np.unique(x)
            freq_list = list()
            
            # For each levels, compute the frequency of y == class0
            freq_list = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._level_freq)(level, x, y) for level in levels))
            freq_df = pd.DataFrame({'freq': freq_list}, index=levels)
            
            # Sort the levels by freq
            freq_order = freq_df.sort_values(ascending=True, by='freq')
            cat_order = pd.Series(range(1, len(levels)+1), index=freq_order.index)
            # Build the dictionary and map the categories to integer
            order_dict = dict(cat_order)
            map_dict[column_names[var]] = order_dict
        return map_dict

    def _reorder_by_PC(self, X, y, column_index, column_names):
        '''
        Reorder for multi-class outcome.
        '''
        map_dict = dict()
        for var in column_index:
            # Step1: Compute the contingency table N between the predictor x and the outcome y
            tab = pd.crosstab(index=X[:, var], columns=y)
            ncat = tab.sum(axis='columns')
            
            # Step2: Convert the contingency table N between the class probability matrix P
            # probability matrix P
            ProbMatrix = tab.divide(ncat, axis='rows')

            # Step3: Compute the weighted covariance matrix
            # p bar
            MeanClassProb = ProbMatrix.mean()
            # weight
#                 W = ncat/(n-1)
            Diff = ProbMatrix-MeanClassProb
            # weighted covariance matrix
            WeighCov = np.cov(Diff.T, ddof = 0, fweights = ncat)

            # Step4: Compute the first principal component of WeighCov and principal component scores
            PC = pca.PCA()
            PC.fit(WeighCov)
            # the first principal component
            PC1 = PC.components_[0]
            # Principal component scores
            PCScore = (ProbMatrix*PC1).sum(1)
            
            # Order by PCScore
            score_order = PCScore.sort_values(ascending=False)
            cat_order = pd.Series(range(1, len(score_order)+1), index=score_order.index)

            # Build the dictionary and map the categories to integer
            order_dict = dict(cat_order)
            map_dict[column_names[var]] = order_dict
        return map_dict



    def transform(self, X, handle_unknown='error'):
        '''
        Transform the categories into the corresponding ordered integers.
        
        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features], required
            The data to be transformed.
            
        handle_unknown: 'error','ignore' or 'na', default = 'error'
            - 'error': Transforming will be stopped and show the error message.
            - 'ignore': The unknown values will be transformed into integers.
            - 'na': The unknown values will be transformed into NA.
        
        Returns
        -------
        enc_X: array or dataframe of shape = [n_samples, n_features]

        '''
        # Check the parameter handle_unknown
        if handle_unknown not in ['error', 'ignore', 'na']:
            raise ValueError("handle_unknown should be 'error', 'ignore' or 'na', got {}".format(handle_unknown))

        # Check the type of the parameter X
        if isinstance(X, np.ndarray):
            enc_X = self._transform_ndarray(X, handle_unknown)
        else:
            enc_X = self._transform_df(X, handle_unknown)
            
        return enc_X

    
    def _transform_ndarray(self, X, handle_unknown):
        enc_X = X.copy()
        for var in self.categorical_features:
            # Get absent levels
            absent = set(X[:, var]).difference(set(self.map_dict[var]))
            # Handle the absent level
            if absent:
                # If handle_unknown == 'ignore', assign new integers to the unknown values
                if handle_unknown == 'ignore':
                    print("The nominal variable \"{}\" has the absent categories {}".format(var, absent))
                    # Assign new integer numbers to the absent levels
                    self._update_mapdict(var, absent)

                # If handle_unknown == 'error', raise error
                elif handle_unknown == 'error':
                    raise ValueError("Unknown categorical feature present {}".format(absent))
                    
            # Map the categories to the integers
            enc_X[:, var] = np.vectorize(self.map_dict[var].get)(X[:,var])
        return enc_X

    
    def _transform_df(self, X, handle_unknown):
        enc_X = X.copy()
        for var in self.categorical_features:
            # Get absent levels
            absent = set(X[var]).difference(set(self.map_dict[var]))
            # Handle the absent levels
            if absent:
                # If handle_unknown == 'ignore', assign new integers to the unknown values
                if handle_unknown == 'ignore':
                    print("The nominal variable \"{}\" has the absent categories {}".format(var, absent))
                    # Assign new integer numbers to the absent levels
                    self._update_mapdict(var, absent)

                # If handle_unknown == 'error', raise error
                elif handle_unknown == 'error':
                    raise ValueError("Unknown categorical feature present {}".format(absent))
                    
            # Map the categories to the integers
            enc_X[var] = X[var].map(self.map_dict[var])
        return enc_X


    def _handle_absent_level(self, X, var, absent, handle_unknown):
        '''
        ignore or raise error according to handle_unknown
        '''
        # Get absent levels
        absent = set(X[var]).difference(set(self.map_dict[var]))
        # Handle the absent levels
        if absent:
            # If handle_unknown == 'ignore', assign new integers to the unknown values
            if handle_unknown == 'ignore':
                print("The nominal variable \"{}\" has the absent categories {}".format(var, absent))
                # Assign new integer numbers to the absent levels
                self._update_mapdict(var, absent)

            # If handle_unknown == 'error', raise error
            elif handle_unknown == 'error':
                raise ValueError("Unknown categorical feature present {}".format(absent))

        
    def _update_mapdict(self, var, absent):
        # Map the unknown categories to integers
        absent_levels = pd.Series(range(len(self.map_dict[var])+1, len(self.map_dict[var])+1+len(absent)), index=absent)
        # Update the map_dict
        self.map_dict[var].update(dict(absent_levels))
    
    
    def _transform_column(self, x, var):
        '''
        Transform a column of array.
        '''
        # Get absent levels
        absent = set(x).difference(set(self.map_dict[var]))
        # Handle the absent levels
        if absent:
            self._update_mapdict(var, absent)
        # Transform to new order
        x = np.vectorize(self.map_dict[var].get)(x)
        return x


    def transform_columns(self, X, n_jobs):
        '''
        Transform the categories into the corresponding ordered integers with mutiple core. The type of X should be np.ndarray.
        '''
        TX = X.T
        enc_X = np.array(Parallel(n_jobs=n_jobs)(delayed(self._transform_column)(x, col) for x, col in zip(TX, self.categorical_features))).T
        return enc_X

    
    def inverse_transform(self, X):
        if self.inv_dict is None:
            inv_dict = dict()
            for key in self.map_dict.keys():
                d = self.map_dict[key]
                inv = {v:k for (k,v) in d.items()}
                inv_dict[key] = inv
            self.inv_dict = inv_dict
        x = X.copy()
        if isinstance(X, np.ndarray):
            for col in self.categorical_features:
                x[:, col] = np.vectorize(self.inv_dict[col].get)(x[:, col])
        else:
            for col in self.categorical_features:
                x[col] = np.vectorize(self.inv_dict[col].get)(x[col])
        return(x)
        
