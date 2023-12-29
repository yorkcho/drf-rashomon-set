import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from pydrf.order import CategoryOrderEncoder

# classification
class DRFModel(BaseEstimator):
    '''
    A Deep Rule Forest Model.
    
    Parameters
    ----------
    forest_type: 'randomForest'
        There is only 'randomForest' option now.
        
    parameter_list: list, required
        The parameter for random forest in each layer. 
        If the 'forest_type' is "randomForest", the paramters in the dictionary should be of the scikit-learn random forests.
        Example:
            parameter_list = [[{"max_leaf_nodes":8, "n_estimators":100, "max_features":0.8},{"max_leaf_nodes":4, "n_estimators":100, "max_features":0.8}],[{"max_leaf_nodes":4, "n_estimators":100, "max_features":0.8}]]
          
    outcome_type: 'classification' or 'regression', default = 'classification'
        Choose classification or regression to decide the base estimator. If the forest_type is 'randomForest', the RandomForestClassifier and RandomForestRegressor from scikit-learn package will be used. 
        - 'classification': RandomForestClassifier
        - 'regression': RandomForestRegressor
     
    handle_leaf_nodes: 'reorder','one-hot' or None , default = 'reorder'
        The way how do the leaf nodes be handled before passed to the next layer as training data.
        - 'reorder': The leaf nodes will be reordered before being the training data of the next layer.
        - 'one-hot': The leaf nodes will be encoded with one-hot encoding before being the training data of the next layer.
        - None: The leaf nodes will be used as training data directly after the leaf encoding has been performed.
        
    parallol_cores: int, default = 1
        The number of core used to perform multiprocessing.
        
    Attributes
    ----------
    forest_type : str
    
    model_ : list of list of DecisionTreeClassifier or DecisionTreeRegressor
        The DRF model.
        
    class_ : array of int, float or string
        The categories of training data.
        
    encoders_ = list of OneHotEncoders or CategoryOrderEncoders
        The encoders used to handle leaf nodes.
        
    type_ : str
        The type of predicting outcome.
        
    parallel_cores : int
        The number of cores used when ``fit`` is performed.
    
    parameter_list : list of dict
        The parameters used when ``fit`` is performed.
    
    handle_leaf_nodes : str
        The way how do the leaf nodes be handled before passed to the next layer as training data.
    '''
    def __init__(self, forest_type="RandomForest", outcome_type="classification", 
                 parameter_list=None, handle_leaf_nodes='reorder',
                 parallel_cores=1, cascade=False):
        self.forest_type = "RandomForest"
        self.model_ = None
        self.class_ = None
        self.encoders_ = list()
        self.type_ = outcome_type #classification or regression
        self.parallel_cores = parallel_cores
        self.parameter_list = parameter_list
        self.handle_leaf_nodes = handle_leaf_nodes
        self.cascade = cascade
        
        
    def fit(self, X_train, y_train):
        '''
        Build a deep rule forest model for training set (X_train, y_train).
        
        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features], required
            The training data.

        y_train: array-like, shape = [n_samples]
            The target value (class for classification or real number for regression).
             
        Returns
        -------
        self: object
        '''
        if self.model_ is not None:
            self.model_ = None
            self.class_ = None
            self.encoders_ = list()
        

        # The function for fitting a model
        if self.type_ not in ('classification', 'regression'):
            raise ValueError("outcome_type should be either 'classification' or 'regression', but got '{}'".format(self.type_))
        
        if self.handle_leaf_nodes not in ('reorder', 'one-hot', None):
            raise ValueError("handle_leaf_nodes should be 'reorder', 'one-hot' or None, but got '{}'".format(self.handle_leaf_nodes))
        
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)

        if isinstance(self.parameter_list, (list, tuple)):
            if self.type_ == 'classification':
                # Keep the classes in y_train
                self.class_ = np.unique(y_train)
                # Get the parameter for every layer
                # = map(the parameter in every layer <- map (the parameter for every RandomFoerstClassifier))
                forest_model = [i for i in map(lambda layer: 
                                    [j for j in map(lambda param: RandomForestClassifier(**param, n_jobs=self.parallel_cores),layer)
                                    ], self.parameter_list)
                               ]
            else:
                forest_model = [i for i in map(lambda layer: 
                                    [j for j in map(lambda param: RandomForestRegressor(**param, n_jobs=self.parallel_cores),layer)
                                    ], self.parameter_list)
                               ]

            # Initialize the list and layer counter for DRF model
            drf_model = list()
            layer_count = 0

            for layer in forest_model:
                print("Layer %d training..."% layer_count, end="")
                drf_layer = []
                leaf_enc = []
                # There are many RandomForestClassifiers in one layer.
                # Extend those RandomForestClassifiers to Decision Trees and assemble them to a layer of the DRF
                for model in layer:
                    model.fit(X_train, y_train)
                    drf_layer.extend(model.estimators_) # Append Decision Trees to the randomforest to the specific layer of the DRF
                    
                    # If training has finished, break. Otherwise, do leaf encoding to generlize training data for the next layer
                    if layer_count == len(forest_model):
                        break

                    # Do leaf encoding with the RandomForests of one layer
                    
                    # If there is no encoded data in leaf_enc
                    if leaf_enc == []: 
                        leaf_enc = np.array(model.apply(X_train))
                    else:
                        # Append the encoded data of the random forest to leaf_enc 
                        leaf_enc = np.append(leaf_enc, model.apply(X_train), axis=1)
                        
                # Use leaf_enc as the training data of the random forest of the next layer
                # If handle_leaf_nodes == "reorder", order the leaf node with CategoryOrderEncoder
                if self.handle_leaf_nodes == "reorder":
                    encoder = CategoryOrderEncoder(n_jobs=self.parallel_cores)
                    encoder.fit(leaf_enc, y_train, categorical_features = range(0, leaf_enc.shape[1]))
                    # Get the training data for the next layer
                    X_train = encoder.transform(leaf_enc)
                    # Keep the encoder for predicting
                    self.encoders_.append(encoder)
                    
                # If handle_leaf_nodes == "one-hot", performing one-hot encoding 
                elif self.handle_leaf_nodes == "one-hot":
                    # Do one-hot encoding, and get the training data for the next layer
                    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
                    X_train = onehot_encoder.fit_transform(leaf_enc)
                    # Keep the encoders for predicting
                    self.encoders_.append(onehot_encoder)
                    
                # If handle_leaf_nodes == "None"
                else:
                    X_train = leaf_enc
                
                # Appending the layer of random forests to the drf_model
                drf_model.append(drf_layer)
                layer_count+=1
                print("finished.")
            self.model_ = drf_model
            return(self)
        

            
    def predict(self, X_test, predict_layer=None, n_jobs=1):
        '''
        Predict class for X_test.
        
        Parameters
        ----------
        X_test: array-like, shape = [n_samples, n_features], required
             The input samples.

        predict_layer: positive integer or 0, default = None
             The layer of random forest used to predict. The number specifies which layer in the DRF you want to use to predict the outcome. If it is None, the last layer will be used.
        
        Returns
        -------
        prediction: array of shape = [n_sample]
        '''
        if self.model_ is None:
            print("Please train the model first.")
        else:
            encode_test = np.array(X_test)
            # If the predict_layer is not specified, using the last layer to predict the outcome. 
            if predict_layer is None:
                predict_layer = len(self.model_)-1
            # If the predict_layer is not the 0th one, performing leaf encoding layer by layer
            
            if predict_layer >= 1:
                # Using the layer 0 to the layer in front of the predict_layer to perform leaf encoding 
                # Getting the encoded outcome from all trees and merge them
                encode_test = self.transform_data(encode_test, predict_layer, n_jobs = n_jobs)

            elif predict_layer == 0:
                encode_test = X_test
            # Use the last layer to predict the outcome
            # Get the prediction from all trees in the last layer 
            trees_prediction = np.array(Parallel(n_jobs = n_jobs)(delayed(tree.predict)(encode_test)for tree in self.model_[predict_layer]))
            
            if self.type_ == 'classification':
                # Find out the mode as the prediction of DRFclassifier
                forest_prediction = stats.mode(trees_prediction)[0].reshape(trees_prediction.shape[1],)
                # Transform the int prediction into class name
                trans = dict(zip(np.unique(forest_prediction), self.class_))
                prediction = np.vectorize(trans.get)(forest_prediction)
            else:
                # Compute the mean of prediction of all trees as the prediction of DRFregressor
                prediction = np.mean(trees_prediction, axis=0)
            
            return prediction


    def transform_data(self, X, stop_layer, n_jobs = 1):
        '''
        Transform data X with DRF model.
        
        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features], required
             The input samples.

        stop_layer: positive integer, required
             The last layer of random forest used to perform leaf encoding. The leaf encoding will be performed from the first layer to stop_layer. Be careful that the stop_layer is not included.
        
        Returns
        -------
        X: array of shape = [ n_samples, len(model[stop_layer]) ]
        '''
        if self.model_ is None:
            print("Please train the model first.")
        else:
            print("Transforming data...", end="")
            for layer_id in range(stop_layer):
                X = np.array(Parallel(n_jobs = n_jobs)(delayed(tree.apply)(X)for tree in self.model_[layer_id])).T
                if self.handle_leaf_nodes == "reorder":
                    X = self.encoders_[layer_id].transform_columns(X, n_jobs = n_jobs)
                elif self.handle_leaf_nodes == "one-hot":
                    X = self.encoders_[layer_id].transform(X)
                    
            print("finished.")
            return X




    def extract_rule(self, n_layer, n_tree, n_rule):
        '''
        Get a rule from the specific layer and tree.
        
        Parameters
        ----------
        n_layer : integer, required
            The layer index for extracting rule. It should be smaller than the number of layers of DRF.
             
        n_tree : integer, required
            The tree index for extracting rule. It should be smaller than the number of the tree in the random forest.
        
        n_rule : integer, required
            The rule index for extracting rule.

        Returns
        -------
        rule[is_leaf][n_rule]: string
        '''
        # Extracting rules from the tree of the specific layer
        if self.model_ == None:
            print("Please train the model first.")
        else:
            # Getting the information of the tree
            tree = self.model_[n_layer][n_tree].tree_
            feature = tree.feature
            threshold = tree.threshold
            n_nodes = tree.node_count
            children_left = tree.children_left
            children_right = tree.children_right
            # Finding out all leave nodes
            # is_leave: left child and right child are both <0    
            is_leaf = np.logical_and(children_left<0, children_right<0)
            sum_leaf = is_leaf.sum()
            print("num of rules: ", sum_leaf)
            
            node_id = 0
            rule_count = 0
            temp_rule = "X[,%d]"%(feature[node_id])# the feature for the root node
            rule = np.zeros(shape=n_nodes, dtype="U500") # the rule string array for each node
            que = [] 
            
            # Traversal the tree
            while(rule_count < sum_leaf):
                left = children_left[node_id]
                right = children_right[node_id]
                # If the node has left or right child node, adding the threshold and put the temp rule into the rule[node] array.
                if(left > 0):
                    temp_left = temp_rule + "<=%.3f"%(threshold[node_id])
                    rule[left] = temp_left
                    que.append(left)
                if(right > 0):
                    temp_right = temp_rule + ">%.3f"%(threshold[node_id])
                    rule[right] = temp_right
                    que.append(right)
                # If the node has no child node, it is a leaf node.
                if(left < 0 and right < 0):
                    rule[node_id] = temp_rule
                    rule_count += 1
                
                if(rule_count >= sum_leaf):
                    break
                node_id = que.pop(0)
                if(feature[node_id] >= 0):
                    temp_rule = rule[node_id] + " & X[,%d]"%feature[node_id]
                else:
                    temp_rule = rule[node_id]
            return rule[is_leaf][n_rule]# get the leaf nodes and the rule
