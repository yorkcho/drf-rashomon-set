# Deep Rule Forest(DRF)
## 安裝套件
下載檔案，開啟終端機確認移動到與setup.py同一層目錄下，並在終端機內輸入：
```
 python setup.py install
```
---
## 初始化模型
進行DRF模型的訓練前必須先利用`DRFModel`設定模型參數與架構。

### 參數
* `parallel_cores`:positive integer，訓練模型時使用的核心數量，預設值為1。
* `parameter_list`:list，訓練模型時必須設定參數`parameter_list`，內容為DRF內每一層隨機森林的參數，以最外層的`list`代表模型整體，第二層的`list`代表模型結構裡的每一層隨機森林，該`list`裡的每個`dictionary`代表該層隨機森林的組成成分，一個`dictionary`表示一群特定參數的隨機森林，可用參數請參考[scikit-learn之隨機森林](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)。

### 範例：
```python
# parameter_list參考格式
param_list = [
    [{"max_leaf_nodes":6, "n_estimators":30, "max_features":0.7},
    {"max_leaf_nodes":8, "n_estimators":50, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":30, "max_features":0.6}],
    [{"max_leaf_nodes":8, "n_estimators":40, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":40, "max_features":0.8}]
    ]

# 模型參數設定
drf = DRFModel(parallel_cores = 2, parameter_list = param_list)
```
---
## 訓練模型
提供`fit`方法進行DRF模型的訓練。

目前只提供scikit-learn的RandomForestClassifier作為組成DRF之隨機森林模型。
### 參數
* `X_train`:array-like，訓練資料的feature。
* `y_train`:array-like，訓練資料的label。

### 訓練模型範例
```python
# Import the package
from pydrf.model import DRFModel 

# Define the structure of DRF
param_list = [
    [{"max_leaf_nodes":16, "n_estimators":300, "max_features":0.8},
    {"max_leaf_nodes":8, "n_estimators":500, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":300, "max_features":0.8}],
    [{"max_leaf_nodes":8, "n_estimators":400, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":400, "max_features":0.8}]
    ]

# Fit the model
drf = DRFModel(parameter_list = param_list)
model = drf.fit(X_train, y_train)
```
---
## 預測結果
先使用`fit`訓練模型後，可以利用`predict`進行結果預測，`predict_layer`可以調整要用第幾層進行預測。

### 參數
* `X_test`:array-like，測試資料。
* `n_jobs`:positive integer，可以調整使用的核心數量，預設值為1。
* `predict_layer`:positive integer 或 0，預設值為None，使用指定隨機森林層進行預測，預設None表示最後一層。

### 預測結果範例
```python
# Use the data X_test to predict the outcome
outcome = model.predict(X_test, predict_layer = 1, n_jobs=1)
```
---
## 利用DRF進行資料轉換
提供`transform_data`可利用隨機森林將資料進行轉換，轉換後的資料可以再拿來訓練其他模型。轉換出來的資料欄位數應與該層的決策樹數量相同。

### 參數
* `data`:array-like，要進行轉換的資料。
* `n_jobs`:positive integer，可以調整使用的核心數量，預設值為1。
* `stop_layer`:positive integer 或 0，預設值為None，使用起始到指定的所有隨機森林層進行資料的轉換，預設None表示最後一層。

```python
transform = model.transform_data(data, stop_layer, n_jobs = 1)
```
### 轉換資料範例
指定使用model的第一層來轉換資料:
```python
transform_train = model.transform_data(train, stop_layer = 1)
transform_test = model.transform_data(test, stop_layer = 1)
```
---
## 萃取規則
提供`extract_rule`可萃取出指定層與決策樹之隨機森林的規則。

### 參數
* `n_layer`:integer，要萃取規則的層數位置。
* `n_tree`:integer，要萃取規則的樹位置。
* `n_rule`:integer，要萃取的規則。

```python
# specify the layer, tree and rule
model.extract_rule(n_layer, n_tree, n_rule)
```
### 萃取規則範例
取出第0層、第5棵樹的第2條規則:
```python
rule = model.extract_rule( 0, 5, 2)
```
---
## Iris資料集範例
```python
# Import the class
from pydrf.model import DRFModel 

# Loading data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data = load_iris()

# Splitting data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], train_size = 0.7, test_size = 0.3, random_state = 1)

# the parameter passed to every layer
parameter_list = [
    [{"max_leaf_nodes":6, "n_estimators":30, "max_features":0.7},
    {"max_leaf_nodes":8, "n_estimators":50, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":30, "max_features":0.6}],
    [{"max_leaf_nodes":8, "n_estimators":40, "max_features":0.8},
    {"max_leaf_nodes":4, "n_estimators":40, "max_features":0.8}]
    ]

drf = DRFModel(parameter_list = parameter_list)
# Fitting the DRF model
model = drf.fit(X_train, y_train)

# Predicting the outcome
outcome = model.predict(X_test)

# Extracting the rule
rule = model.extract_rule(0,2,0)
```


## CategoryOrderEncoder
將類別型的變數轉換為有序的數值變數，經過重新排序讓決策樹能在相對不影響準確度的條件下，以數值方式處理類別型變數。演算法細節參考[Wright and König (2019)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6368971/)
* `fit(X, y, categorical_features)`:訓練編碼器
* `transform(X, handle_unknown)`:利用訓練好的編碼器轉換資料
* `transform_layerwise(X, n_jobs)`:支援多核心的方式轉換資料，但X必須是np.ndarray型態。
---
### 訓練
`fit(X, y, categorical_features)`

用來訓練編碼器

參數:
* `X`:array-like，訓練編碼器的資料
* `y`:array-like，訓練資料的label
* `categorical_features`: iterable，欲轉換的column名稱。若訓練資料為np.ndarray型態，可以使用整數index如：\[0,2,3\]
---

### 轉換
`transform(X, handle_unknown='error')`

利用訓練好的編碼器轉換資料

參數:
* `X`:array-like，欲轉換的資料。
* `handle_unknown`:'error','ignore'或'na'，處理訓練資料內沒有的類別的方法。'error'會產生錯誤而停止；'ignore'會將新出現的類別依序往後編排，並更新map_dict；'na'會將這些類別以NA值處理。
---

`transform_layerwise(X, n_jobs)`

支援多核心的方式轉換資料，但X必須是np.ndarray型態。

參數:
* `X`:array-like，欲轉換的資料。
* `n_jobs`:positive integer，使用的核心數。
---

### 範例
```python
# Import
from pydrf.order import CategoryOrderEncoder
# Load data
X, y = load_lendingClub(preprocess=True, balance=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Fit and transform
encoder = CategoryOrderEncoder()
categories =  X.columns[X.dtypes=="category"]
cat_order.fit(X_train, y_train, categories)
ord_X_train = encoder.transform(X_train)
ord_X_test = encoder.transform(X_test)
```