# %% 导入包
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import random
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor

from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import shap
from pdpbox import pdp, get_dataset,info_plots

# %% 以给定格式导入数据
dtypes = [
    ("Size", "float32"), ("Zeta potential", "float32"), 
    ("Hydrodynamic size", "float32"), ("Concentration", "float32"),
    ("Solution pH", "float32"), ("Solution ionic strength", "float32"), 
    ("Seed thickness", "category"),("Seed width", "category"), 
    ("Zn content", "float32")
]

data = pd.read_excel(
    "Dataset_Nano_SeedSoaking.xlsx",
    names = [d[0] for d in dtypes], 
    index_col = None,
    dtype = dict(dtypes)
)
data

# %% 对类别变量进行label编码，因其具有内在顺序关系
data_label = data.copy()
le = LabelEncoder()
le.fit(['flat', 'thick'])
data_label['Seed thickness'] = le.transform(data_label['Seed thickness'])
print(list(le.inverse_transform([0,1])))
le = LabelEncoder()
le.fit(['narrow', 'wide'])
data_label['Seed width'] = le.transform(data_label['Seed width'])
print(list(le.inverse_transform([0,1])))
print(data_label.describe())
data_label

# %% 查看变量间的相关性
data_corr = data_label.copy()
data_corr = data_corr.drop(['Seed thickness','Seed width'], axis=1)
corr = data_corr.corr()

fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')

h=sns.heatmap(corr, cmap='Blues',  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8})

cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12)

fig.savefig("./Image/data_corr.jpg",dpi=600,bbox_inches='tight')

# %% 将顺序打乱，并分配X和Y
data_sample = data_label.sample(frac=1, replace=False, random_state=1)
X = data_sample.drop(['Zn content'], axis=1)
Y = data_sample['Zn content']
X

# %% 80%的数据用来做5折交叉验证，20%的数据用来做测试集
X_cv, X_test, Y_cv, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_cv.head(5)

# %%
parameter = {

    # 1.避免过拟合
    "min_data_in_leaf":np.arange(1,20,1),
    "min_sum_hessian_in_leaf": np.arange(1,20,1),
    
    # 2.提高模型准确度
    #"max_bin":  np.arange(1,30,1),
    
    # 3.确定树的大小及复杂度
    #"max_depth":np.arange(1,11,1),
    #"num_leaves": np.arange(2,21,1),

    # 4.调整学习率
    #"learning_rate": np.arange(0.001,0.1,0.001),

}

model_gs = lgb.LGBMRegressor(n_jobs=-1,n_estimators=1000,
                             #min_data_in_leaf=6,min_sum_hessian_in_leaf=1,
                             #max_bin=14, 
                             #max_depth=4,num_leaves=5,
                             #learning_rate=0.090,
                            )

                
grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_cv, Y_cv,categorical_feature=['Seed thickness','Seed width'])

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

LGBM_Gs_best = grid_search.best_estimator_

print('RMSE:', abs(sum(cross_val_score(LGBM_Gs_best, X_cv, Y_cv, cv=5, scoring='neg_root_mean_squared_error'))/5))

# %%
model = lgb.LGBMRegressor(n_jobs=-1,n_estimators=1000,
                             min_data_in_leaf=6,min_sum_hessian_in_leaf=1,
                         )

# model = lgb.LGBMRegressor(n_jobs=-1,n_estimators=1000,max_cat_to_onehot=6,
#                              min_data_in_leaf=1,min_sum_hessian_in_leaf=17,
#                              max_bin=15, 
#                              max_depth=3,num_leaves=4,
#                              learning_rate=0.087,
#                          )
model.fit(X_cv, Y_cv,categorical_feature=['Seed thickness','Seed width'])
LightGBM_importance = model.feature_importances_


Y_pred = model.predict(X_test)
Y_cv_pred = model.predict(X_cv)

print('R2_train: %.2f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.2f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.2f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.2f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

# %% 模型性能
figure = plt.figure(figsize=(6,6))  # 设定画布大小
plt.style.use('classic')
plt.xlim(min(Y)-0.1, max(Y)+0.1)
plt.ylim(min(Y)-0.1, max(Y)+0.1)
plt.plot([min(Y)-0.1,max(Y)+0.1],[min(Y)-0.1,max(Y)+0.1],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="p",edgecolors="#9932CC",label="training set")   
plt.scatter(Y_test, Y_pred,marker="o",edgecolors="#FF8C00",label="test set")
plt.xlabel('Observation')
plt.ylabel('Prediction')
plt.legend(loc=0,fontsize=12)
plt.plot()
figure.savefig("./Image/Predcitons_observations_label.jpg",dpi=600,bbox_inches='tight')


# %% LightGBM特征重要性
figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = LightGBM_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

# 删除最重要的两个特征，以实现更好的可视化
importance_plot = list(LightGBM_importance[sorted_idx])
importance_del = [feature_this_plot.index('Concentration'), 
                feature_this_plot.index('Solution ionic strength')]
importance_plot = np.delete(importance_plot,importance_del)
feature_this_plot = np.delete(feature_this_plot,importance_del)

plt.barh(feature_this_plot,
         importance_plot, align='center', color="#1E90FF")

plt.title('LightGBM feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=18)

figure.savefig("./Image/LightGBM_importance.jpg",dpi=600,bbox_inches='tight')


# %%
result = permutation_importance(model, X_cv, Y_cv, scoring='r2', n_repeats=10, random_state=0, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,5))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = Permutation_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

# 删除最重要的两个特征，以实现更好的可视化
importance_plot = list(Permutation_importance[sorted_idx])
importance_del = [feature_this_plot.index('Concentration'), 
                feature_this_plot.index('Solution ionic strength')]
importance_plot = np.delete(importance_plot,importance_del)
feature_this_plot = np.delete(feature_this_plot,importance_del)

plt.barh(feature_this_plot,
         importance_plot, align='center', color="#1E90FF")

plt.title('Permutation feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)

figure.savefig("./Image/Permutation_importance.jpg",dpi=600,bbox_inches='tight')

# %%
explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_cv)
global_shap_values = np.abs(shap_values).mean(0)

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = global_shap_values.argsort()
sorted_features = X_cv.columns[sorted_idx]
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

# 删除最重要的两个特征，以实现更好的可视化
importance_plot = list(global_shap_values[sorted_idx])
importance_del = [feature_this_plot.index('Concentration'), 
                feature_this_plot.index('Solution ionic strength')]
importance_plot = np.delete(importance_plot,importance_del)
feature_this_plot = np.delete(feature_this_plot,importance_del)

plt.barh(feature_this_plot,
         importance_plot, align='center', color="#1E90FF")

plt.title('SHAP feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
figure.savefig("./Image/Shap_importance.jpg",dpi=600,bbox_inches='tight')

# %%
from sklearn.linear_model import LinearRegression
X_linear = X.loc[:,["Concentration","Solution ionic strength"]]
linear_model = LinearRegression()
score = cross_val_score(linear_model, X_linear, Y, cv = 5)
print('%.4f(%.4f)' %(score.mean(),score.std()))

X_linear = X.loc[:,["Concentration"]]
linear_model = LinearRegression()
score = cross_val_score(linear_model, X_linear, Y, cv = 5)
print('%.4f(%.4f)' %(score.mean(),score.std()))

X_linear = X
linear_model = LinearRegression()
score = cross_val_score(linear_model, X_linear, Y, cv = 5)
print('%.4f(%.4f)' %(score.mean(),score.std()))

# %%
