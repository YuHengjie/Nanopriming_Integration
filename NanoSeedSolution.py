# %% 导入包
from numpy.random import RandomState
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import random
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import shap
from pdpbox import pdp, get_dataset,info_plots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn import tree
import graphviz 

from imodels import RuleFitRegressor

# %% 
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% [markdown]

# ### 1、导入数据、分类变量编码、训练集测试集划分
# ### 1、导入数据、分类变量编码、训练集测试集划分
# ### 1、导入数据、分类变量编码、训练集测试集划分
# ### 1、导入数据、分类变量编码、训练集测试集划分



# %% 以给定格式导入数据
dtypes = [
    ("Size (nm)", "float32"), ("Zeta potential (mV)", "float32"), 
    ("Hydrodynamic diameter (nm)", "float32"), ("Nanoparticle concentration (mg/L)", "float32"),
    ("Solution pH", "float32"), ("NaCl concentration (M)", "float32"), 
    ("Seed thickness", "category"),("Seed width", "category"), 
    ("Zn content (mg/Kg)", "float32")
]

data = pd.read_excel(
    "Dataset_Nano_SeedPriming.xlsx",
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
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

h=sns.heatmap(corr, cmap='Blues',  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8})

cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left',fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=13)
fig.savefig("./Image/data_corr.jpg",dpi=600,bbox_inches='tight')

# %% 将顺序打乱，并分配X和Y
data_sample = data_label.sample(frac=1, replace=False, random_state=1)
X = data_sample.drop(['Zn content (mg/Kg)'], axis=1)
Y = data_sample['Zn content (mg/Kg)']
X

# %% 80%的数据用来做5折交叉验证，20%的数据用来做测试集
X_cv, X_test, Y_cv, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_cv.head(5)

# %% [markdown]

# ### 2、使用随机森林进行建模
# ### 2、使用随机森林进行建模
# ### 2、使用随机森林进行建模
# ### 2、使用随机森林进行建模



# %% 建立性能矩阵，保存所有模型的性能结果
performance_df = pd.DataFrame(
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test')
            )

# %% 2.1 使用gridsearch确定对所有特征的随机森林模型超参数

parameter = {

    # 1.
    #"n_estimators":np.arange(0,1001,50), # (0,1001,50)

    # 2.
    #"max_depth":np.arange(1,20,1),
    #"min_samples_split": np.arange(1,20,1),
    
    # 3.
    #"min_samples_leaf":  np.arange(1,20,1),
    #"min_samples_split":  np.arange(1,20,1),
    
    # 4.
    # "max_features":np.arange(1,9,1),

    # 5.
    # "max_leaf_nodes":np.arange(1,50,1),


}

model_gs = RandomForestRegressor(n_jobs = -1, n_estimators = 100,
                             max_depth = 15,
                             min_samples_leaf = 2, min_samples_split = 5, 
                             max_features = 8,
                             max_leaf_nodes = 19,
                             random_state = 1,
                            )

                
grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_cv, Y_cv)

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

RF_Gs_best = grid_search.best_estimator_

print('GridSearchCV后Random forest的性能')
cv_R2 = cross_val_score(RF_Gs_best, X_cv, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(RF_Gs_best, X_cv, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE



# %% 使用gridsearch确定的超参数对所有特征建立随机森林模型
RF_model = RandomForestRegressor(n_jobs = -1, n_estimators = 100,
                             max_depth = 15,
                             min_samples_leaf = 2, min_samples_split = 5, 
                             max_features = 8,
                             max_leaf_nodes = 19,
                             random_state = 1,
                         )

RF_model.fit(X_cv, Y_cv)
RF_importance = RF_model.feature_importances_

Y_cv_pred = RF_model.predict(X_cv)
Y_pred = RF_model.predict(X_test)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['RF_GS_All']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% 模型性能
figure = plt.figure(figsize=(5,5))  # 设定画布大小
plt.style.use('classic')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)
plt.plot()
figure.savefig("./Image/Performance_RF_GS_All.jpg",dpi=600,bbox_inches='tight')


# %% [markdown]

# ### 特征重要性测量
# ### 特征重要性测量


# %% RF特征重要性
figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
sorted_idx = RF_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

""" # 删除最重要的两个特征，以实现更好的可视化
importance_plot = list(RF_importance[sorted_idx])
importance_del = [feature_this_plot.index('Nanoparticle concentration (mg/L)'), 
                feature_this_plot.index('NaCl concentration (M)')]
importance_plot = np.delete(importance_plot,importance_del)
feature_this_plot = np.delete(feature_this_plot,importance_del) """

plt.barh(feature_this_plot,
         RF_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('RF feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=18)
plt.xscale('log')
plt.xticks([0.01,0.1,1])


figure.savefig("./Image/RF_importance.jpg",dpi=600,bbox_inches='tight')

# %% 
result = permutation_importance(RF_model, X_cv, Y_cv, scoring='r2', n_repeats=10, random_state=1, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.margins(0.02)
sorted_idx = Permutation_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         Permutation_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('Permutation feature importance',fontsize=18)
plt.xscale('log')
plt.xlabel('Importance value',fontsize=16)

figure.savefig("./Image/Permutation_importance.jpg",dpi=600,bbox_inches='tight')

# %% 
explainer = shap.TreeExplainer(model=RF_model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_cv)
global_shap_values = np.abs(shap_values).mean(0)

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
sorted_idx = global_shap_values.argsort()
sorted_features = X_cv.columns[sorted_idx]
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         global_shap_values[sorted_idx], align='center', color="#1E90FF")

plt.title('SHAP feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
plt.xscale('log')
figure.savefig("./Image/Shap_importance.jpg",dpi=600,bbox_inches='tight')



# %% 
# 计算相对重要性，最大为1
RF_importance_relative = RF_importance/max(RF_importance)
Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
shap_values__relative = global_shap_values/max(global_shap_values)

# 以 relative importance 之和为基准进行排序
importance_sum = RF_importance_relative+Permutation_importance_relative+shap_values__relative
sorted_idx_sum = importance_sum.argsort()
sorted_features = X_cv.columns[sorted_idx_sum][::-1]

np.save('sorted_features.npy',sorted_features.tolist())

importance_df = pd.DataFrame({'Feature':X_cv.columns[sorted_idx_sum],
                    'Random Forest':RF_importance_relative[sorted_idx_sum],
                    'Permutatio':Permutation_importance_relative[sorted_idx_sum],
                    'SHAP':shap_values__relative[sorted_idx_sum]},
                    )
importance_df

# %% 
importance_df = pd.DataFrame(columns=('Feature','Method','Relative importance value'))
n_feature = len(X_cv.columns)

for i in range(0,n_feature):
    importance_df.loc[i,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i,'Method'] = 'RF'
    importance_df.loc[i,'Relative importance value'] = RF_importance_relative[sorted_idx_sum][-i-1]

for i in range(0,n_feature):
    importance_df.loc[i+n_feature,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature,'Method'] = 'Permutation'
    importance_df.loc[i+n_feature,'Relative importance value'] = Permutation_importance_relative[sorted_idx_sum][-i-1]
    
for i in range(0,n_feature):
    importance_df.loc[i+n_feature*2,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature*2,'Method'] = 'SHAP'
    importance_df.loc[i+n_feature*2,'Relative importance value'] = shap_values__relative[sorted_idx_sum][-i-1]

RF_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][0:n_feature].values,reverse=True)
Permutation_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*1:n_feature*2].values,reverse=True)
SHAP_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values,reverse=True)

# %% 
annotate_RF = []
annotate_Permutation = []
annotate_SHAP = []

for i in range(0,n_feature,1):
    annotate_RF.append(RF_sorted_value.index(importance_df.loc[:,'Relative importance value'][0:n_feature].values[i])+1)
    annotate_Permutation.append(Permutation_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature:n_feature*2].values[i])+1)
    annotate_SHAP.append(SHAP_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values[i])+1)
    
annotate_value = np.hstack((annotate_RF, annotate_Permutation, annotate_SHAP))
annotate_value

# %% 
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

figure = plt.figure(figsize=(6,5))
plt.style.use('classic')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
bar = sns.barplot(data = importance_df,y='Feature',x='Relative importance value',hue='Method',palette="rocket")
bar.set_ylabel('',fontsize=16)
bar.set_xlabel('Relative importance value',fontsize=16)
bar.set_yticklabels(feature_this_plot,fontsize=16)
plt.legend(loc='lower right')
i=0
plt.margins(0.02)
for p in bar.patches:
    bar.annotate("%d" %annotate_value[i], xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    i=i+1

plt.xscale('log')
figure.savefig("./Image/Importance_summary.jpg",dpi=600,bbox_inches='tight')

# %% 
""" from brokenaxes import brokenaxes

figure = plt.figure(figsize=(8,6))

baxes = brokenaxes(xlims=((0,0.0052),(0.0875,0.091),(0.897,0.901)), hspace=0.05)

baxes.barh(feature_this_plot,RF_importance[sorted_idx], color="#1E90FF")

baxes.set_xlabel('\n'+'Relative feature importance',fontsize=18)
baxes.axs[0].set_xticks([0.000,0.002,0.004])
baxes.axs[1].set_xticks([0.088,0.090])
baxes.axs[2].set_xticks([0.898,0.900])

plt.show() """

# %% 2.2 使用默认参数建模对所有特征的随机森林模型超参数

RF_model_default = RandomForestRegressor(n_jobs = -1,
                        random_state = 1,
                         )

RF_model_default.fit(X_cv, Y_cv)
RF_importance = RF_model_default.feature_importances_

Y_cv_pred = RF_model_default.predict(X_cv)
Y_pred = RF_model_default.predict(X_test)

print('默认参数Random forest的性能')

cv_R2 = cross_val_score(RF_model_default, X_cv, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(RF_model_default, X_cv, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE


performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['RF_Default_All']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% 2.3 使用gridsearch确定超参数对两个重要特征的随机森林模型

X_cv_2 = X_cv.loc[:,["Nanoparticle concentration (mg/L)","NaCl concentration (M)"]]
X_test_2 = X_test.loc[:,["Nanoparticle concentration (mg/L)","NaCl concentration (M)"]]

parameter = {

    # 1.
    #"n_estimators":np.arange(0,1001,50), # (0,1001,50)

    # 2.
    #"max_depth":np.arange(1,20,1),
    #"min_samples_split": np.arange(1,20,1),
    
    # 3.
    #"min_samples_leaf":  np.arange(1,20,1),
    #"min_samples_split":  np.arange(1,20,1),
    
    # 4.
    #"max_features":np.arange(1,9,1),

}

RF_model_gs_two = RandomForestRegressor(n_jobs = -1, n_estimators = 100,
                             max_depth=12,
                             min_samples_leaf = 2, min_samples_split = 3, 
                             max_features=2,
                             random_state = 1,
                            )

                
grid_search = GridSearchCV(RF_model_gs_two, param_grid = parameter, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_cv_2, Y_cv)

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

RF_gs_best_Two = grid_search.best_estimator_

print('GridSearchCV后Random forest的性能')
cv_R2 = cross_val_score(RF_gs_best_Two, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(RF_gs_best_Two, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE


# %% 使用GS后的超参数建立基于两个重要特征的随机森林模型
RF_model_gs_two = RandomForestRegressor(n_jobs = -1, n_estimators = 100,
                             max_depth=12,
                             min_samples_leaf = 2, min_samples_split = 3, 
                             max_features=2,
                             random_state = 1,
                         )

RF_model_gs_two.fit(X_cv_2, Y_cv)
RF_importance = RF_model_gs_two.feature_importances_

Y_cv_pred = RF_model_gs_two.predict(X_cv_2)
Y_pred = RF_model_gs_two.predict(X_test_2)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['RF_GS_Two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df



# %% 2.4 使用默认参数建立只有两个重要特征的随机森林模型
RF_model_two = RandomForestRegressor(n_jobs = -1,
                         random_state = 1,
                         )

RF_model_two.fit(X_cv_2, Y_cv)
RF_importance = RF_model_two.feature_importances_

Y_cv_pred = RF_model_two.predict(X_cv_2)
Y_pred = RF_model_two.predict(X_test_2)

print('基于两个参数使用默认参数Random forest的性能')

cv_R2 = cross_val_score(RF_model_two, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(RF_model_two, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['RF_Default_Two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% [markdown]

# ### 3、使用决策树模型
# ### 3、使用决策树模型
# ### 3、使用决策树模型
# ### 3、使用决策树模型



# %% 3.1 使用gridsearch对所有特征进行决策树建模
parameter = {

    # 1.
    #"max_depth":np.arange(1,20,1),
    #"min_samples_split": np.arange(1,20,1),
    
    # 2.
    #"min_samples_leaf":  np.arange(1,20,1),
    #"min_samples_split":  np.arange(1,20,1),
    
    # 3.
    # "max_features":np.arange(1,9,1),
    
    # 4.
    # "max_leaf_nodes":np.arange(1,50,1),

}

model_gs = DecisionTreeRegressor(
                                max_depth = 5, 
                                min_samples_leaf = 3,  min_samples_split = 9,
                                max_features = 7,
                                max_leaf_nodes = 27,
                                random_state = 1,
                                )

grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='r2', cv=5, n_jobs=-1)

grid_search.fit(X_cv, Y_cv)

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

DT_gs_best_All = grid_search.best_estimator_

print('RMSE:', abs(sum(cross_val_score(DT_gs_best_All, X_cv, Y_cv, cv=5, 
                            scoring='neg_root_mean_squared_error'))/5))

cv_R2 = cross_val_score(DT_gs_best_All, X_cv, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(DT_gs_best_All, X_cv, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

# %% 使用调参后的超参数建立全参数决策树模型

DT_GS_All = DecisionTreeRegressor(max_depth = 5, 
                                min_samples_leaf = 3,  min_samples_split = 9,
                                max_features = 7,
                                max_leaf_nodes = 27,
                                random_state = 1,
                                )
DT_GS_All.fit(X_cv, Y_cv)

Y_cv_pred = DT_GS_All.predict(X_cv)
Y_pred = DT_GS_All.predict(X_test)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE



performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['DT_GS_All']
            )
performance_df = performance_df.append(performance_df_this)
performance_df


# %%  3.2 使用默认参数对所有特征进行决策树建模

DT_Default_All = DecisionTreeRegressor(random_state = 1,)

DT_Default_All.fit(X_cv, Y_cv)

Y_cv_pred = DT_Default_All.predict(X_cv)
Y_pred = DT_Default_All.predict(X_test)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

cv_R2 = cross_val_score(DT_Default_All, X_cv, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(DT_Default_All, X_cv, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE


performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['DT_Default_All']
            )
performance_df = performance_df.append(performance_df_this)
performance_df


# %%  3.3 使用gridsearch对两个重要特征进行决策树建模

parameter = {

    # 1.
    #"max_depth":np.arange(1,20,1),
    #"min_samples_split": np.arange(1,20,1),
    
    # 2.
    #"min_samples_leaf":  np.arange(1,20,1),
    #"min_samples_split":  np.arange(1,20,1),
    
    # 3.
     #"max_features":np.arange(1,9,1),
    
    # 4.
      "max_leaf_nodes":np.arange(1,50,1),

}

model_gs = DecisionTreeRegressor(
                                max_depth = 6, 
                                min_samples_leaf = 1,  min_samples_split = 5,
                                max_features = 2,
                                max_leaf_nodes = 21,
                                random_state = 1,
                                )


grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_cv_2, Y_cv)

print('best score: ', grid_search.best_score_)
print('best_params:', grid_search.best_params_)

DT_Gs_best_Two = grid_search.best_estimator_

print('RMSE:', abs(sum(cross_val_score(DT_Gs_best_Two, X_cv_2, Y_cv, cv=5, 
                            scoring='neg_root_mean_squared_error'))/5))


# %%

DT_GS_two = DecisionTreeRegressor( max_depth = 6, 
                                min_samples_leaf = 1,  min_samples_split = 5,
                                max_features = 2,
                                max_leaf_nodes = 21,
                                random_state = 1,)
DT_GS_two.fit(X_cv_2, Y_cv)

Y_cv_pred = DT_GS_two.predict(X_cv_2)
Y_pred = DT_GS_two.predict(X_test_2)

cv_R2 = cross_val_score(DT_GS_two, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(DT_GS_two, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['DT_GS_two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% 性能

figure = plt.figure(figsize=(4,4))  # 设定画布大小
plt.style.use('classic')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)
plt.plot()
figure.savefig("./Image/Performance_DT_GS_Two.jpg",dpi=600,bbox_inches='tight')

# %% 3.4 使用默认参数对两个重要特征进行决策树建模

DT_Deafult_Two = DecisionTreeRegressor(random_state = 1,)
DT_Deafult_Two.fit(X_cv_2, Y_cv)

Y_cv_pred = DT_Deafult_Two.predict(X_cv_2)
Y_pred = DT_Deafult_Two.predict(X_test_2)

cv_R2 = cross_val_score(DT_Deafult_Two, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(DT_Deafult_Two, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['DT_Deafult_Two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df


# %% 3.5 决策树树形图输出

dot_data = tree.export_graphviz(DT_GS_two, out_file=None,
                        feature_names=['NP Conc.','NaCl Conc.'],) 
graph = graphviz.Source(dot_data) 
graph.render("DT_Deafult_Two_All_Nodes") 


DT_GS_two_8_Nodes = DecisionTreeRegressor( max_depth = 6, 
                                min_samples_leaf = 1,  min_samples_split = 5,
                                max_features = 2,
                                max_leaf_nodes = 8,)

DT_GS_two_8_Nodes.fit(X_cv_2, Y_cv)

Y_cv_pred = DT_GS_two_8_Nodes.predict(X_cv_2)
Y_pred = DT_GS_two_8_Nodes.predict(X_test_2)

cv_R2 = cross_val_score(DT_GS_two_8_Nodes, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(DT_GS_two_8_Nodes, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['DT_Deafult_Two_8_nodes']
            )
performance_df = performance_df.append(performance_df_this)

dot_data = tree.export_graphviz(DT_GS_two_8_Nodes, 
                        out_file=None,
                        feature_names=['NP Conc.','NaCl Conc.'],) 
graph = graphviz.Source(dot_data) 
graph.render("DT_Deafult_Two_8_Nodes") 




# %% [markdown]
# ### 4、重要特征与预测值之间的可视化图
# ### 4、重要特征与预测值之间的可视化图
# ### 4、重要特征与预测值之间的可视化图
# ### 4、重要特征与预测值之间的可视化图



# %% 重要特征和Zn含量的散点图
figure = plt.figure(figsize=(6,6))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax = figure.add_subplot(111)

img = ax.scatter(X['Nanoparticle concentration (mg/L)'], Y, c=X['NaCl concentration (M)'], 
                linewidths=0, cmap='rainbow')

cb = figure.colorbar(img)
cb.set_label('NaCl concentration (M) (mol/L)')

ax.set_xlabel('Nanoparticle concentration (mg/L) (mg/L)')
ax.set_ylabel('Zn content (mg/Kg) (mg/Kg)')

plt.show()

figure.savefig("./Image/Conc._IS_Zn.jpg",dpi=600,bbox_inches='tight')

# %%

figure = plt.figure(figsize=(8,6))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax = figure.add_subplot(111)
sns.scatterplot(
    data=data_label, x="Nanoparticle concentration (mg/L)", y="Zn content (mg/Kg)", hue="NaCl concentration (M)", 
    legend="full", palette="tab10", s = 50, markers = 'X',
    )

ax.set_xlabel('Nanoparticle concentration (mg/L) ', fontsize='16')
plt.xticks(fontsize=14)
ax.set_ylabel('Zn content (mg/Kg) ', fontsize='16')
plt.yticks(fontsize=14)

ax.legend(loc="lower right", title='NaCl concentration (M)')
plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title

figure.savefig("./Image/Conc._IS_Zn.jpg",dpi=600,bbox_inches='tight')


# %% [markdown]
# ### 5、根据两个重要特征的线性回归模型
# ### 5、根据两个重要特征的线性回归模型
# ### 5、根据两个重要特征的线性回归模型
# ### 5、根据两个重要特征的线性回归模型


# %% 5、根据两个重要特征的线性回归模型
reg = LinearRegression()

reg.fit(X_cv_2, Y_cv)

print(reg.coef_, reg.intercept_)

Y_cv_pred = reg.predict(X_cv_2)
Y_pred = reg.predict(X_test_2)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

cv_R2 = cross_val_score(reg, X_cv_2, Y_cv, cv=5)
print('R2_CV:: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(reg, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['LinearReg_two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% 线性回归模型性能

figure = plt.figure(figsize=(4,4))  # 设定画布大小
plt.style.use('classic')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)
plt.plot()
figure.savefig("./Image/Performance_LinearReg_Two.jpg",dpi=600,bbox_inches='tight')


# %% [markdown]
# ### 6、根据两个重要特征的非线性回归模型
# ### 6、根据两个重要特征的非线性回归模型
# ### 6、根据两个重要特征的非线性回归模型
# ### 6、根据两个重要特征的非线性回归模型

# %% 6、非线性回归
ploy_reg = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression())])

ploy_reg = ploy_reg.fit(X_cv_2, Y_cv)
print(ploy_reg.named_steps['linear'].coef_)
print(ploy_reg.named_steps['linear'].intercept_)
Y_cv_pred = ploy_reg.predict(X_cv_2)
Y_pred = ploy_reg.predict(X_test_2)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE

cv_R2 = cross_val_score(ploy_reg, X_cv_2, Y_cv, cv=5)
print('R2_CV: %.3f(%.3f)' %(cv_R2.mean(),cv_R2.std())) # 5折交叉验证R2
cv_RMSE = -cross_val_score(ploy_reg, X_cv_2, Y_cv, cv=5,scoring='neg_root_mean_squared_error')
print('RMSE_CV: %.3f(%.3f)' %(cv_RMSE.mean(),cv_RMSE.std())) # 5折交叉验证RMSE

performance_list_this = [
            [cv_R2.mean()],
            [cv_R2.std()],
            [cv_RMSE.mean()],
            [cv_RMSE.std()],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['NonLinearReg_two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %% 非线性线性回归模型性能

figure = plt.figure(figsize=(4,4))  # 设定画布大小
plt.style.use('classic')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)

plt.plot()
figure.savefig("./Image/Performance_NonLinearReg_Two.jpg",dpi=600,bbox_inches='tight')

# %% [markdown]
# ### 7、两个重要特征的RF模型解释
# ### 7、两个重要特征的RF模型解释
# ### 7、两个重要特征的RF模型解释
# ### 7、两个重要特征的RF模型解释


# %% SHAP_Effect_Concentration

feature = 'Nanoparticle concentration (mg/L)'
shap_values = explainer.shap_values(X_cv)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax_shap = shap.dependence_plot(feature, shap_values, X_cv, ax=ax, show=False,
                               display_features=X_cv,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/SHAP_Effect_Concentration.jpg",dpi=600,bbox_inches='tight')

# %% SHAP_Effect_IS
feature = 'NaCl concentration (M)'

plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
fig, ax = plt.subplots(figsize=(4, 3))
ax_shap = shap.dependence_plot(feature, shap_values, X_cv, ax=ax, show=False,
                               display_features=X_cv,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/SHAP_Effect_IS.jpg",dpi=600,bbox_inches='tight')


# %% SHAP 水动力直径
feature = 'Hydrodynamic diameter (nm)'

plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
fig, ax = plt.subplots(figsize=(4, 3))
ax_shap = shap.dependence_plot(feature, shap_values, X_cv, ax=ax, show=False,
                               display_features=X_cv,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/SHAP_Effect_Hydro_diameter.jpg",dpi=600,bbox_inches='tight')

# %% SHAP zeta电位
feature = 'Zeta potential (mV)'

plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
fig, ax = plt.subplots(figsize=(4, 3))
ax_shap = shap.dependence_plot(feature, shap_values, X_cv, ax=ax, show=False,
                               display_features=X_cv,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/SHAP_Effect_zeta.jpg",dpi=600,bbox_inches='tight')

# %% SHAP 交互作用强度
shap_interaction_values = shap.TreeExplainer(model=RF_model).shap_interaction_values(X_cv)

fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X_cv.columns[inds]
df_temp2.index = X_cv.columns[inds]

h=sns.heatmap(df_temp2, cmap='viridis', square=True, center=200,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='left',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,ha='right',rotation_mode='anchor',fontsize=15)

fig.savefig("./Image/SHAP_Interaction_strength.jpg",dpi=600,bbox_inches='tight')

# %% SHAP_Interact_MONP_Clay
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
feature1 = 'Nanoparticle concentration (mg/L)'
feature2 = 'NaCl concentration (M)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X_cv,ax=ax,show=False,
                    display_features=X_cv)


fig.savefig("./Image/SHAP_Interact_Nc_IS.jpg",dpi=600,bbox_inches='tight')

# %% PDP浓度

feature = "Nanoparticle concentration (mg/L)"

pdp_NP_none_M = pdp.pdp_isolate(model=RF_model,
                        dataset=X_cv,
                        model_features=X_cv.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=20)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/PDP_Effect_Conc.jpg",dpi=600,bbox_inches='tight') 

# %% PDP溶液离子强度
feature = 'NaCl concentration (M)'

pdp_NP_none_M = pdp.pdp_isolate(model=RF_model,
                        dataset=X_cv,
                        model_features=X_cv.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=10)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/PDP_Effect_IS.jpg",dpi=600,bbox_inches='tight') 

# %% PDP水动力直径
feature = 'Hydrodynamic diameter (nm)'

pdp_NP_none_M = pdp.pdp_isolate(model=RF_model,
                        dataset=X_cv,
                        model_features=X_cv.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=10)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/PDP_Effect_Hydro_diameter.jpg",dpi=600,bbox_inches='tight') 


# %% PDPzeta电位
feature = 'Zeta potential (mV)'

pdp_NP_none_M = pdp.pdp_isolate(model=RF_model,
                        dataset=X_cv,
                        model_features=X_cv.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=10)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/PDP_Effect_Zeta.jpg",dpi=600,bbox_inches='tight') 


# %% PDP交互作用图

inter1 = pdp.pdp_interact(model=RF_model,
                          dataset=X_cv,
                          model_features=X_cv.columns,
                          features=['Nanoparticle concentration (mg/L)', 'NaCl concentration (M)'],
                          num_grid_points=[20, 10],
                          percentile_ranges=[(5, 95), (5, 95)])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1,
                                  feature_names=['Nanoparticle concentration (mg/L)', 'NaCl concentration (M)'],
                                  plot_type='contour',
                                  x_quantile=True,
                                  plot_pdp=False,figsize=(4, 5))   

fig.savefig("./Image/PDP_Interact_Conc_IS.jpg",dpi=600,bbox_inches='tight') 



# %% [markdown]
# ### 8、分段线性回归
# ### 8、分段线性回归
# ### 8、分段线性回归
# ### 8、分段线性回归


# %% 浓度小于等于 50 mg/L 的线性回归
X_cv_low = X_cv_2.loc[X_cv_2['Nanoparticle concentration (mg/L)'] <= 50]
Y_cv_low = Y_cv[X_cv_low.index]

X_test_low = X_test_2.loc[X_test_2['Nanoparticle concentration (mg/L)'] <= 50]
Y_test_low = Y_test[X_test_low.index]

reg = LinearRegression()
reg.fit(X_cv_low, Y_cv_low)

low_coef = reg.coef_
low_intercept = reg.intercept_
print(reg.coef_, reg.intercept_)


# %% 浓度大于 50 mg/L 的线性回归
X_cv_high = X_cv_2.loc[X_cv_2['Nanoparticle concentration (mg/L)'] > 50]
Y_cv_high = Y_cv[X_cv_high.index]

X_test_high = X_test_2.loc[X_test_2['Nanoparticle concentration (mg/L)'] > 50]
Y_test_high = Y_test[X_test_high.index]

reg = LinearRegression()
reg.fit(X_cv_high, Y_cv_high)

high_coef = reg.coef_
high_intercept = reg.intercept_
print(reg.coef_, reg.intercept_)


# %% 训练集五折划分测试分段函数性能

# 重置训练集索引（已经是随机排序的顺序 ）

kf = KFold(n_splits=5)
kf.get_n_splits(X_cv_2)

r2_5fold_piecewise = []
RMSE_5fold_piecewise = []
for train_index, test_index in kf.split(X_cv_2):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_piecewise, X_test_piecewise = X_cv_2.iloc[train_index], X_cv_2.iloc[test_index] # iloc按照行号索引，loc按index
    Y_train_piecewise, Y_test_piecewise = Y_cv.iloc[train_index], Y_cv.iloc[test_index]
    Y_test_piecewise_pred = []
    for i in range(0,len(X_test_piecewise),1):
        if X_test_piecewise.iloc[i,0] <= 50:
            y_test_piecewise_pred =  np.dot(X_test_piecewise.iloc[i,:].values,low_coef) + low_intercept
        else:
            y_test_piecewise_pred =  np.dot(X_test_piecewise.iloc[i,:].values,high_coef) + high_intercept
        Y_test_piecewise_pred.append(y_test_piecewise_pred)
    r2_5fold_piecewise.append(r2_score(Y_test_piecewise, Y_test_piecewise_pred))
    RMSE_5fold_piecewise.append(sqrt(mean_squared_error(Y_test_piecewise, Y_test_piecewise_pred)))


# %% 分段函数测试集和测试集上的性能

Y_cv_piecewise_pred = []
for i in range(0,len(X_cv_2),1):
    if X_cv_2.iloc[i,0] <= 50:
        y_cv_piecewise_pred =  np.dot(X_cv_2.iloc[i,:].values,low_coef) + low_intercept
    else:
        y_cv_piecewise_pred =  np.dot(X_cv_2.iloc[i,:].values,high_coef) + high_intercept
    Y_cv_piecewise_pred.append(y_cv_piecewise_pred)

Y_test_piecewise_pred = []
for i in range(0,len(X_test_2),1):
    if X_test_2.iloc[i,0] <= 50:
        y_test_piecewise_pred =  np.dot(X_test_2.iloc[i,:].values,low_coef) + low_intercept
    else:
        y_test_piecewise_pred =  np.dot(X_test_2.iloc[i,:].values,high_coef) + high_intercept
    Y_test_piecewise_pred.append(y_test_piecewise_pred)

print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_piecewise_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_piecewise_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_test_piecewise_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_test_piecewise_pred)))  # 测试集MSE


performance_list_this = [
            [np.mean(r2_5fold_piecewise)],
            [np.std(r2_5fold_piecewise)],
            [np.mean(RMSE_5fold_piecewise)],
            [np.std(RMSE_5fold_piecewise)],
            [r2_score(Y_cv, Y_cv_piecewise_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_piecewise_pred))],
            [r2_score(Y_test, Y_test_piecewise_pred)],
            [sqrt(mean_squared_error(Y_test, Y_test_piecewise_pred))]
            ]

performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['Piecewise_Linear_Two']
            )
performance_df = performance_df.append(performance_df_this)

performance_df
# %% 分段函数性能图
figure = plt.figure(figsize=(4,4))  # 设定画布大小
plt.style.use('classic')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_piecewise_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_test_piecewise_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)
plt.plot()
figure.savefig("./Image/Performance_PiecewiseReg_Two.jpg",dpi=600,bbox_inches='tight')

# %%
dtypes = [
    ("Nanoparticle concentration (mg/L)", "float32"),
    ("NaCl concentration (M)", "float32"), 
    ("Shaking condition", "category"),
    ("Zn content (mg/Kg)", "float32")
]

data_validation = pd.read_excel(
    "Dataset_Nano_SeedPriming_Validation.xlsx",
    names = [d[0] for d in dtypes], 
    index_col = None,
    dtype = dict(dtypes)
)
data_validation

# %%
data_validation_Shaking = data_validation[data_validation['Shaking condition']=='Shaking']
y_validation = data_validation_Shaking['Zn content (mg/Kg)']
data_validation_Shaking = data_validation_Shaking.drop(['Shaking condition','Zn content (mg/Kg)'], axis=1)
data_validation_Shaking
# %%

Y_cv_piecewise_pred_validation = []
for i in range(0,len(data_validation_Shaking),1):
    if data_validation_Shaking.iloc[i,0] <= 50:
        y_cv_piecewise_pred =  np.dot(data_validation_Shaking.iloc[i,:].values,low_coef) + low_intercept
    else:
        y_cv_piecewise_pred =  np.dot(data_validation_Shaking.iloc[i,:].values,high_coef) + high_intercept
    Y_cv_piecewise_pred_validation.append(y_cv_piecewise_pred)
Y_cv_piecewise_pred_validation

print('R2_train: %.3f ' %r2_score(y_validation, Y_cv_piecewise_pred_validation))  # 验证数据R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(y_validation, Y_cv_piecewise_pred_validation)))  # 验证数据RMSE

# %% 规则拟合算法
feature_names = X_cv_2.columns
rulefit = RuleFitRegressor(random_state=2,tree_generator=RandomForestRegressor(n_jobs=-1,n_estimators = 100))
rulefit.fit(X_cv_2, Y_cv, feature_names=feature_names)

Y_pred = rulefit.predict(X_test_2)
Y_cv_pred = rulefit.predict(X_cv_2)


print('R2_train: %.3f ' %r2_score(Y_cv, Y_cv_pred))  # 训练集R方
print('RMSE_train: %.3f ' %sqrt(mean_squared_error(Y_cv, Y_cv_pred)))  # 训练集MSE
print('R2_test: %.3f ' %r2_score(Y_test, Y_pred))  # 测试集R方
print('RMSE_test: %.3f ' %sqrt(mean_squared_error(Y_test, Y_pred)))  # 测试集MSE


# %%

kf = KFold(n_splits=5)
i = 0 # 0
train_r2 = []
test_r2 = []
train_RMSE = []
test_RMSE = []
for train_index, test_index in kf.split(X_cv_2):
    i += 1
    X_train, X_test = X_cv_2.iloc[train_index,:], X_cv_2.iloc[test_index,:]
    y_train, y_test = Y_cv.values[train_index], Y_cv.values[test_index]
    model = RuleFitRegressor(random_state=i*5,tree_generator=RandomForestRegressor(n_jobs=-1,n_estimators = 100))  # initialize a model
    model.fit(X_train, y_train, feature_names=feature_names)   # fit model
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    train_r2.append(r2_score(y_train,y_train_pred))
    train_RMSE.append(sqrt(mean_squared_error(y_train, y_train_pred)))
    test_r2.append(r2_score(y_test,y_pred))
    test_RMSE.append(sqrt(mean_squared_error(y_test, y_pred)))


# %%
performance_list_this = [
            [np.mean(test_r2)],
            [np.std(test_r2)],
            [np.mean(test_RMSE)],
            [np.std(test_RMSE)],
            [r2_score(Y_cv, Y_cv_pred)],
            [sqrt(mean_squared_error(Y_cv, Y_cv_pred))],
            [r2_score(Y_test, Y_pred)],
            [sqrt(mean_squared_error(Y_test, Y_pred))]
            ]
performance_df_this = pd.DataFrame(
            list(map(list,zip(*performance_list_this))),
            columns = ('R2_CV','R2_CV_std','RMSE_CV','RMSE_CV_std','R2_train','RMSE_train','R2_test','RMSE_test'),
            index=['RuleFit_two']
            )
performance_df = performance_df.append(performance_df_this)
performance_df

# %%
figure = plt.figure(figsize=(4,4))  # 设定画布大小
plt.style.use('classic')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.xlim(min(Y)-20, max(Y)+20)
plt.ylim(min(Y)-20, max(Y)+20)
plt.plot([min(Y)-20,max(Y)+20],[min(Y)-20,max(Y)+20],color="slategrey",linestyle='--')
plt.scatter(Y_cv, Y_cv_pred,marker="o",color="#9932CC", s=20,label="training set")   
plt.scatter(Y_test, Y_pred,marker="*",color="#FF8C00", s=40, label="test set")
plt.xlabel('Observed Zn content (mg/Kg)',fontsize=15)
plt.ylabel('Predicted Zn content (mg/Kg)',fontsize=15)
plt.legend(loc=0,fontsize=12,scatterpoints=1)
plt.plot()
figure.savefig("./Image/Performance_RuleFit_Two.jpg",dpi=600,bbox_inches='tight')

# %%
rules = model.get_rules()
rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
rules.to_excel('rules_two.xlsx')
rules
# %%
