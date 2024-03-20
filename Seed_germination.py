# %% 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from matplotlib import rcParams
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imodels import RuleFitClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestRegressor

from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import shap
from pdpbox import pdp, get_dataset,info_plots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn import tree
import graphviz 
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 计算单个特征的重要性
def find_mk(input_vars:list, rule:str):
    """
    Finds the number of input variables in a rule.
    
    Parameters:
    -----------
        input_vars (list): 
        
        rule (str):
    """
    var_count = 0
    for var in input_vars:
        if var in rule:
            var_count += 1
    return(var_count)

def get_feature_importance(feature_set: list, rule_set: pd.DataFrame, scaled = False):
    """
    Returns feature importance for input features to rulefit model.
    
    Parameters:
    -----------
        feature_set (list): 
        
        rule (str): 
    """
    feature_imp = list()
    
    rule_feature_count = rule_set.rule.apply(lambda x: find_mk(feature_set, x))

    for feature in feature_set:
        
        # find subset of rules that apply to a feature
        feature_rk = rule_set.rule.apply(lambda x: feature in x)
        
        # find importance of linear features
        linear_imp = rule_set[(rule_set.type=='linear')&(rule_set.rule==feature)].importance.values
        
        # find the importance of rules that contain feature
        rule_imp = rule_set.importance[(rule_set.type=='rule')&feature_rk]
        
        # find the number of features in each rule that contain feature
        m_k = rule_feature_count[(rule_set.type=='rule')&feature_rk]
        
        # sum the linear and rule importances, divided by m_k
        if len(linear_imp)==0:
            linear_imp = 0
        # sum the linear and rule importances, divided by m_k
        if len(rule_imp) == 0:
            feature_imp.append(float(linear_imp))
        else:
            feature_imp.append(float(linear_imp + (rule_imp/m_k).sum()))
        
    if scaled:
        feature_imp = 100*(feature_imp/np.array(feature_imp).max())
    
    return(feature_imp)

# %% 读取数据
data = pd.read_excel(
    "Dataset_germination_4.xlsx",
    index_col = 0,
)
data

# %% 
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

# %% 
data_corr = data_label.copy()
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
fig.savefig("./Image/data_corr_all.jpg",dpi=600,bbox_inches='tight')

# %% 
data_corr = data_label.iloc[:,0:8]
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


# %% 数据集随机化与五折划分
data_sample = data_label.sample(frac=1, replace=False, random_state=1)
data_sample.index = range(0,len(data_sample.index))
data_sample


# %%
'''
performance_df = pd.DataFrame(index=['Train mean','Train std','Test mean','Test std'],
                                columns = data_sample.columns[8:])
key = 0
for col in data_sample.columns[8:]:
    train_auc = []
    test_auc = []

    for i in range(0,10,1):
        X_train, X_test, y_train, y_test = train_test_split(data_sample.iloc[:,0:8], 
                                     data_sample[col], test_size=0.3, random_state=i)
        
        for j in range(0,10,1):
            print('Processing: '+ col + str(key))
            model = RuleFitClassifier(max_rules=10,random_state=key,tree_generator=RandomForestRegressor(n_jobs=-1))  # initialize a model
            model.fit(X_train, y_train)   # fit model
            y_train_pred = model.predict_proba(X_train)[:,1]
            y_pred = model.predict_proba(X_test)[:,1]
            train_auc.append(roc_auc_score(y_train,y_train_pred))
            test_auc.append(roc_auc_score(y_test,y_pred))
            key += 1
    performance_df[col] = [np.mean(train_auc), np.std(train_auc), np.mean(test_auc), np.std(test_auc)]

performance_df

'''
# %%

#performance_df.to_excel('performance_df_30_100.xlsx')
performance_df = data = pd.read_excel(
    "performance_df_30_100.xlsx",
    index_col = 0,
)
fig, ax= plt.subplots(figsize = (6,5))

plt.errorbar(range(0,len(performance_df.columns)), performance_df.loc['Test mean',:],
                    yerr=performance_df.loc['Test std',:],
                    color='#FFA07A', marker='o',linewidth=2, capsize=6, )
ax.set_xticks(range(0,len(performance_df.columns))) 
ax.set_xticklabels(performance_df.columns, rotation=45,horizontalalignment='right')
ax.margins(0.1,0.1) 
plt.xlabel('Predicted target',fontsize=14)
plt.ylabel('AUC',fontsize=14)

fig.savefig("./Image/RuleFit_performance.jpg",dpi=600,bbox_inches='tight')


# %% 
'''
performance_df_shoot = pd.DataFrame(index=['Train mean','Train std','Test mean','Test std'],
                                columns = range(1,11,1))
col = data_sample.columns[10]

key = 200
for i in range(0,10,1):
    X_train, X_test, y_train, y_test = train_test_split(data_sample.iloc[:,0:8], 
                                    data_sample[col], test_size=0.3, random_state=i)
    train_auc = []
    test_auc = []
    for j in range(0,10,1):
            print('Processing:  '+ str(key))
            model = RuleFitClassifier(random_state=key,max_rules=10,tree_generator=RandomForestRegressor(n_jobs=-1))  # initialize a model
            model.fit(X_train, y_train)   # fit model
            y_train_pred = model.predict_proba(X_train)[:,1]
            y_pred = model.predict_proba(X_test)[:,1]
            train_auc.append(roc_auc_score(y_train,y_train_pred))
            test_auc.append(roc_auc_score(y_test,y_pred))
            key += 1

    performance_df_shoot.iloc[:,i] = [np.mean(train_auc), np.std(train_auc), np.mean(test_auc), np.std(test_auc)]

performance_df_shoot
'''

# %%

#performance_df_shoot.to_excel('performance_df_shoot.xlsx')
performance_df_shoot = data = pd.read_excel(
    "performance_df_shoot.xlsx",
    index_col = 0,
)

fig, ax= plt.subplots(figsize = (6,5))

plt.errorbar(range(0,len(performance_df_shoot.columns)), performance_df_shoot.loc['Test mean',:],
                    yerr=performance_df_shoot.loc['Test std',:],
                    color='#87CEFA', marker='o',linewidth=2, capsize=6, )

ax.set_xticks(range(0,len(performance_df_shoot.columns))) 
ax.set_xticklabels(performance_df_shoot.columns,)
ax.margins(0.1,0.1) 
plt.xlabel('Dataset split',fontsize=14)
plt.ylabel('AUC',fontsize=14)

fig.savefig("./Image/RuleFit_performanc_shoot.jpg",dpi=600,bbox_inches='tight')

# %% 
# 模型解释
# 模型解释
# 模型解释
# 模型解释
# 模型解释
# 模型解释
# 模型解释

# %% 用于解释的模型 shoot fresh weight

col = data_sample.columns[10]
X_train, X_test, y_train, y_test = train_test_split(data_sample.iloc[:,0:8], 
                                data_sample[col], test_size=0.3, random_state=9)
# random_state: 240-249
model_shoot = RuleFitClassifier(random_state=242,max_rules=10,tree_generator=RandomForestRegressor(n_jobs=-1))  # initialize a model
model_shoot.fit(X_train, y_train)   # fit model
y_pred = model_shoot.predict_proba(X_test)[:,1]
print('Test AUC: ',roc_auc_score(y_test,y_pred))


# %%

fpr_shoot_rulefit, tpr_shoot_rulefit, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc_shoot_rulefit = metrics.auc(fpr_shoot_rulefit, tpr_shoot_rulefit)

fig, ax= plt.subplots(figsize = (6,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='8'
plt.margins(0.02)
plt.plot(fpr_shoot_rulefit, tpr_shoot_rulefit, 'b',
        label = 'AUC_Shoot fresh weight (RuleFit)= %0.3f' % roc_auc_shoot_rulefit)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# %% 获取规则 shoot fresh weight
# inspect and print the rules
rules_shootfreshweight = model_shoot.get_rules()
rules_shootfreshweight = rules_shootfreshweight[rules_shootfreshweight.coef != 0].sort_values("importance", ascending=False)
rules_shootfreshweight.to_excel('rules_shootfreshweight.xlsx')
rules_shootfreshweight

# %% RuleFit 特征重要性获取 shoot fresh weight
feature_importances = get_feature_importance(X_train.columns, rules_shootfreshweight, scaled=False)
importance_RuleFit = pd.DataFrame(feature_importances, index = X_train.columns, columns = ['importance']).sort_values(by='importance',ascending=False)

RuleFit_importance = importance_RuleFit.loc[X_train.columns]['importance'].values

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)

plt.barh(importance_RuleFit.index[::-1], importance_RuleFit.importance[::-1],
          align='center', color="#1E90FF")

plt.title('RuleFit feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
figure.savefig("./Image/Shoot_RuleFit_importance.jpg",dpi=600,bbox_inches='tight')

# %% permutation_importance shoot fresh weight
result = permutation_importance(model_shoot, X_train, y_train, scoring='roc_auc', n_repeats=10, random_state=1, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.margins(0.02)
sorted_idx = Permutation_importance.argsort()
sorted_features = X_train.columns[sorted_idx]
fature_name = X_train.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         Permutation_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('Permutation feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)

figure.savefig("./Image/Shoot_Permutation_importance.jpg",dpi=600,bbox_inches='tight')

# %% 用于SHAP解释的模型 shoot fresh weight
# random_state = [7,8,9,1,10,11,12,6,2,21]
rf_model_shoot = RandomForestClassifier(n_jobs=-1,n_estimators = 100,
                                            random_state=9,max_depth=4)  # initialize a model
rf_model_shoot.fit(X_train, y_train)   # fit model
y_pred = rf_model_shoot.predict_proba(X_test)[:,1]
print('Test AUC: ',roc_auc_score(y_test,y_pred))

# %%
fpr_shoot_RF, tpr_shoot_RF, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc_shoot_RF = metrics.auc(fpr_shoot_RF, tpr_shoot_RF)

fig, ax= plt.subplots(figsize = (6,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='8'
plt.margins(0.02)
plt.plot(fpr_shoot_RF, tpr_shoot_RF, 'b',
        label = 'AUC_shoot fresh weight (RF)= %0.3f' % roc_auc_shoot_RF)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# %% shap重要性 shoot fresh weight
explainer = shap.TreeExplainer(model=rf_model_shoot, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_train)
global_shap_values = np.abs(shap_values[1]).mean(0)

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
sorted_idx = global_shap_values.argsort()
sorted_features = X_train.columns[sorted_idx]
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         global_shap_values[sorted_idx], align='center', color="#1E90FF")

plt.title('SHAP feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
figure.savefig("./Image/Shoot_shap_importance.jpg",dpi=600,bbox_inches='tight')

# %%  shoot fresh weight
# 计算相对重要性，最大为1
RuleFit_importance_relative = RuleFit_importance/max(RuleFit_importance)
Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
shap_values__relative = global_shap_values/max(global_shap_values)

# 以 relative importance 之和为基准进行排序
importance_sum = RuleFit_importance_relative+Permutation_importance_relative+shap_values__relative
sorted_idx_sum = importance_sum.argsort()
sorted_features = X_train.columns[sorted_idx_sum][::-1]

np.save('sorted_features.npy',sorted_features.tolist())

importance_df = pd.DataFrame({'Feature':X_train.columns[sorted_idx_sum],
                    'RuleFit':RuleFit_importance_relative[sorted_idx_sum],
                    'Permutatio':Permutation_importance_relative[sorted_idx_sum],
                    'SHAP':shap_values__relative[sorted_idx_sum]},
                    )
importance_df


# %% shoot fresh weight
importance_df = pd.DataFrame(columns=('Feature','Method','Relative importance value'))
n_feature = len(X_train.columns)

for i in range(0,n_feature):
    importance_df.loc[i,'Feature'] = X_train.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i,'Method'] = 'RuleFit'
    importance_df.loc[i,'Relative importance value'] = RuleFit_importance_relative[sorted_idx_sum][-i-1]

for i in range(0,n_feature):
    importance_df.loc[i+n_feature,'Feature'] = X_train.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature,'Method'] = 'Permutation'
    importance_df.loc[i+n_feature,'Relative importance value'] = Permutation_importance_relative[sorted_idx_sum][-i-1]
    
for i in range(0,n_feature):
    importance_df.loc[i+n_feature*2,'Feature'] = X_train.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature*2,'Method'] = 'SHAP'
    importance_df.loc[i+n_feature*2,'Relative importance value'] = shap_values__relative[sorted_idx_sum][-i-1]

RuleFit_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][0:n_feature].values,reverse=True)
Permutation_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*1:n_feature*2].values,reverse=True)
SHAP_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values,reverse=True)

annotate_RuleFit = []
annotate_Permutation = []
annotate_SHAP = []

for i in range(0,n_feature,1):
    annotate_RuleFit.append(RuleFit_sorted_value.index(importance_df.loc[:,'Relative importance value'][0:n_feature].values[i])+1)
    annotate_Permutation.append(Permutation_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature:n_feature*2].values[i])+1)
    annotate_SHAP.append(SHAP_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values[i])+1)
    
annotate_value = np.hstack((annotate_RuleFit, annotate_Permutation, annotate_SHAP))
annotate_value

# %% shoot fresh weight
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
plt.title('Shoot fresh weight',fontsize=16)
i=0
plt.margins(0.02)
for p in bar.patches:
    bar.annotate("%d" %annotate_value[i], xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    i=i+1

figure.savefig("./Image/Shoot_Importance_summary.jpg",dpi=600,bbox_inches='tight')

# %% shoot fresh weight  Shoot_PDP_Effect_Conc
feature = "Nanoparticle concentration (mg/L)"

pdp_NP_none_M = pdp.pdp_isolate(model=model_shoot,
                        dataset=X_train,
                        model_features=X_train.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=20)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/Shoot_PDP_Effect_Conc.jpg",dpi=600,bbox_inches='tight') 


# %% shoot fresh weight  Shoot_PDP_Effect_NaCl
feature = "NaCl concentration (M)"

pdp_NP_none_M = pdp.pdp_isolate(model=model_shoot,
                        dataset=X_train,
                        model_features=X_train.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=3)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/Shoot_PDP_Effect_NaCl.jpg",dpi=600,bbox_inches='tight') 

# %% shoot fresh weight  Shoot_PDP_Effect_Zeta
feature = "Zeta potential (mV)"

pdp_NP_none_M = pdp.pdp_isolate(model=model_shoot,
                        dataset=X_train,
                        model_features=X_train.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=20)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/Shoot_PDP_Effect_Zeta.jpg",dpi=600,bbox_inches='tight') 

# %% shoot fresh weight  Shoot_PDP_Effect_pH
feature = "Solution pH"

pdp_NP_none_M = pdp.pdp_isolate(model=model_shoot,
                        dataset=X_train,
                        model_features=X_train.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points=4)

fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_NP_none_M, 
                    plot_lines=True, center=False, 
                    plot_pts_dist=False, 
                    x_quantile=False,
                    feature_name=feature,
                    figsize=(4, 5))

fig.savefig("./Image/Shoot_PDP_Effect_pH.jpg",dpi=600,bbox_inches='tight') 

# %% SHAP
explainer = shap.TreeExplainer(model=rf_model_shoot, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_train)

fig, ax = plt.subplots()
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
shap.summary_plot(shap_values[1],X_train, auto_size_plot=False,plot_size=(4,4))

fig.savefig("./Image/Shoot_SHAP_summary.jpg",dpi=600,bbox_inches='tight')

# %% SHAP Shoot_SHAP_Effect_Conc
feature = "Nanoparticle concentration (mg/L)"
plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax_shap = shap.dependence_plot(feature, shap_values[1], X_train, ax=ax, show=False,
                               display_features=X_train,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/Shoot_SHAP_Effect_Conc.jpg",dpi=600,bbox_inches='tight')

# %% SHAP Shoot_SHAP_Effect_Zeta
feature = "Zeta potential (mV)"
plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax_shap = shap.dependence_plot(feature, shap_values[1], X_train, ax=ax, show=False,
                               display_features=X_train,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/Shoot_SHAP_Effect_Zeta.jpg",dpi=600,bbox_inches='tight')


# %% SHAP Shoot_SHAP_Effect_NaCl
feature = "NaCl concentration (M)"
plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax_shap = shap.dependence_plot(feature, shap_values[1], X_train, ax=ax, show=False,
                               display_features=X_train,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/Shoot_SHAP_Effect_NaCl.jpg",dpi=600,bbox_inches='tight')



# %% SHAP Shoot_SHAP_Effect_pH
feature = "Solution pH"
plt.style.use('default')
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
ax_shap = shap.dependence_plot(feature, shap_values[1], X_train, ax=ax, show=False,
                               display_features=X_train,)
plt.rcParams.update({'font.size': 5})

fig.savefig("./Image/Shoot_SHAP_Effect_pH.jpg",dpi=600,bbox_inches='tight')


# %% 用于预测的模型性能

fig, ax= plt.subplots(figsize = (5,5))
plt.style.use('classic')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
plt.plot([0, 1], [0, 1],color='#C0C0C0', linestyle=(0,(5,10)),linewidth=1,)

plt.plot(fpr_shoot_rulefit, tpr_shoot_rulefit, color='#1F77B4', linestyle=(0,(3,5,1,5)),linewidth=2,
        label = 'AUC_RuleFit= %0.3f' % roc_auc_shoot_rulefit)
plt.plot(fpr_shoot_RF, tpr_shoot_RF, color='#FF7F0E', linestyle='dashed',linewidth=2,
        label = 'AUC_RF= %0.3f' % roc_auc_shoot_RF)

plt.legend(loc = 'lower right',fontsize=10,handlelength=3)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True positive Rate')
plt.xlabel('False positive Rate')

fig.savefig("./Image/ROC.jpg",dpi=600,bbox_inches='tight')


# %%
# %% SHAP 交互作用强度
shap_interaction_values = shap.TreeExplainer(model=rf_model_shoot).shap_interaction_values(X_train)[1]

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
df_temp2.columns = X_train.columns[inds]
df_temp2.index = X_train.columns[inds]

h=sns.heatmap(df_temp2, cmap='viridis', square=True, center=0.2,
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
shap.dependence_plot((feature1, feature2),shap_interaction_values,X_train,ax=ax,show=False,
                    display_features=X_train)


fig.savefig("./Image/SHAP_Interact_Nonc_IS.jpg",dpi=600,bbox_inches='tight')

# %% SHAP_Interact_MONP_Clay
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
feature1 = 'Nanoparticle concentration (mg/L)'
feature2 = 'Zeta potential (mV)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X_train,ax=ax,show=False,
                    display_features=X_train)


fig.savefig("./Image/SHAP_Interact_Nonc_zeta.jpg",dpi=600,bbox_inches='tight')


# %% SHAP_Interact_MONP_Clay
fig, ax = plt.subplots(figsize=(4, 3))
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
feature2 = 'NaCl concentration (M)'
feature1 = 'Zeta potential (mV)'
shap.dependence_plot((feature1, feature2),shap_interaction_values,X_train,ax=ax,show=False,
                    display_features=X_train)


fig.savefig("./Image/SHAP_Interact_Nacl_zeta.jpg",dpi=600,bbox_inches='tight')
# %%
