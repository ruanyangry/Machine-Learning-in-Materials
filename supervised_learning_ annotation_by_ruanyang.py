#Code by Daniel V. Samarov, Ph.D. 
#National Instiute of Standards and Technology 
#Inofrmation Technology Laboratory
#Statistical Engineering Division 

#Prepared for day 2 of the 2017 "Machine Learning for Materials Research" bootcamp and workshop at the U. of Maryland Nanocenter

# We start off by loading the modules and libraries that we will be using. 
# Note, there are various "best practices" that are out there for this, 
# including selecting the specific functions with a module that will used 
# rather than selecting all of (say) numpy. We won't worry about this too much
# here, but it's something to keep in mind.


#%%
# numpy is a module for a variety of numeric operations and helper functions, 
# but also linear algebra
import numpy as np
# sklearn is a general machine learning and statistics module, for our purposes
# we grab the linear regression module
from sklearn.linear_model import LinearRegression, ElasticNetCV, \
Ridge, Lasso, LassoCV, LogisticRegression

# LinearRegression: 线性回归
# ElasticNetCV：是一种使用L1和L2先验作为正则化矩阵的线性回归模型。就是同时使用L1正则和L2正则作用于线性模型
# Ridge/RidgeCV：L2正则
# Lasso/LassoCV：L1正则
# LogisticRegression：逻辑回归分类

from sklearn.neural_network import MLPClassifier

# MLPClassifier：多层感知机中的分类器

from sklearn import svm

# svm：支持向量机

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# discriminant_analysis：判别分析
#LinearDiscriminantAnalysis：线性判别分析是一种分类模型，它通过在k维空间选择一个投影超平面，使得不同类别在该超平面上的投影之间的距离尽可能近，同时不同类别的投影之间的距离尽可能远，在LDA中，我们假设每一个类别的数据服从高斯分布，且具有相同协方差矩阵ΣΣ。具备一定的数据降维功能
#QuadraticDiscriminantAnalysis：似于LDA，不同的地方是它可以形成非线性的边界，并且不同的类所属的高斯分布具有不同的协方差矩阵

# Get performance metrics 评价性能
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, \
confusion_matrix, roc_curve

# sklearn.metrics 保存了模型性能的评价方法
# accuracy_score：分类准确率分数是指所有分类正确的百分比
# recall_score：召回率  = 提取出的正确信息条数 /样本中的信息条数
# roc_curve：反映灵敏性和特效性连续变量的综合指标
#            纵坐标=真正例率（也就是灵敏度）（True Positive Rate,TPR）
#            TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）
#            横坐标=假正例率（1-特效性）（False Positive Rate,FPR）
#            FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
# 模型的准确率 = ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）
# AUC：计算AUC值，其中x,y分别为数组形式，根据(xi,yi)在坐标上的点，生成的曲线，然后计算AUC值
# roc_auc_score：直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，
#                中间过程的roc计算省略
# confusion_matrix：混淆矩阵

# pandas has some similar functionality to numpy but provides some improved
# data processing steps functionality.
# pandas 在数据清洗方面会更加的优秀

import pandas as pd

# scipy is a general purpose scientific computing module
# scipy 主要是进行一些计算操作
# 从scipy中导入了 svd 奇异值分解求法

from scipy.linalg import svd
from scipy.stats import f

# matplotlibn is the primary plotting library for Python
# 画图主要还是基于 matplotlib

import matplotlib.pyplot as plt
# Plot style
plt.style.use('ggplot')
# Function for loading MATLAB files
from scipy.io import loadmat

# Another plotting library
import seaborn as sns
#%%


# Start by loading our regression example - start with the training data
# Our files directory
data_dir = '/home/dan/Dropbox/MLMR_bootcamp/Supervised_Learning/examples.xlsx'
# NOTE: the constant term b0 is included in the matrix, for most the of the 
# models in Python we can pass this in without issue, we'd just need to specify
# that an intercept term should NOT be fit. For our purposes we're going to
# drop that first column since for some of our functions it creates some issues
# (more on this below).

# 从excel中读取 训练和测试数据
# correlation matrix
# formation energies

x_train = pd.read_excel(data_dir, sheetname='Training correlation matrix',
                        header=None).values
y_train = pd.read_excel(data_dir,sheetname='Training formation energies',
                        header=None).values[:, 0]

# Testing
x_test = pd.read_excel(data_dir, sheetname='Test correlation matrix',
                       header=None).values
y_test = pd.read_excel(data_dir, sheetname='Test formation energies',
                       header=None).values[:, 0]
#%%
                       
# range() creates an ordered list starting with 0, try it out. Python uses 0
# indexing in it's indexing (unlike R and MATLAB). Using the pandas 
cols = range(1, x_train.shape[1])
x_train = x_train[:, cols]
x_test = x_test[:, cols]

# Look at structure of the data and it's distribution
# 下面列出的部分在 pandas 中是比较好计算的
print(x_train.shape)
print(x_train.mean(0))
print(x_train.var(0))
print(x_train.min(0))
print(x_train.max(0))

#%%
# Let's take a look at our training data
corr_fig = plt.figure()
corr_plot = corr_fig.add_subplot(1,1,1)
corr_plot.plot(x_train[:,0], y_train, 'o', label = 'Training')

#%%
# Next we can add some additional points from the test data set and see
# how well the model generalizes (not too bad given that we're looking at just
# one feature from 146)
corr_plot.plot(x_test[:,0], y_test, 'o', color='green', label = 'Testing')

# Can easily add a legend, note the "label" argument is automatically passed in
# 增加一些label标签 
corr_plot.legend(loc='upper right')
corr_plot.set_xlabel('Cluster Function 1')
corr_plot.set_ylabel('Formation Energy')
#%%
corr_fig.savefig('plots/CF1_vs_FE.png')

#%%
# FE：formation energy

# If we want to get a clearer visualization of what's going on can look at
# several group values versus FE. This is probably not as transparent as what
# you might find in R, but provides an illustration of some of the "more
# advanced" plotting features. There is a manual way to do this using similar
# syntax to MATPLOT's subplot. 
fig_data = plt.figure()

for i in range(9):
    ax_data = fig_data.add_subplot(3,3,i+1)
    ax_data.scatter(x_train[:,i], y_train, marker='o', s=10)
    plt.xlabel('CF = '+str(i+1))
    plt.ylabel('FE')
plt.tight_layout()

#%%
# Some alternative approaches that provide easy access to more aesthetically
# pleasing plots (but are syntactically a bit less clear)
n_cols = range(9)
x_col = x_train[:,n_cols].T.reshape((x_train[:,n_cols].shape[0]*len(n_cols), 1))[:,0]
g_col = sum([[i]*x_train.shape[0] for i in n_cols], [])
y_col = sum([y_train.tolist()]*len(n_cols), [])
data_df = pd.DataFrame({'Cluster Function': x_col, 'CF':g_col, 'FE': y_col})
scatplt_mat = sns.FacetGrid(data_df, col = 'CF', col_wrap=int(np.sqrt(len(n_cols))), 
                  sharex=False)
scatplt_mat.map(sns.regplot, 'Cluster Function', 'FE', fit_reg = False, scatter_kws={'s':10})

#%%
scatplt_mat.savefig('plots/scatplt_mat.png')

#%%
# Let's look at a simple linear model fit to the data. 
# The structure for fitting models in Python is a bit different then what
# you might be accustomed to in R, MATLAB, etc. Not all models are fit in 
# the same way, but they tend to follow a similar syntax so it's pretty easy
# to get things up and running once you have the hang of it.
# 首先都是导入模型，其次是拟合，最后是预测

lm1 = LinearRegression()
lm1.fit(x_train[:, [0]], y_train)
p1 = lm1.predict(x_train[:, [0]])
#%%

# Let's take a look at what information the "ols" object contains
# 线性回归一般是得到 系数+截距

print(lm1.coef_)
print(lm1.intercept_)

#%%
# Check the above
# np.corrcoef()这个是进行了关联分析

rho = np.corrcoef(x_train[:,0],y_train)[0,1]
sy = np.sqrt(y_train.var())
sx = np.sqrt(x_train[:,0].var())
# Slope
print(rho * sy / sx)
# Intercept
print(np.mean(y_train - x_train[:,0] * rho * sy / sx))


#%%
# Add the regression line to the plot
corr_plot.plot(x_train[:,0], p1, color='r')

#%%
corr_fig.savefig('plots/CF1_vs_FE_fit.png')

#%%
# Looking at the residuals we can see that there distribution isn't as 
# randomly distributed as we might like, suggesting that our model isn't quite
# capturing everything that's going on. This is of course to be expected since
# we're just using one of the available predictors.
# y_train - p1 表示的是实际值和预测值之间的差别，即残差

fig_plot, res_plot = plt.subplots()
res_plot.plot(y_train, y_train - p1, "o")
res_plot.axhline(0, color="r")
res_plot.set_xlabel('Observed')
res_plot.set_ylabel('Residual')

#%%
fig_plot.savefig('plots/CF1_resid.png')

#%%
# 不是很理解这里面的 p > n 是个什么意思？
# Recall from our discussion in class, when p > n the problem becomes "ill-posed", 
# as such specific considerations need to be taken into account. Before getting 
# to that let's try selecting multiple columns to get used to the syntax of
# both data manipulation, model fitting and output
sub_cols = [0, 10, 30, 73]
lm2 = LinearRegression()
# Fit the model (you'll notice as we go through the examples that the structure
# of fitting the model is fairly consistent)
# 这里面应该是同时拟合了多组数据（4组数据）

lm2.fit(x_train[:, sub_cols], y_train)

# Get predictions
p2 = lm2.predict(x_test[:, sub_cols])

# Look at the resulting coefficient estimate
print(lm2.coef_)
print(lm2.intercept_)

#%%
# NOTE: as stated when p > n standard regression runs into some problems
# which requires care. in particular we note that the
# LinearRegression function will still fit a model irrespective of the fact
# that the inverse does not exist. To account for this it is likely using a 
# "pseudo-inverse". To see what the issue is more explicitly let's take a look 
# at the eigen values （本征值）
# NOTE: For this example we need to explicitly include the intercept term in
# our design matrix
x_tr = np.hstack((np.ones((x_train.shape[0],1)), x_train))
e_vecs_l, e_vals, e_vecs_r = svd(x_tr.T.dot(x_tr))
eig_fig, eig_plot = plt.subplots()
eig_plot.plot(e_vals)
eig_plot.set_xlabel('Index')
eig_plot.set_ylabel('Value')
eig_plot.set_title('Eigen values of the covariance matrix')

# covariance matrix：协方差矩阵

#%%
eig_fig.savefig('plots/eigen.png')


#%%
# And to see how this effects the prediction let's work through the rest of the
# least squares solution 最小二乘法
# First compute (x^Tx)^-1
print(1/e_vals)
e_vals_trunc_inv = 1/e_vals
e_vals_trunc_inv[e_vals <= 1e-11] = 0.0
xtx_inv = e_vecs_l.dot(np.diag(e_vals_trunc_inv)).dot(e_vecs_r)

# Then multiply the latter by x^Ty
b_full = xtx_inv.dot(x_tr.T).dot(y_train)
x_te = np.hstack((np.ones((x_test.shape[0],1)), x_test))
p_full = x_te.dot(b_full)

#%%
# So, now let's take a look at what happens when we fit a regression without
# accounting for the fact that p > n using sklearn's model. This should be
# fairly similar to our results above. 
# 这个的意思表示在进行线性拟合数据的时候不考虑截距带来的影响

lm_tot = LinearRegression(fit_intercept=False)
lm_tot.fit(x_tr, y_train)
p_full2 = lm_tot.predict(x_te)

#%%
# Fit with testing data, if our model is doing a good job the hope is that
# the difference between coefficient estimates isn't too big
lm_tot2 = LinearRegression(fit_intercept=False)
lm_tot2.fit(x_te, y_test)

# 直接计算均方误差
print(mean_squared_error(lm_tot2.coef_, lm_tot.coef_))

#%%
# Next look at a couple of performance metrics against training and testing. 
# First take a look at how well our model fit the observed (training) data 
# when using one CF
# 评价模型的性能：performance metrics

print(r2_score(y_train, p1))
print(mean_squared_error(y_train, p1))

#%%
# Of course what's of more interest to us though is the performance of our model on the
# test data set, this tells us how well it generalizes
# 比较预测值与实际值

p1_test = lm1.predict(x_test[:,[0]])
print(r2_score(y_test, p1_test))
print(mean_squared_error(y_test, p1_test))

#%%
# Next let's look at the performance with 4 features
# 评价4组数据的性能

print( r2_score(y_test, p2))
print( mean_squared_error(y_test, p2))

#%%
# And the full model
print(r2_score(y_test, p_full))
print(mean_squared_error(y_test, p_full))

#%%
# NOTE: Depending on what your objectives are the sklearn interface may not
# quite capture all your needs. For example, in the R function "lm" we get
# a number of helpful summary statistics on the model fit (e.g. p-values,
# F-statistics, etc.). There are some ways of getting at this but it isn't the
# default. That said something like the F-statistic can easily be computed 
# manually
# 不是很理解这里面计算的是什么数据

mse1 = mean_squared_error(y_train, p1)
df1 = len(lm1.coef_) + 1
mse4 = mean_squared_error(y_train, lm2.predict(x_train[:,sub_cols]))
df2 = len(lm2.coef_) + 1
n = x_train.shape[0]
f_stat = ((mse1 - mse4)/(df2 - df1))/(mse4/(n - df2))
print(f_stat)
print(1 - f.cdf(f_stat, df2 - df1, n - df2))

#%%
# Next let's try ridge regression and compare results. First we look at what 
# happens to the eigenvalues when we add an off set
# ridge regression：对应的就是L2正则化
# L0，L1，L2中均存在一个 alpha 参数，是一个超参数

print(1/(e_vals + 10.0))

#%%
lm_ridge = Ridge(alpha = 10.0)

# 对所有的数据进行了训练
lm_ridge.fit(x_train, y_train) 

p_ridge = lm_ridge.predict(x_test)
print(r2_score(y_test, p_ridge))

# 这里面使用 np.sqrt()函数的意思在哪里？
print(np.sqrt(mean_squared_error(y_test, p_ridge)))

#%%
lm_ridge2 = Ridge(alpha = 10.0)
lm_ridge2.fit(x_test, y_test)

#%%
# See how consistent the estimates are
# 相当于是对训练数据和测试数据同时使用 Ridge()方法进行拟合，主要是想评价拟合出来的函数系数

print(mean_squared_error(lm_ridge.coef_, lm_ridge2.coef_))

#%%
# How does this impact the coefficient estimates?
comp_fig = plt.figure()
comp_plot = comp_fig.add_subplot(1,1,1)
comp_plot.bar(np.arange(x_train.shape[1]), lm_tot.coef_[1:], label='LinReg')
comp_plot.bar(np.arange(x_train.shape[1]), lm_ridge.coef_, label='Ridge')
comp_plot.legend()
comp_plot.set_title('Linear Regression vs. Ridge Estimates')

#%%
comp_fig.savefig('plots/LR_vs_RIDGE.png')

#%%
# We can also try out a number of alpha values and see how they each perform.
# Create a series of alpha values to looks at
# 测试 alpha 的值，其实是在寻找合适的超参数
alpha = np.arange(.1, 20.1, .1)
# An array to store the results
ridge_res = np.empty((0, 3))
for a in alpha:
    lm_ridge = Ridge(alpha=a)
    lm_ridge.fit(x_train, y_train)
    p_ridge = lm_ridge.predict(x_test)
    r2 = r2_score(y_test, p_ridge)
    rmse = mean_squared_error(y_test, p_ridge)   # 计算 rmse
    # These lines store the associated results into a 2D array
    res_a = [[a, r2, rmse]]
    ridge_res = np.append(ridge_res, res_a, axis = 0)
# Look at the results
print(ridge_res)

#%%

# Let's now plot the results (and a slightly more complex plotting example)
ridge_fig = plt.figure()
ridge_r2 = ridge_fig.add_subplot(1,1,1)
ridge_r2_line = ridge_r2.plot(ridge_res[:, 0], ridge_res[:, 1], label='R2')
ridge_r2.set_ylabel('R2')
ridge_rmse = ridge_r2.twinx()
ridge_rmse_line = ridge_rmse.plot(ridge_res[:, 0], ridge_res[:, 2], 
                                  label = 'MSE', color='blue')
ridge_r2.set_xlabel('gamma')
ridge_rmse.set_ylabel('MSE')
ridge_line = ridge_r2_line + ridge_rmse_line
ridge_labs = [l.get_label() for l in ridge_line]
ridge_r2.legend(ridge_line, ridge_labs, loc='center right')

#%%
ridge_fig.savefig('plots/ridge_sol_path.png')
#%%

# Compare results of standard and ridge regression
# 这个 dataframe 应该是自己生成的吧
res_df = pd.DataFrame({'Method':['Standard', 'Ridge'], 
'MSE':[mean_squared_error(y_test, p_full), mean_squared_error(y_test, p_ridge)]})

#%%
# From these results we can see that ridge regression provides an improvement
# over standard regression. 
# 这里的含义是岭回归的效果是优于标准回归的

# Next we consider the LASSO and Elastic Net models
# Traing the lasso model
# Lasso model: L1正则

lasso = Lasso(fit_intercept=True, alpha = 0.1)
lasso.fit(x_train, y_train)
p_lasso = lasso.predict(x_test)
print(r2_score(y_test, p_lasso))
print(mean_squared_error(y_test, p_lasso))
print(sum(lasso.coef_ == 0))

#%%

# Next fitting the lasso and finding the optimal regularization using
# built in cross-validation provides further improvement over the LASSO
# 下面应该是基于交叉验证来测试超参数了

lasso_cv = LassoCV(fit_intercept=True, n_alphas=100, normalize=False)
lasso_cv.fit(x_train, y_train)
p_lasso_cv = lasso_cv.predict(x_test)
print(r2_score(y_test, p_lasso_cv))
print(mean_squared_error(y_test, p_lasso_cv))
print(sum(lasso_cv.coef_ == 0) )

#%%
# Let's visualize the solution path
# 这里面是将上述的超参数求解过程看成了是 solution path 了

lassop = lasso_cv.path(x_train, y_train)
lassop_fig = plt.figure()
lassop_plot = lassop_fig.add_subplot(1,1,1)
lassop_plot.plot(np.log(lassop[0]),lassop[1].T)
lassop_plot.set_xlabel('lambda value (log scale)')
lassop_plot.set_ylabel('Coefficient estimate value')
lassop_plot.set_title('lasso solution path')

#%%
lassop_fig.savefig('plots/lasso_path.png')
#%%
# Next let's try some different values of l1_ratio (which then incorporates
# the l2 constraint)
# Next fitting the lasso and finding the optimal regularization using
# built in cross-validation provides further improvement over the LASSO
# ElasticNetCV：包含了 L1和L2 正则化
en_cv = ElasticNetCV(fit_intercept=True, n_alphas=100, normalize=False, 
                  l1_ratio=0.01)
en_cv.fit(x_train, y_train)
p_en_cv = en_cv.predict(x_test)
print(r2_score(y_test, p_en_cv))
print(mean_squared_error(y_test, p_en_cv))
print(sum(en_cv.coef_ == 0))

#%%
enp = en_cv.path(x_train, y_train)
enp_fig = plt.figure()
enp_plot = enp_fig.add_subplot(1,1,1)
enp_plot.plot(np.log(enp[0]),enp[1].T)  # 这里面使用 np.log()函数的意思在哪里？
enp_plot.set_xlabel('lambda value (log scale)')
enp_plot.set_ylabel('Coefficient estimate value')
enp_plot.set_title('EN solution path')

#%%
enp_fig.savefig('plots/en_path.png')

#%%
# Update results
res_df.loc[2,:] = [mean_squared_error(y_test, p_lasso_cv), 'lasso']
res_df.loc[3,:] = [mean_squared_error(y_test, p_en_cv), 'EN']
print(res_df)
#%%
# Next we take a look at a model motivated by the underlying physics associated
# with the material. A key takeaway here is how much better this approach does
# than an "out of the box" machine learning technique.
ds_reg = pd.read_excel(data_dir, sheetname = 'Distance and Shape regularizer', 
                        header=None).values

# Helper function for fitting our model
def mat_pow(x, p):
    u, d, v = svd(x)
    return u.dot(np.diag(d**p)).dot(v)
    
sq_ds_reg = mat_pow(ds_reg, 0.5)
x_train_aug = np.vstack((x_tr, sq_ds_reg))

# Create augmented matrix 生成一个增强矩阵
y_train_aug = np.concatenate([y_train, np.zeros(x_train_aug.shape[1])])
en_ds_reg = LinearRegression(fit_intercept=False, normalize=False)
en_ds_reg.fit(x_train_aug, y_train_aug)
# Predict on test
p_ds_reg_test = en_ds_reg.predict(x_te)
print(mean_squared_error(y_test, p_ds_reg_test))
print(r2_score(y_test, p_ds_reg_test))


#%% 
###############################################################################
# Next we'll look at a hyperspectral imaging (HSI) application
# 这个例子演示的应该是在光谱领域的应用
###############################################################################
# Load HSI data

hsi = loadmat('data/HSIData.mat')
keys = hsi.keys()
# Take a look at what's contained in the object
print(keys)

# Grab the objects we'll be using
Y = hsi['Y']
Yarr = hsi['Yarr']
lab = hsi['labels']
lab2 = hsi['label2']
all_indx = np.arange(len(lab2))
train_indx = hsi['indx']
test_indx = np.delete(all_indx, train_indx)

#%%
# Take a look at the image
img_fig = plt.figure()
img_plot = img_fig.add_subplot(1,1,1)
img_plot.grid(False)
img_plot.imshow(Yarr[:,:,[100,50,10]])
img_plot.set_xticklabels(['']*Yarr.shape[0])
img_plot.set_yticklabels(['']*Yarr.shape[0])

#%%
img_fig.savefig('plots/kidney_image.png')
#%% 
# Grab two wavelengths and visualize the distribution of classes, kidney vs.
# other
# kidney： 肾脏

kid2_fig = plt.figure(figsize=(8,8))
kid2_class = kid2_fig.add_subplot(1,1,1)
kid2_mark = ['o', 'x']
kid2_col = ['red', 'blue']
kid2_leg = ['kidney', 'other']
for i in range(2):
    ind = lab2 == i + 1
    kid2_class.scatter(Y[ind,80], Y[ind,122], color=kid2_col[i], 
                       marker=kid2_mark[i], label = kid2_leg[i], alpha=0.5)
kid2_class.set_xlabel('Wavelength 81')
kid2_class.set_ylabel('Wavelength 123')
kid2_class.legend()
kid2_fig

#%%
kid2_fig.savefig('plots/kid_2class.png')

#%%
# Next lets plot the image with the 2 class labels. Since we're going to be
# reusing this set of plots let's write a simple function to simplify things
def plot_labels(l,s,cmap=plt.cm.hot_r):
        
    fig = plt.figure(figsize=(8,8))
    plot = fig.add_subplot(1,1,1)
    plot.grid(False)
    plot.imshow(np.reshape(l, s).T, cmap=cmap)
    plot.set_xticklabels(['']*s[0])
    plot.set_yticklabels(['']*s[0])
    plt.show()
    
    return fig

class_fig = plot_labels(lab2, (Yarr.shape[0],Yarr.shape[1]))

#%%
class_fig.savefig('plots/img_2class.png')

#%%
# Run LDA
lda = LinearDiscriminantAnalysis(solver='lsqr')
lda.fit(Y[train_indx,:], lab2[train_indx])
p_lda = lda.predict_proba(Y[test_indx,:]) # 预测概率值
l_lda = lda.predict(Y[test_indx,:])
confusion_matrix(lab2[test_indx], l_lda)  # 基于混合矩阵来判断降维是否合理？LinearDiscriminantAnalysis()有降维功能

#%% Visualize results
lda_fig = plot_labels(lda.predict(Y), (Yarr.shape[0],Yarr.shape[1]))

#%%
lda_fig.savefig('plots/lda_2class_pred.png')

#%%
# Try out sparse lda
slda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
slda.fit(Y[train_indx,:], lab2[train_indx])
p_slda = slda.predict_proba(Y[test_indx,:])
l_slda = slda.predict(Y[test_indx,:])
confusion_matrix(lab2[test_indx], l_slda)

#%%
slda_fig = plot_labels(slda.predict(Y), (Yarr.shape[0],Yarr.shape[1]))

#%%
slda_fig.savefig('plots/slda_2class_pred.png')

#%%
# Let's take a look at some performance metrics
aroc_lda = roc_auc_score(lab2[test_indx]-1, p_lda[:,1])
aroc_slda = roc_auc_score(lab2[test_indx]-1, p_slda[:,1])

# We can always look at plots of the results
roc_lda = roc_curve(lab2[test_indx]-1, p_lda[:,1])
roc_slda = roc_curve(lab2[test_indx]-1, p_slda[:,1])

roc_fig = plt.figure(figsize=(8,8))
roc_plot = roc_fig.add_subplot(1,1,1)
roc_plot.plot(roc_lda[0], roc_lda[1], label = 'LDA: AUC = ' + str(aroc_lda))
roc_plot.plot(roc_slda[0], roc_slda[1], label = 'SLDA: AUC = ' + str(aroc_slda))
roc_plot.legend()
roc_plot.set_xlabel('False Positive Rate')
roc_plot.set_ylabel('True Positive Rate')

#%%
roc_fig.savefig('plots/roc_lda_slda.png')

#%%
# Next running logistic regression
# 使用逻辑回归进行数据的拟合

lr = LogisticRegression(C=1e6, fit_intercept=False)
lr.fit(Y[train_indx,:], lab2[train_indx])
p_lr = lr.predict_proba(Y[test_indx,:]) # 逻辑回归方法也可以得到对应的概率密度分布
l_lr = lr.predict(Y[test_indx,:])
aroc_lr = roc_auc_score(lab2[test_indx]-1, p_lr[:,1])

#%% Can also try sparse logistic regression
# 这个是在逻辑回归中增加了惩罚项，使用L1正则操作
# 增加了惩罚项与 sparse 是什么关系？

slr = LogisticRegression(C=1.0, fit_intercept=False, penalty='l1')
slr.fit(Y[train_indx,:], lab2[train_indx])
p_slr = slr.predict_proba(Y[test_indx,:])
l_slr = slr.predict(Y[test_indx,:])
aroc_slr = roc_auc_score(lab2[test_indx]-1, p_slr[:,1])

#%% And visualize LR
lr_fig = plot_labels(lr.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
#%%
lr_fig.savefig('plots/lr_2class_pred.png')

#%% SLR
slr_fig = plot_labels(slr.predict(Y), (Yarr.shape[0],Yarr.shape[1]))

#%%
slr_fig.savefig('plots/slr_2class_pred.png')

#%%
# And look at the roc curves
# We can always look at plots of the results
roc_lr = roc_curve(lab2[test_indx]-1, p_lr[:,1])
roc_slr = roc_curve(lab2[test_indx]-1, p_slr[:,1])

roc_lrfig = plt.figure(figsize=(8,8))
roc_lrplot = roc_lrfig.add_subplot(1,1,1)
roc_lrplot.plot(roc_lda[0], roc_lda[1], 
                label = 'LR: AUC = ' + str(aroc_lr))
roc_lrplot.plot(roc_slda[0], roc_slda[1], 
                label = 'SLR: AUC = ' + str(aroc_slr))
roc_lrplot.legend()
roc_lrplot.set_xlabel('False Positive Rate')
roc_lrplot.set_ylabel('True Positive Rate')

#%%
roc_lrfig.savefig('plots/roc_lr_slr.png')

#%% 
# Take a look at the coefficient estimates
coefs_fig = plt.figure()
lr_coefs_plot = coefs_fig.add_subplot(2, 1, 1)
lr_coefs_plot.bar(np.arange(Y.shape[1]), lr.coef_[0])
lr_coefs_plot.set_title('Logistic Regression')
lr_coefs_plot.set_ylabel('Value')

slr_coefs_plot = coefs_fig.add_subplot(2, 1, 2)
slr_coefs_plot.bar(np.arange(Y.shape[1]), slr.coef_[0])
slr_coefs_plot.set_title('Sparse Logistic Regression')
slr_coefs_plot.set_ylabel('Value')
slr_coefs_plot.set_xlabel('Wavelength')

#%%
coefs_fig.savefig('plots/lr_slr_coef.png')

#%%
# These models can also be easily extended to handle multiple classes
# 将模型扩展到了多分类问题
mlr = LogisticRegression(C=1e6, fit_intercept=False, multi_class='multinomial',
                         solver='lbfgs')
mlr.fit(Y[train_indx,:], lab[train_indx])
p_mlr = mlr.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx], p_mlr))

#%%
# Sparse version
smlr = LogisticRegression(C=1000, fit_intercept=False,
                          penalty='l1', solver='liblinear')
smlr.fit(Y[train_indx,:], lab[train_indx])
p_smlr = smlr.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx], p_smlr))

#%% 
# Visualize the results
mclass_fig = plot_labels(lab, (Yarr.shape[0],Yarr.shape[1]), 
                         cmap = plt.cm.Blues_r)

#%%
mclass_lr_fig = plot_labels(mlr.predict(Y), (Yarr.shape[0],Yarr.shape[1]), 
                            cmap = plt.cm.Blues_r)

#%%
mclass_slr_fig = plot_labels(smlr.predict(Y), (Yarr.shape[0],Yarr.shape[1]), 
                             cmap = plt.cm.Blues_r)

#%%
mclass_fig.savefig('plots/mclass.png')
mclass_lr_fig.savefig('plots/lr.png')
mclass_slr_fig.savefig('plots/slr.png')

#%%
# Try fitting SVM
# 尝试使用支持向量机来拟合数据
# 其中核函数使用的是线性

lsvc = svm.SVC(kernel='linear')
lsvc.fit(Y[train_indx,:], lab2[train_indx])
l_lsvc = lsvc.predict(Y[test_indx,:])
lsvc_fig = plot_labels(lsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
confusion_matrix(lab2[test_indx], l_lsvc)
#%%
# 这里面设定了核函数为 径向基核函数

ksvc = svm.SVC(kernel='rbf', gamma = 0.5)
ksvc.fit(Y[train_indx,:], lab2[train_indx])
l_ksvc = ksvc.predict(Y[test_indx,:])
ksvc_fig = plot_labels(ksvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
confusion_matrix(lab2[test_indx], l_ksvc)

#%%
# Save images
lsvc_fig.savefig('plots/lsvc.png')
ksvc_fig.savefig('plots/ksvc.png')

#%%
# Multi-class SVM
# 使用多分类的支持向量机

# Try fitting SVM
lmsvc = svm.SVC(kernel='linear', C=1000)
lmsvc.fit(Y[train_indx,:], lab[train_indx])
l_lmsvc = lmsvc.predict(Y[test_indx,:])
lmsvc_fig = plot_labels(lmsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]), 
                        plt.cm.Blues_r)
pd.DataFrame(confusion_matrix(lab[test_indx], l_lmsvc))
#%%
kmsvc = svm.SVC(kernel='rbf', gamma = 0.5, C=1000)
kmsvc.fit(Y[train_indx,:], lab[train_indx])
l_kmsvc = kmsvc.predict(Y[test_indx,:])
kmsvc_fig = plot_labels(kmsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]), 
                        plt.cm.Blues_r)
pd.DataFrame(confusion_matrix(lab[test_indx], l_kmsvc))

#%%
# Save images
lmsvc_fig.savefig('plots/lmsvc.png')
kmsvc_fig.savefig('plots/kmsvc.png')

#%% 
# Next we try fitting a simple NN
# 使用多层感知机来拟合数据
# 只设定了隐含层的层数（这个描述应该是不正确）

mlp = MLPClassifier(hidden_layer_sizes=(50,50,))
mlp.fit(Y[train_indx,:], lab[train_indx])
l_mlp = mlp.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx], l_mlp))

mlp_fig = plot_labels(mlp.predict(Y), (Yarr.shape[0],Yarr.shape[1]), 
                        plt.cm.Blues_r)

#%%
mlp_fig.savefig('plots/mlp.png')