# _*_ coding:utf-8 _*_

'''
Author: Ruan Yang
Email: ruanyang_njut@163.com

This code reference https://github.com/delton137/Machine-Learning-for-Materials-Research-Bootcamp
I have made modifications according to my requirements, mainly to \
understand the research of common machine learning algorithms in the \
field of materials.

This code focuse on supervised learning.
'''

# First import sklearn,numpy,scipy and matplotlib modules.

import numpy as np

# Import linear model in sklearn.

from sklearn.linear_model import LinearRegression,ElasticNetCV,Ridge,Lasso,\
LassoCV,LogisticRegression

# ElasticNetCV: contain L1 and L2 regularization
# Ridge: L2 regularization (RidgeCV)
# Lasso and LaasoCV: L1 regularization
# LogisticRegression: classifier methods.

# import neural network methods named Multi-layer Perceptron classifier

from sklearn.neural_network import MLPClassifier

# import support vector machine

from sklearn import svm

# import discriminant_analysis, Includes data dimensionality reduction

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# import model performance metrics

from sklearn.metrics import r2_score,mean_squared_error,roc_auc_score,\
confusion_matrix,roc_curve

# import pandas

import pandas as pd

# import scipy
# svd: singular value decomposition

from scipy.linalg import svd
from scipy.stats import f

# import matplotlib

import matplotlib.pyplot as plt

# plot style

plt.style.use("ggplot")

# another plotting library

import seaborn as sns

# from scipy import function for MATLAB files

from scipy.io import loadmat

# set the paths for input data files.

paths=r"C:\Users\RY\Desktop\Machine-Learning-for-Materials-Research-Bootcamp-master\Supervised_Learning"

# xlsx file

xlsx=paths+"\examples.xlsx"

# mat file

matdata=paths+"\HSIData.mat"

# load the train and test correlation matrix and formation energies

x_train=pd.read_excel(xlsx,sheet_name="Training correlation matrix",header=None).values
y_train=pd.read_excel(xlsx,sheet_name="Training formation energies",header=None).values[:,0]

x_test=pd.read_excel(xlsx,sheet_name="Test correlation matrix",header=None).values
y_test=pd.read_excel(xlsx,sheet_name="Test formation energies",header=None).values[:,0]

# Add columns index

cols=range(1,x_train.shape[1])
x_train=x_train[:,cols]
x_test=x_test[:,cols]

# print basic information about x_train data
# mean,var,min and max function focused on each columns
# 0: represent column

print("#--------------------------------------")
print(x_train.shape)
print(x_train.mean(0))
print(x_train.var(0))
print(x_train.min(0))
print(x_train.max(0))
print("#--------------------------------------")
print("\n")

# visualize the training data

corr_fig=plt.figure()
corr_plot=corr_fig.add_subplot(1,1,1)
corr_plot.plot(x_train[:,0],y_train,"o",label="Training")
corr_plot.plot(x_test[:,0],y_test,'o',color="green",label="Testing")
corr_plot.legend(loc="best")
corr_plot.set_xlabel("Cluster Function")
corr_plot.set_ylabel("Formation Energy")
plt.savefig('CF1_vs_FE.jpg',dpi=300)
plt.show()

# visualize with more datas

fig_data=plt.figure()

for i in range(9):
	ax_data=fig_data.add_subplot(3,3,i+1)
	ax_data.scatter(x_train[:,i],y_train,marker="o",s=10)
	plt.xlabel("CF = "+str(i+1))
	plt.ylabel("FE")
plt.tight_layout()
plt.savefig("CF1-9_vs_FE.jpg",dpi=300)
plt.show()

# some more aesthetically pleasing plots
# I have questions about some of these Settings
# Using pandas dataframe format

n_cols=range(9)
x_col=x_train[:,n_cols].T.reshape((x_train[:,n_cols].shape[0]*len(n_cols), 1))[:,0]
g_col=sum([[i]*x_train.shape[0] for i in n_cols], [])
y_col=sum([y_train.tolist()]*len(n_cols), [])
data_df=pd.DataFrame({'Cluster Function': x_col, 'CF':g_col, 'FE': y_col})
scatplt_mat = sns.FacetGrid(data_df, col = 'CF', col_wrap=int(np.sqrt(len(n_cols))), 
                  sharex=False)
scatplt_mat.map(sns.regplot,'Cluster Function', 'FE', fit_reg = False, scatter_kws={'s':10})
scatplt_mat.savefig("scatplt_mat.jpg",dpi=300)
#scatplt_mat.show()

# LinearRegression

lm1=LinearRegression()
lm1.fit(x_train[:,[0]],y_train)
p1=lm1.predict(x_train[:,[0]])

print("#--------------------------------------")
print("linear model coefficient = %.4f"%(lm1.coef_))
print("linear model intercept = %.4f"%(lm1.intercept_))
print("#--------------------------------------")
print("\n")

# Use mathematical methods for validation

rho=np.corrcoef(x_train[:,0],y_train)[0,1]
sx=np.sqrt(x_train[:,0].var())
sy=np.sqrt(y_train.var())

# Slope == linear model coefficient

print("#--------------------------------------")
print("Slope = %.4f"%(rho*sy/sx))
print("Intercept = %.4f"%(np.mean(y_train - x_train[:,0] * rho * sy / sx)))
print("#--------------------------------------")
print("\n")

# Add the regression line to the plot

fit_fig=plt.figure()
fit_plot=fit_fig.add_subplot(1,1,1)
fit_plot.plot(x_train[:,0],y_train,"o",label="Training")
fit_plot.plot(x_train[:,0],p1,color='r',label="Predict")
plt.savefig("CF1_vs_FE_fit_1.jpg",dpi=300)
plt.show()

# Plot the residuals between Training and predict data

fig_plot,res_plot=plt.subplots()
res_plot.plot(y_train,y_train-p1,"o")
res_plot.axhline(0,color="r")
res_plot.set_xlabel("Observed")
res_plot.set_ylabel("Residual")
fig_plot.savefig("CF1_residual.jpg",dpi=300)
fig_plot.show()

# Fitting more columns data

sub_cols=[0,10,30,73]
lm2=LinearRegression()
lm2.fit(x_train[:,sub_cols],y_train)
p2=lm2.predict(x_test[:,sub_cols])

print("#--------------------------------------")
#print("linear model coefficient = %.4f"%(lm2.coef_))
print(lm2.coef_)
print(lm2.intercept_)
#print("linear model intercept = %.4f"%(lm2.intercept_))
print("#--------------------------------------")
print("\n")

# "pseudo-inverse" 

x_tr=np.hstack((np.ones((x_train.shape[0],1)),x_train))
e_vecs_l, e_vals, e_vecs_r = svd(x_tr.T.dot(x_tr))
eig_fig, eig_plot = plt.subplots()
eig_plot.plot(e_vals)
eig_plot.set_xlabel('Index')
eig_plot.set_ylabel('Value')
eig_plot.set_title('Eigen values of the covariance matrix')
plt.savefig("eigen.jpg",dpi=300)
plt.show()

# Least squares solution

print(1/e_vals)
e_vals_trunc_inv=1/e_vals
e_vals_trunc_inv[e_vals <= 1e-11] =0.0
xtx_inv=e_vecs_l.dot(np.diag(e_vals_trunc_inv)).dot(e_vecs_r)

# Then multiply the latter by x^Ty

b_full = xtx_inv.dot(x_tr.T).dot(y_train)
x_te = np.hstack((np.ones((x_test.shape[0],1)), x_test))
p_full = x_te.dot(b_full)

# Fit model not with intercept

lm_tot=LinearRegression(fit_intercept=False)
lm_tot.fit(x_tr,y_train)
p_full2=lm_tot.predict(x_te)

lm_tot2=LinearRegression(fit_intercept=False)
lm_tot2.fit(x_te,y_test)

# calculated mean squared error

print("#--------------------------------------")
print(mean_squared_error(lm_tot2.coef_,lm_tot.coef_))
print("#--------------------------------------")
print("\n")

# model performance metrics

print("#--------------------------------------")
print("LinearRegression with intercept inputdata 1 column")
print("Performace on training data")
print(r2_score(y_train,p1))
print(mean_squared_error(y_train,p1))
print("\n")
print("Performance on testing data")

p1_test=lm1.predict(x_test[:,[0]])
print(r2_score(y_test,p1_test))
print(mean_squared_error(y_test,p1_test))
print("#--------------------------------------")
print("\n")

print("#--------------------------------------")
print("LinearRegression with intercept inputdata 4 column")
print(r2_score(y_test,p1))
print(mean_squared_error(y_test,p1))
print("#--------------------------------------")
print("\n")

print("#--------------------------------------")
print("Full model")
print(r2_score(y_test,p_full))
print(mean_squared_error(y_test,p_full))
print("#--------------------------------------")
print("\n")

# F-statistic import by scipy.stats name f

mse1=mean_squared_error(y_train,p1)
df1=len(lm1.coef_)+1
mse4=mean_squared_error(y_train,lm2.predict(x_train[:,sub_cols]))
df2=len(lm2.coef_)+1
n=x_train.shape[0]
f_stat=((mse1 - mse4)/(df2 - df1))/(mse4/(n - df2))
print("#--------------------------------------")
print(f_stat)
print(1-f.cdf(f_stat,df2-df1,n-df2))
print("#--------------------------------------")
print("\n")

# Ridge Regression
# Hyperparameter: alpha

print("#--------------------------------------")
print(1/(e_vals+10.0))
print("#--------------------------------------")
print("\n")

# Training model using all the data in x_train

lm_ridge=Ridge(alpha=10.0)
lm_ridge.fit(x_train,y_train)
p_ridge=lm_ridge.predict(x_test)

print("#--------------------------------------")
print(" Ridge regression result")
print(r2_score(y_test,p_ridge))
print(np.sqrt(mean_squared_error(y_test,p_ridge)))


# Fitting test data

lm_ridge2=Ridge(alpha=10.0)
lm_ridge2.fit(x_test,y_test)

print("\n")
print("Compare training and testing data fitting result")
print(mean_squared_error(lm_ridge.coef_,lm_ridge2.coef_))
print("#--------------------------------------")
print("\n")

# Visualize the result to estimated How does this impact the coefficient estimate

comp_fig=plt.figure()
comp_plot=comp_fig.add_subplot(1,1,1)
comp_plot.bar(np.arange(x_train.shape[1]),lm_tot.coef_[1:],label="LinearRegression")
comp_plot.bar(np.arange(x_train.shape[1]),lm_ridge.coef_,label="Ridge regression")
comp_plot.legend(loc="best")
plt.savefig("LR_vs_Ridge.jpg",dpi=300)
plt.show()

# Testing hyperparameter alpha in Ridge Regression

alpha=np.arange(.1,20.1,.1)

# An array to store the results

ridge_res = np.empty((0,3))

for a in alpha:
	lm_ridge=Ridge(alpha=a)
	lm_ridge.fit(x_train,y_train)
	p_ridge=lm_ridge.predict(x_test)
	r2=r2_score(y_test,p_ridge)
	rmse=mean_squared_error(y_test,p_ridge)
	res_a=[[a,r2,rmse]]
	ridge_res=np.append(ridge_res,res_a,axis=0)
	
print("#--------------------------------------")
print("Testing a serials alpha value")
print(ridge_res)
print("#--------------------------------------")
print("\n")

# Plot the testing alpha result

ridge_fig=plt.figure()
ridge_r2=ridge_fig.add_subplot(1,1,1)
ridge_r2_line=ridge_r2.plot(ridge_res[:,0],ridge_res[:,1],label="R2")
ridge_r2.set_ylabel('R2')
ridge_rmse=ridge_r2.twinx()
ridge_rmse_line=ridge_rmse.plot(ridge_res[:,0],ridge_res[:,2],label="MSE",color="blue")
ridge_r2.set_xlabel("gamma")
ridge_rmse.set_ylabel("MSE")
ridge_line=ridge_r2_line+ridge_rmse_line
ridge_labs=[l.get_label() for l in ridge_line]
ridge_r2.legend(ridge_line,ridge_labs,loc="best")
plt.savefig("ridge_sol_path.jpg",dpi=300)
plt.show()

# Compare results of standard and ridge regression

res_df=pd.DataFrame({'Method':['Standard', 'Ridge'],\
'MSE':[mean_squared_error(y_test, p_full), mean_squared_error(y_test, p_ridge)]})

print("#--------------------------------------")
print("Compare results of standard and ridge regression")
print(res_df)
print("#--------------------------------------")
print("\n")

# Lasso Regression: L1 regulation

lasso=Lasso(fit_intercept=True,alpha=0.1)
lasso.fit(x_train,y_train)
p_lasso=lasso.predict(x_test)

print("#--------------------------------------")
print("Lasso Regression Result")
print("r2_score = %.4f"%(r2_score(y_test,p_lasso)))
print("mean squared error = %.4f"%(mean_squared_error(y_test,p_lasso)))
print(sum(lasso.coef_==0))
print("#--------------------------------------")
print("\n")

# Next fitting the lasso and finding the optimal regularization using
# built in cross-validation provides further improvement over the LASSO
# That is to say, lassoCV contained the cross-validation methods

lasso_cv=LassoCV(fit_intercept=True,n_alphas=100,normalize=False)
lasso_cv.fit(x_train,y_train)
p_lasso_cv=lasso_cv.predict(x_test)

print("#--------------------------------------")
print("LassoCV Regression Result")
print("r2_score = %.4f"%(r2_score(y_test,p_lasso_cv)))
print("mean squared error = %.4f"%(mean_squared_error(y_test,p_lasso_cv)))
print(sum(lasso_cv.coef_==0))
print("#--------------------------------------")
print("\n")

# Let's visualize the solution path

lassop=lasso_cv.path(x_train,y_train)
lassop_fig=plt.figure()
lassop_plot=lassop_fig.add_subplot(1,1,1)
lassop_plot.plot(np.log(lassop[0]),lassop[1].T)
lassop_plot.set_xlabel("lambda value (log scale)")
lassop_plot.set_ylabel("Coefficient estimate value")
lassop_plot.set_title("lasso solution path")
plt.savefig("lasso_path.jpg",dpi=300)
plt.show()

# ElasticNetCV: contained L1 and L2 regularization
# CV == cross-validation

en_cv=ElasticNetCV(fit_intercept=True,n_alphas=100,normalize=False,l1_ratio=0.01)
en_cv.fit(x_train,y_train)
p_en_cv=en_cv.predict(x_test)

print("#--------------------------------------")
print("ElasticNetCV regression result")
print("r2_score = %.4f"%(r2_score(y_test,p_en_cv)))
print("mean squared error = %.4f"%(mean_squared_error(y_test,p_en_cv)))
print(sum(en_cv.coef_ == 0))
print("#--------------------------------------")
print("\n")

enp=en_cv.path(x_train,y_train)
enp_fig=plt.figure()
enp_plot=enp_fig.add_subplot(1,1,1)
enp_plot.plot(np.log(enp[0]),enp[1].T)
enp_plot.set_xlabel("lambda vale (log scale)")
enp_plot.set_ylabel("Coefficient estimate value")
enp_plot.set_title("EN solution path")
plt.savefig("en_path.jpg",dpi=300)
plt.show()

# Update results
res_df.loc[2,:]=[mean_squared_error(y_test,p_lasso_cv),'lasso']
res_df.loc[3,:]=[mean_squared_error(y_test,p_en_cv),'ElasticNetCV']

print("#--------------------------------------")
print("Update results")
print("LinearRegression Ridge LassoCV ElasticNetCV")
print(res_df)
print("#--------------------------------------")
print("\n")

# Load new data
# Input data have some problems

#ds_reg=pd.read_excel(xlsx,sheet_name="Distance and Shape regularizer",\
#headr=None).values
#
#print(ds_reg.shape)
#
## Helper function for fitting our model.
#
#def mat_pow(x, p):
#    u, d, v = svd(x)
#    return u.dot(np.diag(d**p)).dot(v)
#    
#sq_ds_reg = mat_pow(ds_reg, 0.5)
#x_train_aug = np.vstack((x_tr, sq_ds_reg))
#
## Create augmented matrix
#
## Training model
#y_train_aug=np.concatenate([y_train,np.zeros(x_train_aug.shape[1])])
#en_ds_reg=LinearRegression(fit_intercept=False,normalize=False)
#en_ds_reg.fit(x_train_aug,y_train_aug)
#
## Using model to fit testing data
#
#p_ds_reg_test=en_ds_reg.predict(x_te)
#
#print("#--------------------------------------")
#print("r2_score = %.4f"%(r2_score(y_test,p_ds_reg_test)))
#print("mean squared error = %.4f"%(mean_squared_error(y_test,p_ds_reg_test)))
#print("#--------------------------------------")
#print("\n")

hsi=loadmat(matdata)

print("#--------------------------------------")
print(" Load data shape")
#print(hsi.type)
print("#--------------------------------------")
print("\n")

keys=hsi.keys()

print("#--------------------------------------")
print("Take a look at what's contained in load matrix")
print(keys)
print("#--------------------------------------")
print("\n")

# Grab the objects we'll be using
# Take more attenation about all_indx,train_indx,test_indx
# That's to say split the data to train and test data

Y=hsi['Y']
Yarr=hsi["Yarr"]
lab=hsi["labels"]
lab2=hsi["label2"]
all_indx=np.arange(len(lab2))
train_indx=hsi["indx"]
test_indx=np.delete(all_indx,train_indx)

# Visualize the image data

img_fig=plt.figure()
img_plot=img_fig.add_subplot(1,1,1)
img_plot.grid(False)
img_plot.imshow(Yarr[:,:,[100,50,10]])
img_plot.set_xticklabels(['']*Yarr.shape[0])
img_plot.set_yticklabels(['']*Yarr.shape[0])
plt.savefig("kidney_image.jpg",dpi=300)
plt.show()

# Grab two wavelengths and visualize the distribution of classes
# kidney vs other

kid2_fig=plt.figure(figsize=(8,8))
kid2_class=kid2_fig.add_subplot(1,1,1)
kid2_mark=['o','x']
kid2_col=['red','blue']
kid2_leg=['kidney','other']
for i in range(2):
	ind=lab2==i+1
	kid2_class.scatter(Y[ind,80],Y[ind,122],color=kid2_col[i],marker=kid2_mark[i],\
	label=kid2_leg[i],alpha=0.5)
	
kid2_class.set_xlabel("Wavelength 81")
kid2_class.set_ylabel("Wavelength 123")
kid2_class.legend()
plt.savefig("kid_2class.jpg",dpi=300)
plt.show()

# Define plot function

def plot_labels(l,s,cmap=plt.cm.hot_r):
	fig=plt.figure(figsize=(8,8))
	plot=fig.add_subplot(1,1,1)
	plot.grid(False)
	plot.imshow(np.reshape(l,s).T,cmap=cmap)
	plot.set_xticklabels(['']*s[0])
	plot.set_yticklabels(['']*s[0])
	#plt.show()
	return fig
	
class_fig=plot_labels(lab2,(Yarr.shape[0],Yarr.shape[1]))

plt.savefig("img_2class.jpg",dpi=300)
plt.show()

# Run LDA: LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(solver="lsqr")
lda.fit(Y[train_indx,:],lab2[train_indx])
p_lda=lda.predict_proba(Y[test_indx,:])
l_lda=lda.predict(Y[test_indx,:])

# confusion matrix

print("#--------------------------------------")
print("Confusion matrix in LDA")
print(confusion_matrix(lab2[test_indx],l_lda))
print("#--------------------------------------")
print("\n")

# Visualize results
lda_fig=plot_labels(lda.predict(Y),(Yarr.shape[0],Yarr.shape[1]))
plt.savefig("lda_2class_pred.jpg",dpi=300)
plt.show()

# Try to sparse lda

slda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
slda.fit(Y[train_indx,:], lab2[train_indx])
p_slda = slda.predict_proba(Y[test_indx,:])
l_slda = slda.predict(Y[test_indx,:])

print("#--------------------------------------")
print("Confusion matrix in SLDA")
print(confusion_matrix(lab2[test_indx], l_slda))
print("#--------------------------------------")
print("\n")

slda_fig=plot_labels(slda.predict(Y),(Yarr.shape[0],Yarr.shape[1]))
plt.savefig("slda_2class_pred.jpg",dpi=300)
plt.show()

# Take look about performance metrics

aroc_lda=roc_auc_score(lab2[test_indx]-1,p_lda[:,1])
aroc_slda=roc_auc_score(lab2[test_indx]-1,p_slda[:,1])

roc_lda = roc_curve(lab2[test_indx]-1, p_lda[:,1])
roc_slda = roc_curve(lab2[test_indx]-1, p_slda[:,1])

roc_fig = plt.figure(figsize=(8,8))
roc_plot = roc_fig.add_subplot(1,1,1)
roc_plot.plot(roc_lda[0], roc_lda[1], label = 'LDA: AUC = ' + str(aroc_lda))
roc_plot.plot(roc_slda[0], roc_slda[1], label = 'SLDA: AUC = ' + str(aroc_slda))
roc_plot.legend()
roc_plot.set_xlabel('False Positive Rate')
roc_plot.set_ylabel('True Positive Rate')

plt.savefig("roc_lda_slda.jpg",dpi=300)
plt.show()

# Logistic Regression

lr=LogisticRegression(C=1e6,fit_intercept=False)
lr.fit(Y[train_indx,:],lab2[train_indx])
p_lr=lr.predict_proba(Y[test_indx,:])
l_lr=lr.predict(Y[test_indx,:])
aroc_lr=roc_auc_score(lab2[test_indx]-1,p_lr[:,1])

print("#--------------------------------------")
print("Logistic Regression roc_auc_score")
print("roc_auc_score = %.4f"%(aroc_lr))
print("#--------------------------------------")
print("\n")

# Try sparse Logistic Regression, add regulizration parts.

slr = LogisticRegression(C=1.0, fit_intercept=False, penalty='l1')
slr.fit(Y[train_indx,:], lab2[train_indx])
p_slr = slr.predict_proba(Y[test_indx,:])
l_slr = slr.predict(Y[test_indx,:])
aroc_slr = roc_auc_score(lab2[test_indx]-1, p_slr[:,1])

print("#--------------------------------------")
print("Sparse Logistic Regression roc_auc_score")
print("roc_auc_score = %.4f"%(aroc_slr))
print("#--------------------------------------")
print("\n")

# Visualoze Logistic Regression results

lr_fig = plot_labels(lr.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
plt.savefig("lr_2class_pred.jpg",dpi=300)
plt.show()

slr_fig = plot_labels(slr.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
plt.savefig("slr_2class_pred.jpg",dpi=300)
plt.show()

# Visualize roc curve

roc_lr = roc_curve(lab2[test_indx]-1, p_lr[:,1])
roc_slr = roc_curve(lab2[test_indx]-1, p_slr[:,1])

roc_lrfig = plt.figure(figsize=(8,8))
roc_lrplot = roc_lrfig.add_subplot(1,1,1)
roc_lrplot.plot(roc_lr[0], roc_lr[1],label = 'LR: AUC = ' + str(aroc_lr))
roc_lrplot.plot(roc_slr[0], roc_slr[1],label = 'SLR: AUC = ' + str(aroc_slr))
roc_lrplot.legend()
roc_lrplot.set_xlabel('False Positive Rate')
roc_lrplot.set_ylabel('True Positive Rate')
plt.savefig("roc_lr_slr.jpg",dpi=300)
plt.show()

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

plt.savefig("lr_slr_coef.jpg",dpi=300)
plt.show()

# Extend to multiple classifier

mlr=LogisticRegression(C=1e6,fit_intercept=False,multi_class="multinomial",\
solver="lbfgs")
mlr.fit(Y[train_indx,:],lab[train_indx])
p_mlr=mlr.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx],p_mlr))

# Sparse Logistic Regression

smlr = LogisticRegression(C=1000, fit_intercept=False,penalty='l1', solver='liblinear')
smlr.fit(Y[train_indx,:], lab[train_indx])
p_smlr = smlr.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx], p_smlr))

# Visualize results

mclass_fig=plot_labels(lab,(Yarr.shape[0],Yarr.shape[1]),cmap=plt.cm.Blues_r)
plt.savefig("mclass.jpg",dpi=300)
plt.show()

mclass_lr_fig = plot_labels(mlr.predict(Y), (Yarr.shape[0],Yarr.shape[1]),\
cmap = plt.cm.Blues_r)
plt.savefig("mclass_lr.jpg",dpi=300)
plt.show()

mclass_slr_fig = plot_labels(smlr.predict(Y), (Yarr.shape[0],Yarr.shape[1]),\
cmap = plt.cm.Blues_r)
plt.savefig("mclass_slr.jpg",dpi=300)
plt.show()

# SVM: support vector machine
# kernel function: linear

lsvc=svm.SVC(kernel="linear")
lsvc.fit(Y[train_indx,:], lab2[train_indx])
l_lsvc = lsvc.predict(Y[test_indx,:])
lsvc_fig = plot_labels(lsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
confusion_matrix(lab2[test_indx], l_lsvc)
plt.savefig("lsvc.jpg",dpi=300)
plt.show()

# kernel function: rbf
# gamma: maybe a parameters in rbf function

ksvc = svm.SVC(kernel='rbf', gamma = 0.5)
ksvc.fit(Y[train_indx,:], lab2[train_indx])
l_ksvc = ksvc.predict(Y[test_indx,:])
ksvc_fig = plot_labels(ksvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]))
confusion_matrix(lab2[test_indx], l_ksvc)
plt.savefig("rbfsvc.jpg",dpi=300)
plt.show()

# Multi-Class SVM

lmsvc = svm.SVC(kernel='linear', C=1000)
lmsvc.fit(Y[train_indx,:], lab[train_indx])
l_lmsvc = lmsvc.predict(Y[test_indx,:])
lmsvc_fig = plot_labels(lmsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]),\
plt.cm.Blues_r)
pd.DataFrame(confusion_matrix(lab[test_indx], l_lmsvc))
plt.savefig("lmsvc.jpg",dpi=300)
plt.show()

kmsvc = svm.SVC(kernel='rbf', gamma = 0.5, C=1000)
kmsvc.fit(Y[train_indx,:], lab[train_indx])
l_kmsvc = kmsvc.predict(Y[test_indx,:])
kmsvc_fig = plot_labels(kmsvc.predict(Y), (Yarr.shape[0],Yarr.shape[1]),\
plt.cm.Blues_r)
pd.DataFrame(confusion_matrix(lab[test_indx], l_kmsvc))
plt.savefig("rbfmsvc.jpg",dpi=300)
plt.show()

# Neural Network: Multi-layer Perceptron classifier

mlp=MLPClassifier(hidden_layer_sizes=(50,50,))
mlp.fit(Y[train_indx,:],lab[train_indx])
l_mlp=mlp.predict(Y[test_indx,:])
pd.DataFrame(confusion_matrix(lab[test_indx], l_mlp))

mlp_fig = plot_labels(mlp.predict(Y), (Yarr.shape[0],Yarr.shape[1]),\
plt.cm.Blues_r)
plt.savefig("mlp.jpg",dpi=300)
plt.show()


