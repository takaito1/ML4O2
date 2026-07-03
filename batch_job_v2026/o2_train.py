#!/usr/bin/env python
# coding: utf-8
# %%

# # Training algorithm for O2 gapfill
#     - Needs to enter a 6 digit input parameter as follows : 
#     - First digit = Algorithm type (1=RF, 2=NN)
#     - Second digit = Data Source (1=Ship only, 2=Ship+Argo)
#     - Third digit = Ocean basin (1=Atlantic, 2=Pacific, 3=Indian, 4=Southern, 5=Arctic)
#     - Fourth digit = T/S data source (1=EN4)
#     - Fifth digit = predictor variable set (1=default, 2=cos/sin_month)
#     - Sixth digit = hyperparameter set (1=default, 2=preset hyperparameters)

# %%


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sklearn as skl
import gsw
import cartopy.crs as ccrs
from scipy.interpolate import interp1d
import os
import warnings
import joblib
#import cuml
warnings.filterwarnings('ignore')


# %%
os.system('echo $USER > userid')
usrid=np.genfromtxt('userid',dtype='<U16')
os.system(f'mkdir -p /glade/derecho/scratch/{usrid}/ML4O2_temp/')
os.system(f'mkdir -p /glade/derecho/scratch/{usrid}/ML4O2_results/')
os.system(f'mkdir -p /glade/derecho/scratch/{usrid}/WOD18_OSDCTD/')
#
# version information
ver = np.genfromtxt('data_XXX.txt',dtype='U12').tolist()
rerun = False    # indicate again whether you are re-running previous results
#
dirout=f'/glade/derecho/scratch/{usrid}/ML4O2_temp/'
# ### display selection

# %%


selection = ver.split('.')
basin = ['atlantic','pacific','indian','southern','arctic',\
         'mediterranean','baltic','black','red','persian',\
         'hudson','japan-east','caspian']
#
if selection[0] == '1':
    print('Random Forst algorithm will be used.')
    alg = 'RF'
elif selection[0] == '2':
    print('Neural Network algorithm will be used.')
    alg = 'NN'
else:
    print('error - incorrect algorithm type')
#
if selection[1] == '1':
    print('Ship-based O2 data will be used. ')
    date2='08222025'
elif selection[1] == '2':
    print('Ship-based and Argo-O2 data will be used. ')
    date2='07082025'
elif selection[1] == '3':
    print('Ship-based and modified Argo-O2 data will be used. ')
    date2='12032025'
else:
    print('error - incorrect input data type')
date1=date2
#
print(f'-----------------------------------------------')
print(f' Machine Learning For Dissolved Oxygen (ML4O2) ')
print(f' version{ver} date:{date1}')
print(f'-----------------------------------------------')
#
print(basin[int(selection[2])-1]+' will be mapped')
#
if selection[3] == '1':
    print('EN4 dataset will be used for T/S input. ')
    endyear=2021
    if selection[1]=='1':
        dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_20250708'
elif selection[3] == '2':
    print('ORAS4 dataset will be used for T/S input. ')
    endyear=2018
elif selection[3] == '3':
    print('SODA3.4.2 dataset will be used for T/S input. ')
elif selection[3] == '4':
    print('EN4 (C14) dataset will be used for T/S input. ')
    endyear=2025
    if selection[1]=='1':
        #dirinput = f'/glade/work/ito/ML4O2/apr2025/input_202504_ship/'
        dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_20250822/'
    elif selection[1]=='2':
        #dirinput = f'/glade/work/ito/ML4O2/apr2025/input_202504_noadj/'
        dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_20250708/'
    elif selection[1]=='3':
        #dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_20250823/'
        #dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_20251203/'
        dirinput = f'/glade/campaign/univ/ugit0034/ML4O2/input_202602/'
    #Nlev=63
    Nlev=102
else:
    print('error - incorrect T/S data type')
#
if selection[4] == '1':
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, month')
elif selection[4] == '2':
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month)')
elif selection[4] == '3':
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month), sigma')
elif selection[4] == '4':
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month), sigma, N2')
#NEW
elif selection[4] == '5':
    print('Predictor variables include T, S, lon, lat, depth (pressure), cos(month), sin(month)')
elif selection[4] == '6':
    print('Predictor variables include T, S, lon, lat, depth (pressure), GMT, cos(month), sin(month)')
#NEW
else:
    print('error - incorrect predictor variable type')
#
if selection[5] == '1':
    print('Hyperparameter set is optimized via K-fold CV')
elif selection[5] == '2':
    print('A pre-set hyperparameter set is used')
elif selection[5] == '4':
    print('New K-fold cross validation')
else:
    print('error - incorrect hyperparameter type')


# %%


# Define the input and output folders
#
diro = f'/glade/derecho/scratch/{usrid}/WOD18_OSDCTD/'
if selection[3] == 1:
    dirf = '/glade/campaign/univ/ugit0034/EN4/L09_20x180x360/'
elif selection[3] == 2:
    dirf = '/glade/campaign/univ/ugit0034/ORAS4/TSN2/'
#
dirin = '/glade/campaign/univ/ugit0034/WOD18_OSDCTD/'
fargo = '/glade/campaign/univ/ugit0034/bgcargo/o2_Global_ARGO_Type12_47lev.nc'
fosd='_1x1bin_osd_'
fctd='_1x1bin_ctd_'
fmer='_1x1bin_osdctd_'
var=['o2','TSN2']
os.system('mkdir -p '+diro)
os.system('mkdir -p '+diro+'temp')


# ### Preprocessing the data

# %%


# obtain vertical grid
ds=xr.open_dataset(dirin+var[0]+fmer+str(1965)+'.nc')
Z=ds.depth.to_numpy()
Nz=np.size(Z)


# %%


# select analysis period
# do not change the start year from 1965 (this is when Carpenter 1965 established modern Winkler method)
yrs=np.arange(1965,endyear+1,1)


# %%


# basin-specific input data loading
dsm=xr.open_dataset('/glade/campaign/univ/ugit0034/wod18/basin_mask_01.nc')
#
bname0=basin[int(selection[2])-1]
#
if selection[3]=='1':
    if selection[1]=='2':
        doa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/o20_{bname0}_1x1_47lev.npy')
        dta1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/t0_{bname0}_1x1_47lev.npy')
        dsa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/s0_{bname0}_1x1_47lev.npy')
        xx1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/lon0_{bname0}_1x1_47lev.npy')
        yy1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/lat0_{bname0}_1x1_47lev.npy')
        zz1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/depth0_{bname0}_1x1_47lev.npy')
        tt1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/time0_{bname0}_1x1_47lev.npy')
        tc1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/month0_{bname0}_1x1_47lev.npy')
        dsga1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/sigma0_{bname0}_1x1_47lev.npy')
        dn2a1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4_offsetWOA/N20_{bname0}_1x1_47lev.npy')
    elif selection[1]=='1':
        doa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/o20_{bname0}_1x1_47lev_ship.npy')
        dta1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/t0_{bname0}_1x1_47lev_ship.npy')
        dsa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/s0_{bname0}_1x1_47lev_ship.npy')
        xx1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/lon0_{bname0}_1x1_47lev_ship.npy')
        yy1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/lat0_{bname0}_1x1_47lev_ship.npy')
        zz1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/depth0_{bname0}_1x1_47lev_ship.npy')
        tt1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/time0_{bname0}_1x1_47lev_ship.npy')
        tc1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/month0_{bname0}_1x1_47lev_ship.npy')
        dsga1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/sigma0_{bname0}_1x1_47lev_ship.npy')
        dn2a1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/EN4/N20_{bname0}_1x1_47lev_ship.npy')
    if (selection[4]=='6')&(selection[1]=='1'):
        doa1 = np.load(f'{dirinput}/o20_{bname0}_1x1_47lev.npy')
        dta1 = np.load(f'{dirinput}/t0_{bname0}_1x1_47lev.npy')
        dsa1 = np.load(f'{dirinput}/s0_{bname0}_1x1_47lev.npy')
        xx1 = np.load(f'{dirinput}/lon0_{bname0}_1x1_47lev.npy')
        yy1 = np.load(f'{dirinput}/lat0_{bname0}_1x1_47lev.npy')
        zz1 = np.load(f'{dirinput}/depth0_{bname0}_1x1_47lev.npy')
        tt1 = np.load(f'{dirinput}/time0_{bname0}_1x1_47lev.npy')
        tc1 = np.load(f'{dirinput}/month0_{bname0}_1x1_47lev.npy') 
        GMT1 = np.load(f'{dirinput}/GMT0_{bname0}_1x1_47lev.npy') 
#
elif selection[3]=='2':
    if selection[1]=='2':
        doa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/o20_{bname0}_1x1_47lev.npy')
        dta1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/t0_{bname0}_1x1_47lev.npy')
        dsa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/s0_{bname0}_1x1_47lev.npy')
        xx1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/lon0_{bname0}_1x1_47lev.npy')
        yy1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/lat0_{bname0}_1x1_47lev.npy')
        zz1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/depth0_{bname0}_1x1_47lev.npy')
        tt1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/time0_{bname0}_1x1_47lev.npy')
        tc1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/month0_{bname0}_1x1_47lev.npy')
        dsga1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/sigma0_{bname0}_1x1_47lev.npy')
        dn2a1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/N20_{bname0}_1x1_47lev.npy')
    elif selection[1]=='1':
        doa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/o20_{bname0}_1x1_47lev_ship.npy')
        dta1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/t0_{bname0}_1x1_47lev_ship.npy')
        dsa1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/s0_{bname0}_1x1_47lev_ship.npy')
        xx1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/lon0_{bname0}_1x1_47lev_ship.npy')
        yy1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/lat0_{bname0}_1x1_47lev_ship.npy')
        zz1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/depth0_{bname0}_1x1_47lev_ship.npy')
        tt1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/time0_{bname0}_1x1_47lev_ship.npy')
        tc1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/month0_{bname0}_1x1_47lev_ship.npy')
        dsga1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/sigma0_{bname0}_1x1_47lev_ship.npy')
        dn2a1 = np.load(f'/glade/campaign/univ/ugit0034/ML4O2/input_202404/ORAS4/N20_{bname0}_1x1_47lev_ship.npy')
elif selection[3]=='4':
    doa1 = np.load(f'{dirinput}/o20_{bname0}_1x1_{Nlev}lev.npy')
    dta1 = np.load(f'{dirinput}/t0_{bname0}_1x1_{Nlev}lev.npy')
    dsa1 = np.load(f'{dirinput}/s0_{bname0}_1x1_{Nlev}lev.npy')
    xx1 = np.load(f'{dirinput}/lon0_{bname0}_1x1_{Nlev}lev.npy')
    yy1 = np.load(f'{dirinput}/lat0_{bname0}_1x1_{Nlev}lev.npy')
    zz1 = np.load(f'{dirinput}/depth0_{bname0}_1x1_{Nlev}lev.npy')
    tt1 = np.load(f'{dirinput}/time0_{bname0}_1x1_{Nlev}lev.npy')
    tc1 = np.load(f'{dirinput}/month0_{bname0}_1x1_{Nlev}lev.npy')
    GMT1 = np.load(f'{dirinput}/GMT0_{bname0}_1x1_{Nlev}lev.npy')

# %%

Nsample = np.size(doa1)
print(Nsample)


# ### This is where we choose what variables to include

# %%


# generate data matrix and standardize it
if selection[4] == '1':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, tc1])
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, month')
elif selection[4] == '2':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12)])
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month)')
elif selection[4] == '3':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12), dsga1])
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month), sigma')
elif selection[4] == '4':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12), dsga1, dn2a1])
    print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month), sigma, N2')
elif selection[4] == '5':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12)])
    print('Predictor variables include T, S, lon, lat, depth (pressure), cos(month), sin(month)')
elif selection[4] == '6':
    X = np.array([dsa1, dta1, xx1, yy1, zz1, GMT1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12)])
    print('Predictor variables include T, S, lon, lat, depth (pressure), GMT, cos(month), sin(month)')
else:
    print('error - incorrect predictor variable type')
#X = np.array([dsa1, dta1, xx1, yy1, tt1, tc1])
#
y = doa1
#
Xm = np.mean(X,axis=1)
Xstd = np.std(X,axis=1)
#
N=np.size(y)
# normalize x and y
Xa = (X.T - Xm)/Xstd
ym = np.mean(y)
ystd = np.std(y)
ya = (y-ym)/ystd
#
np.savez(dirout+f'ML_params_v{ver}.npz',Xm=Xm,Xstd=Xstd,ym=ym,ystd=ystd)


# ## ML

# ### Manually configure K-fold cross validation
# - 80-20 split by randomly selecting 11 years
# - K-fold CV with split by decade
# - Skip the next 3 cells if re-using the previous train-test split

# %%


# determine which year to be used for test data
if rerun==False:
    yr_drop = np.random.choice(yrs,11,replace=False)
    print(yr_drop)

# group these years together into a single array
if rerun==False:
    yr1=np.round(tt1/12+1965)
    ind=(yr1==int(yr_drop[0]))
    N=np.sum(ind)
    for n in np.arange(1,11,1):
        tmp=(yr1==yr_drop[n])
        ind=(ind==True)|(tmp==True)
        N=N+np.sum(tmp)
    #print(N,np.sum(ind))
    print(f'the count of data point (bins) = {yr1.size}')
    print(f'the count of test data (bins) = {N}, which is {N/yr1.size*100}%')

# Assemble into input data (train/test) and save it for record
if rerun==False:
    ind1 = (ind==False)
    X_train = Xa[ind1,:]
    X_test = Xa[ind,:]
    y_train = ya[ind1]
    y_test = ya[ind]
    print(f'the count of train data point (bins) = {y_train.size}')
    np.savez(dirout+f'train_test_v{ver}.npz',X_train=X_train,X_test=X_test,
             y_train=y_train,y_test=y_test,yr_drop=yr_drop)

# ### Start here to re-use previous train-test split

# Read from the saved input data file
tmp=np.load(dirout+f'train_test_v{ver}.npz')
X_train = tmp['X_train']
X_test  = tmp['X_test']
y_train = tmp['y_train']
y_test  = tmp['y_test']
#
tmp = np.load(dirout+f'ML_params_v{ver}.npz')
Xstd = tmp['Xstd']
Xm   = tmp['Xm']
#
#
if (selection[4] == '5')|(selection[4] == '6'):
    ttmp0 = tt1[ind1]
else:
    ttmp0 = X_train[:,5]*Xstd[5]+Xm[5]
#
yr1 = ttmp0/12+1965

# Calculate the Decadal Group K-fold
tbnds=[1965,1977,1989,2000,2012,2022]
Kval = 5
#yr1=X[5,ind1]/12+1965
print(f'The total count of data points = {yr1.size}')
for n in range(5):
    K_test=(yr1>=tbnds[n])&(yr1<tbnds[n+1])
    K_train=(K_test==False)
    X_trainK = X_train[K_train,:]
    X_testK = X_train[K_test,:]
    y_trainK = y_train[K_train]
    y_testK = y_train[K_test]
    # check
    print(f'N,train = {y_train.size}, Group {n} train size = {y_trainK.size}, Group {n} test size = {y_testK.size}, {y_testK.size/y_train.size*100}%')


# ### Algorithm selection & training

# %%


RF_parameters = {'min_samples_split':[2,4,8,16,32,64],'max_features':[2,3,5]}
NN_parameters = {'hidden_layer_sizes':[[5,5],[10,10],[20,20],[40,40],
                                       [20,15,10,5],[20,15,10,5,5,5]],'alpha':[.001, .01, .1]}


# %%
def train_K(k):
    if alg =='RF':
        from sklearn.ensemble import RandomForestRegressor
        mxf=RF_parameters['max_features'][parm2]
        msp=RF_parameters['min_samples_split'][parm1]
        msl=5
        nest=500
        regr=RandomForestRegressor(n_jobs=-1,n_estimators=nest,min_samples_split=msp,
                                   min_samples_leaf=msl,max_features=mxf)
        K_test=(yr1>=tbnds[k])&(yr1<tbnds[k+1])
        K_train=(K_test==False)
        X_trainK = X_train[K_train,:]
        X_testK = X_train[K_test,:]
        y_trainK = y_train[K_train]
        y_testK = y_train[K_test]
        regr.fit(X_trainK, y_trainK)
        y_est = regr.predict(X_testK)
        np.savez(dirout+f'RFtest_pred_v{ver}_cv{k}_{parm1}_{parm2}.npz',Xtest=X_testK,test=y_testK,est=y_est)
        filename = dirout+f'algorithm_v{ver}_cv{k}_{parm1}_{parm2}.sav'
        joblib.dump(regr, filename)
    elif alg == 'NN':
        from sklearn.neural_network import MLPRegressor
        hls=NN_parameters['hidden_layer_sizes'][parm1]
        alp=NN_parameters['alpha'][parm2]
        regr=MLPRegressor(max_iter=1000,hidden_layer_sizes=hls,alpha=alp)
        K_test=(yr1>=tbnds[k])&(yr1<tbnds[k+1])
        K_train=(K_test==False)
        X_trainK = X_train[K_train,:]
        X_testK = X_train[K_test,:]
        y_trainK = y_train[K_train]
        y_testK = y_train[K_test]
        regr.fit(X_trainK, y_trainK)
        y_est = regr.predict(X_testK)
        np.savez(dirout+f'NNtest_pred_v{ver}_cv{k}_{parm1}_{parm2}.npz',Xtest=X_testK,test=y_testK,est=y_est)
        filename = dirout+f'algorithm_v{ver}_cv{k}_{parm1}_{parm2}.sav'
        joblib.dump(regr, filename)
    r=np.corrcoef(y_est,y_testK)
    return np.round(r[0,1]**2,4)

# save the normalization factors first
# then perform gridsearch K-fold cross validation
for parm1 in range(6):
    for parm2 in range(3):
        if alg =='NN':
            from multiprocessing import Pool
            if __name__ == '__main__':
                with Pool(5) as p:
                    print(p.map(train_K, [0, 1, 2, 3, 4]))
        elif alg=='RF':
            for n in range(5):
                r2=train_K(n)
                print(n,parm1,parm2,r2)
                


# ### Completed. Next, evaluate the results in o2_eval_XXXX script

# %%




