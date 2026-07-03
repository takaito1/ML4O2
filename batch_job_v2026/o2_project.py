#!/usr/bin/env python
# coding: utf-8

# # O2 gapfill projection
#     - Needs to enter a 6 digit input parameter as follows : 
#     - First digit = Algorithm type (1=RF, 2=NN)
#     - Second digit = Data Source (1=Ship only, 2=Ship+Argo)
#     - Third digit = Ocean basin (1=Atlantic, 2=Pacific, 3=Indian, 4=Southern, 5=Arctic)
#     - Fourth digit = T/S data source (1=EN4)
#     - Fifth digit = predictor variable set (1=default, 2=cos/sin_month)
#     - Sixth digit = hyperparameter set (1=default)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sklearn as skl
import gsw
import cartopy.crs as ccrs
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')
import joblib
from multiprocessing import Pool


# In[2]:
os.system('echo $USER > userid')
usrid=np.genfromtxt('userid',dtype='<U16')
os.system('rm userid')
#
#
# version information
ver = np.genfromtxt('data_XXX.txt',dtype='U12').tolist()
dirtmp=f'/glade/derecho/scratch/{usrid}/ML4O2_temp/'
dirout=f'/glade/derecho/scratch/{usrid}/ML4O2_results/'
diro = dirtmp
# 
# The version information will determine which basin / algorithm will be used to calculate the O2 maps. 
os.system(f'mkdir -p {dirtmp}')
os.system(f'mkdir -p {dirout}')
os.system(f'mkdir -p {dirtmp}/temp')

# In[3]:


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
    #date2='08232025'
    date2='12032025'
else:
    print('error - incorrect input data type')
#
print(basin[int(selection[2])-1]+' will be mapped')
#
if selection[3] == '1':
    print('EN4 dataset will be used for T/S input. ')
    endyear=2021
elif selection[3] == '2':
    print('ORAS4 dataset will be used for T/S input. ')
    endyear=2018
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
        #dirinput = f'/glade/work/ito/ML4O2/apr2025/input_202504_woaadj/'
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


# In[4]:


# Define the input and output folders
#
if selection[3] == '1':
    dirf = '/glade/campaign/univ/ugit0034/EN4/L09_20x180x360/'
elif selection[3] == '2':
    dirf = '/glade/campaign/univ/ugit0034/ORAS4/TSN2/'
elif selection[3] == '4':
    dirf = '/glade/campaign/univ/ugit0034/EN4/C14/'
    
# In[5]:

if selection[3]=='1':
    dc=xr.open_dataset(dirf+'EN4_TSN2_L09_180x360_196501.nc')
elif selection[3]=='2':
    dc=xr.open_dataset(dirf+'ORAS4_TSN2_196501.nc')
    dc.coords['lon'] = (dc.coords['lon'] + 180) % 360 - 180
    dc = dc.sortby(dc.lon)
elif selection[3]=='4':
    dirf = '/glade/campaign/univ/ugit0034/EN4/C14/'
    dbase='EN4_C14_180x360_'
    dc=xr.open_dataset(dirf+dbase+'196501.nc')
# obtain vertical grid
depth1 = dc.depth.sel(depth=slice(0,5500)).to_numpy()
#ds=xr.open_dataset(dirin+var[0]+fmer+str(1965)+'.nc')
#Z=ds.depth.to_numpy()
Z=depth1
Nz=np.size(Z)

# In[6]:


# select analysis period
# do not change the start year from 1965 (this is when Carpenter 1965 established modern Winkler method)
yrs=np.arange(1965,endyear+1,1)
t=np.arange('1965-01',str(endyear)+'-01',dtype='datetime64[M]')


# In[7]:

optparm=np.load(dirtmp+f'optparm_v{ver}.npz')
optparm1=optparm['parm1']
optparm2=optparm['parm2']
#MLmodel0=joblib.load(dirtmp+f'algorithm_v{ver}_cv0_{optparm1}_{optparm2}.sav')
#MLmodel1=joblib.load(dirtmp+f'algorithm_v{ver}_cv1_{optparm1}_{optparm2}.sav')
#MLmodel2=joblib.load(dirtmp+f'algorithm_v{ver}_cv2_{optparm1}_{optparm2}.sav')
#MLmodel3=joblib.load(dirtmp+f'algorithm_v{ver}_cv3_{optparm1}_{optparm2}.sav')
#MLmodel4=joblib.load(dirtmp+f'algorithm_v{ver}_cv4_{optparm1}_{optparm2}.sav')

MLmodel = joblib.load(dirout+f'algorithm_v{ver}.sav')
# read in additional parameters
params = np.load(dirout+f'ML_params_v{ver}.npz')
Xm=params['Xm']
Xstd=params['Xstd']
ym=params['ym']
ystd=params['ystd']


# In[8]:


# basin mask
dsm=xr.open_dataset('/glade/campaign/univ/ugit0034/wod18/basin_mask_01.nc')
#ma = dsm.basin_mask.sel(depth=Z).to_numpy()
#
mask2d=np.zeros((1,180,360))
mask2d[0,:,:] = dsm.basin_mask.sel(depth=0)
mask2d = np.where(mask2d==56,3,mask2d)
mask3d = np.tile(mask2d,[np.size(Z),1,1])
mask= dsm.basin_mask.sel(depth=Z)
ma=np.where(mask==0,0,mask3d)

# In[9]:


zlev=300
kind=[idx for idx,elem in enumerate(Z) if elem==zlev]
maz=np.squeeze(ma[kind,:,:])
#
mon=["%.2d" % i for i in np.arange(1,13,1)]
#
if selection[3]=='1':
    dc=xr.open_dataset(dirf+'EN4_TSN2_L09_180x360_'+str(1965)+mon[0]+'.nc')
elif selection[3]=='2':
    dc=xr.open_dataset(dirf+'ORAS4_TSN2_'+str(1965)+mon[0]+'.nc')
    dc.coords['lon'] = (dc.coords['lon'] + 180) % 360 - 180
    dc = dc.sortby(dc.lon)
elif selection[3]=='4':
    dirf = '/glade/campaign/univ/ugit0034/EN4/C14/'
    dbase='EN4_C14_180x360_'
    dc=xr.open_dataset(dirf+dbase+str(1965)+mon[0]+'.nc')
#
# use alternative x coordinate: longitude - 20
y=dc.lat.to_numpy()
x=dc.lon.to_numpy()
xa0 = x - 20
xalt = np.where(xa0<0,xa0+360,xa0)
#
Ny=np.size(y)
Nx=np.size(x)
Nt=np.size(yrs)*12
xx,yy=np.meshgrid(xalt,y)
#
depth1 = dc.depth.sel(depth=slice(0,5500)).to_numpy()
Nz1 = np.size(depth1)
print(Nz1)
#


# In[10]:


# apply basin mask 
bindex=[1,2,3,10,11,4,5,6,7,8,9,12,53]
def apply_basinmask(datain):
    return np.where((maz==bindex[int(selection[2])-1]),datain,np.nan)

# In[11]:


# get input data from full model
def get_inputdata(zlev,it,year,mn):
    zlev=float(zlev)
    #dc = xr.open_dataset(dirf+'EN4_TSN2_G10_180x360_'+str(year)+mon[mn]+'.nc')
    if selection[3] == '1':
        dc = xr.open_dataset(dirf+'EN4_TSN2_L09_180x360_'+str(year)+mon[mn]+'.nc')
        soa=dc.SA.interp(depth=zlev).to_numpy().squeeze()
        toa=dc.CT.interp(depth=zlev).to_numpy().squeeze()
    elif selection[3] == '2':
        dc=xr.open_dataset(dirf+'ORAS4_TSN2_'+str(year)+mon[0]+'.nc')
        dc.coords['lon'] = (dc.coords['lon'] + 180) % 360 - 180
        dc = dc.sortby(dc.lon)
        soa=dc.SA.interp(depth=zlev).to_numpy().squeeze()
        toa=dc.CT.interp(depth=zlev).to_numpy().squeeze()
    elif selection[3] == '4':
        dc = xr.open_dataset(dirf+dbase+str(year)+mon[mn]+'.nc')
        soa=dc.so.interp(depth=zlev).to_numpy().squeeze()
        toa=dc.thetao.interp(depth=zlev).to_numpy().squeeze()
    #
    
#    sigma=dc.sigma0.interp(depth=zlev).to_numpy().squeeze()
#    N2=dc.N2.interp(depth=zlev).to_numpy().squeeze()
    return soa,toa#,sigma,N2

dsg=xr.open_dataset('NOAA_Global_SST.nc')
GMT=dsg.SST.sel(time=slice('1965-01','2025-12')).to_numpy()
#print(Nt,GMT.size)

# generate data matrix
def gen_datamatrix(xi,yi,it,x1,x2,x3,x4):#,xsig,xn2):
    X1 = x1.flatten() # 
    X2 = x2.flatten() # 
    #Xsig = xsig.flatten()
    #Xn2  = xn2.flatten()
    #
    if selection[2] == '4':
        newx3 = (x4+90)*np.cos(np.deg2rad(x3))
        newx4 = (x4+90)*np.sin(np.deg2rad(x3))
        X3=newx3.flatten()
        X4=newx4.flatten()
    elif selection[2] == '5':
        newx3 = (-x4+90)*np.cos(np.deg2rad(x3))
        newx4 = (-x4+90)*np.sin(np.deg2rad(x3))
        X3=newx3.flatten()
        X4=newx4.flatten()
    else:
        X3 = x3.flatten() # 
        X4 = x4.flatten() # 
    tt0  = np.ones((Ny,Nx))*it
    X5 = tt0.flatten() # decimal year 
    X6 = X5%12         # month
    xxi = xi.flatten() # lon
    yyi = yi.flatten() # lat
    #
    GMT0 = np.ones((Ny,Nx))*GMT[it]
    GMT1 = GMT0.flatten()
    #ml1 = mld.flatten()
    #X6 = np.where(ml1>zlev-zoff,X6,2)
    # remove nan
    #print([np.size(X1),np.size(X2),np.size(X3),np.size(X4),np.size(X5)])
    dd = X1+X2+X3+X4+X5#+Xsig+Xn2
    X11=X1[np.isnan(dd)==False]
    X21=X2[np.isnan(dd)==False]
    X31=X3[np.isnan(dd)==False]
    X41=X4[np.isnan(dd)==False]
    X51=X5[np.isnan(dd)==False]
    X61=X6[np.isnan(dd)==False]
    #
    Xi=xxi[np.isnan(dd)==False]
    Yi=yyi[np.isnan(dd)==False]
    #
    GMT11=GMT1[np.isnan(dd)==False]
    #Xsig1=Xsig[np.isnan(dd)==False]
    #Xn21 =Xn2[np.isnan(dd)==False]
    #
    zin = np.ones(np.size(X11))*zlev
    # Normalize data
    # generate data matrix and standardize it
    if selection[4] == '1':
        #X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, tc1])
        X = np.array([X11, X21, X31, X41, zin, X51, X61])
        #print('Predictor variables include T, S, lon, lat, depth (pressure), year, month')
    elif selection[4] == '2':
        X = np.array([X11, X21, X31, X41, zin, X51, np.cos(2*np.pi*X61/12), np.sin(2*np.pi*X61/12)])
        #X = np.array([dsa1, dta1, xx1, yy1, zz1, tt1, np.cos(2*np.pi*tc1/12), np.sin(2*np.pi*tc1/12)])
        #print('Predictor variables include T, S, lon, lat, depth (pressure), year, cos(month), sin(month)')
    elif selection[4] == '3':
        X = np.array([X11, X21, X31, X41, zin, X51, np.cos(2*np.pi*X61/12), np.sin(2*np.pi*X61/12), Xsig1])
    elif selection[4] == '4':
        X = np.array([X11, X21, X31, X41, zin, X51, np.cos(2*np.pi*X61/12), np.sin(2*np.pi*X61/12), Xsig1, Xn21])
    elif selection[4] == '5':
        X = np.array([X11, X21, X31, X41, zin, np.cos(2*np.pi*X61/12), np.sin(2*np.pi*X61/12)])
        #print('Predictor variables include T, S, lon, lat, depth (pressure), cos(month), sin(month)')
    elif selection[4] == '6':
        X = np.array([X11, X21, X31, X41, zin, GMT11, np.cos(2*np.pi*X61/12), np.sin(2*np.pi*X61/12)])
        #print('Predictor variables include T, S, lon, lat, depth (pressure), GMT, cos(month), sin(month)')
    else:
        print('error - incorrect predictor variable type')
    #
    Xa = (X.T - Xm)/Xstd
    Nsample = np.size(X11)
    #print(Nsample)
    return Xa,Xi,Yi


# In[13]:


def map_yearly(year):
    Nx=np.size(x)
    Ny=np.size(y)
    zlev_arr=np.array([zlev])
    # display zlev
    #print(zlev_arr)
    #
    o2est2=np.zeros((12,1,Ny,Nx))*np.nan
    #o2est2=np.zeros((5,12,1,Ny,Nx))*np.nan
    #o2std=np.zeros((12,1,Ny,Nx))*np.nan
    xxi,yyi=np.meshgrid(np.arange(0,Nx,1),np.arange(0,Ny,1))
    if year%10 == 5:
        print('year = '+str(year))
    t=np.arange(str(year)+'-01',str(year+1)+'-01',dtype='datetime64[M]')
    if year <= 2024:
        Nmon=12
    else:
        Nmon=12
    #
    for month in range(Nmon):
        it = month+(year-1965)*12
        #soa,toa,sigma,N2 = get_inputdata(zlev,it,year,month)
        soa,toa = get_inputdata(zlev,it,year,month)
        # apply mask
        soa=apply_basinmask(soa)
        toa=apply_basinmask(toa)
        #sigma=apply_basinmask(sigma)
        #N2=apply_basinmask(N2)
        # generate data matrix
        Xa,xi,yi=gen_datamatrix(xxi,yyi,it,soa,toa,xx,yy)#,sigma,N2)
        temp = np.shape(Xa)
        Nsample=temp[0]
        # projection
        if Nsample >= 1:
            out = np.zeros((1,Nsample))
            out[0,:] = reg.predict(Xa)
            # map it back to lon-lat grid
            for n in range(Nsample):
                for m in range(1):
                    o2est2[month,0,yi[n],xi[n]]=out[m,n]*ystd+ym
    #
    da1=xr.DataArray(data=o2est2,name='o2est',dims=['time','depth','lat','lon'],
                     coords={'time':t,'depth':zlev_arr,'lat':yout,'lon':xout})
    ds=da1.to_dataset()
    ds.to_netcdf(diro+f'temp/o2est_v{ver}_{year}.nc')
    return 0


# In[ ]:


zlevels = depth1
#
# reconstruction in parallel mode
#
reg=MLmodel
xout=dc.lon
yout=dc.lat
#
for zlev_cnt,zlev in enumerate(zlevels):
    print(f'calculating {zlev}m')
    zlev=float(zlev)
    #maz = dsm.basin_mask.interp(depth=zlev).to_numpy()
    maz = np.squeeze(ma[zlev_cnt,:,:])
    #kind=[idx for idx,elem in enumerate(Z) if elem==zlev]
    #maz=np.squeeze(ma[kind,:,:])
    os.system('rm '+diro+f'/temp/o2est_v{ver}_*.nc')
    os.system('rm '+diro+f'/O2map_v{ver}_z{int(zlev)}.nc')
    #
    if __name__ == '__main__':
        with Pool(10) as p:
            print(p.map(map_yearly, yrs))
    #
    # save the result as a netCDF file
    #
    dtemp=xr.open_mfdataset(diro+f'temp/o2est_v{ver}_*.nc')
    dtemp.to_netcdf(diro+'/O2map_v'+ver+'_z'+str(int(zlev))+'.nc')

ds=xr.open_mfdataset(f'{diro}O2map_v{ver}*')

ds.to_netcdf(f'{dirout}O2map_v{ver}.nc')
