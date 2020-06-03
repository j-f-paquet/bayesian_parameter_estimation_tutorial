import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import colors, ticker, cm
#import scipy.interpolate
#from matplotlib.mlab import bivariate_normal
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as mtick
from scipy.stats import linregress
import scipy.interpolate
import pandas as pd
import os.path
import glob
import re

################
# Get the data #
################

#data_pT, data_mean, data_err = np.loadtxt("RAA_AuAu200_jet_R0.4_cut2_coef1.txt").T
data_jet04_pT, data_jet04_mean, data_jet04_err = np.loadtxt("data/RAA_AuAu200_jet_R0.4_cut2_coef0.8.txt").T
data_charged_hadron_pT, data_charged_hadron_mean, data_charged_hadron_err = np.loadtxt("data/RAA_AuAu200_charged_hadron_cut2_coef0.8.txt").T
data_dict={
    'charged_hadron':{
        'pT':data_charged_hadron_pT,
        'mean':data_charged_hadron_mean,
        'err':data_charged_hadron_err
    },
    'jet_R0.4':{
        'pT':data_jet04_pT,
        'mean':data_jet04_mean,
        'err':data_jet04_err
    }
}

########################
# Get the calculations #
########################

# Where are the calculations?
calcs_dir=os.path.dirname(os.path.abspath(__file__))

# Get list of calculations
all_file_in_local_dir=glob.glob(os.path.join(calcs_dir,"calcs","RAA*.txt"))

subdir_regex_file = re.compile("RAA_AuAu200_(.+)_cut2_coef(.+).txt")

tmp_dict={}

for tmp_file in all_file_in_local_dir:

    if (not os.path.isfile(tmp_file)):
        continue

    filename=os.path.basename(tmp_file)

    match=subdir_regex_file.match(filename)
    if (match != None)and(filename != None):
        obs_name=match.group(1)
        kappa=match.group(2)

        if (not kappa in tmp_dict.keys()):
            tmp_dict[kappa]={obs_name:tmp_file}
        else:
            tmp_dict[kappa][obs_name]=tmp_file


# Get observable list, and make sure all observables are available for all design points
obs_list=[]
for n, (design_pt, obs_dict) in enumerate(tmp_dict.items()):

    for obs in obs_dict.keys():

        if (0 == n):
            obs_list.append(obs)
        elif (not obs in obs_list):
                print("Mismatch in observable list.")
                exit(1)

print(obs_list)

#########################################
# Make interpolator for each observable #
#########################################

interp_dict={}
# For each observable
for obs in obs_list:

    pT_list_ref=[]
    design_pt_list=[]
    obs_value_list=[]
    obs_err_list=[]
    # Loop over design points
    for n, (design_pt, obs_dict) in enumerate(tmp_dict.items()):

        filename=obs_dict[obs]
        pT_list, obs_val, obs_err=np.loadtxt(filename).T
        design_pt_list.append(float(design_pt))
        obs_value_list.append(obs_val)
        obs_err_list.append(obs_err)

        if (0 == n):
            pT_list_ref=pT_list
        elif np.any(pT_list != pT_list_ref):
            print("Mismatch in pT binning")
            exit(1)

    # I assume that the pT binning is the same for each design points!
    #scipy.interpolate.interp2d(x, y, z, kind='linear', copy=True, bounds_error=False, fill_value=None)
#    print(pT_list_ref,design_pt_list,obs_value_list,obs_err_list)
    interp_dict[obs]={
        'mean':scipy.interpolate.interp2d(design_pt_list, pT_list_ref, np.transpose(obs_value_list), kind='linear', copy=True, bounds_error=False, fill_value=None),
        'err':scipy.interpolate.interp2d(design_pt_list, pT_list_ref, np.transpose(obs_err_list), kind='linear', copy=True, bounds_error=False, fill_value=None)
    }



# How to use the interpolator: 
# f = interp_dict[obs]['mean']
# observable value = f(design_value, p_T)


#########################
# Compute the posterior #
#########################

# Under the approximations that we're using, the posterior is 
# exp(-1/2*\sum_{observables, pT} (model(observable,pT)-data(observable,pT))^2/(model_err(observable,pT)^2+exp_err(observable,pT)^2)

# Here 'kappa' is the only model parameter
def posterior(kappa):

    res=0.0

    # Sum over observables
    for obs in obs_list:

        model_mean_interp=interp_dict[obs]['mean']
        model_err_interp=interp_dict[obs]['err']

        data_pT_list=data_dict[obs]['pT']
        data_mean_list=data_dict[obs]['mean']
        data_err_list=data_dict[obs]['err']

        # Sum over p_T
        for n, pT in enumerate(data_pT_list):
            tmp_model_mean=model_mean_interp(kappa, pT)
            tmp_model_err=model_err_interp(kappa, pT)
            #print(kappa, pT, tmp_model_mean, tmp_model_err)

            data_mean=data_mean_list[n]
            data_err=data_err_list[n]

            res+=np.power(tmp_model_mean-data_mean,2)/(tmp_model_err*tmp_model_err+data_err*data_err)

    res*=-0.5

    return np.exp(res)

##################
# Plot posterior #
##################

font = {'family' : 'URW Gothic',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

####################
### Plot spectra ###
####################
plt.figure()
plt.xscale('linear')
plt.yscale('linear')
#plt.xlim(0,2)
#plt.ylim(1e-5,1e2)
plt.xlabel(r'$\kappa$')
plt.ylabel(r'Posterior')

kappa_range=np.arange(0.0, 2.0, 0.01)
posterior = [ posterior(kappa) for kappa in kappa_range ]

plt.plot(kappa_range, posterior, "-", color='black', lw=4)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#plt.legend(loc='upper right',fontsize=16)
plt.tight_layout()
plt.savefig("posterior.pdf")
plt.show()
