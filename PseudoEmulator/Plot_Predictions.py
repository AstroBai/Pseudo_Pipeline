# 31/08/2018, B. M. Giblin, PhD Student Edinburgh
# Plot the predictions for a given suite of pseudo-simulations defined by
# Seed, CPrior, Basis, Data and Nodes

import numpy as np
from pyDOE import lhs
import pylab as plt
import matplotlib
from matplotlib import rc
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import sys
import os
from scipy.interpolate import interp1d
rc('text',usetex=True)
rc('font',size=24)
rc('legend',**{'fontsize':30})
rc('font',**{'family':'serif','serif':['Computer Modern']})

Dewiggle = False
TroubleShoot = False	# Used to investigate the bad predictions
Fixed_Shape = True
if TroubleShoot:
	DIR += '/TroubleShoot'
	if Fixed_Shape:
		Fix_ID = 1						# The fixed ID of the shape for cycling through cosmologies
		sfx = '_SID%s' %Fix_ID			# suffix for prediction filename
	else:
		Fix_ID = 1						# The fixed ID of the cosmology for cycling through shapes
		sfx = '_CID%s' %Fix_ID			# suffix for prediction filename
else:
	sfx = ''



# PARAMS TO SET
CPrior = "Final"	# Range of cosmology nodes. MiraTitan is wider than Planck18
Basis = "Mixed"		# Read these basis functions... 
Data = "Ultimate12.5"		# ...and project them with these curves,... 
						# ...to get acceptable range of weights

Redshift = ['0.000']																	# USE THIS FOR A SINGLE REDSHIFT
#f = open('/home/bengib/PseudoEmulator/Training_Set/Redshifts_0-2_41.dat')		# OR THESE... 
#Redshift = filter(None,f.read().split('\n'))										# ...FOR MULTIPLE REDSHIFTS

cosmol_dim = 5
weight_dim = 8
wiggle_dim = 0
dimensions = cosmol_dim + weight_dim + wiggle_dim
Seed=2         #sys.argv[1]		 # 2 for trial, 1 for training	
Mx="1.0" 		# "1.0" for trial, "Optimal" for training			

OnAGrid = False 			# Set True if making predictions for the trial grids, False if for regular trial set
if OnAGrid:
	DIR = 'Training_Set/TwoD_Grids/Dim0-1_Ommh2-Ombh2/Resolution50/' 
	Nodes = 2500
else:
	DIR = 'Trial_Set' #'Training_Set' # 'Trial_Set'
	Nodes = 200


# 27/05/2019: Want to make a training set optimised in 13D, NOT 15D.
# So intoroduce a new variable to facilitate this:
if wiggle_dim == 2:
	Nodes_Label = "%s" %Nodes
elif wiggle_dim == 0:
	# IF you're reading in 13D LHC, WHICH ONE (there's 11)?
	# THE ID NEEDS TO BE GIVEN ON THE COMMAND LINE
	try:
		lhc_id = Seed #sys.argv[1]
	except IndexError:
		print( "NO ID GIVEN ON THE COMMAND LINE FOR 13D LHC (0-10)" )
		print( "Either give an ID or set wiggle_dim back to 2." )
		sys.exit()
	Nodes_Label = Nodes #"%sd%s_%s" %(Nodes,dimensions,lhc_id)
else:
	print( "wiggle_dim is not set to 0 or 2. Set it to something sensible." )
	sys.exit()




k = np.loadtxt('%s/Predictions/z0.000/NLPK_Seed%sMx%s_CP%s_BF%s-Data%s_ID0of%s%s.dat' %(DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label,sfx), usecols=(0,), unpack=True)		 
NLPK = np.zeros([Nodes,len(Redshift),len(k)])			# NLCDM, NON-linear shape-modified predictions
LPK = np.empty_like(NLPK)								# NLCDM, LINEAR shape-modified predictions
Boost = np.empty_like(NLPK)								# Ratio
weepers = []
# Make stacked k-arrays, to pickle with the predictions
krep_rtimes = np.repeat(np.reshape(k,(1,-1)), len(Redshift), axis=0)		# k-array "Redshift" times stacked: used for pickled arrays.
krep = np.repeat( np.reshape(krep_rtimes, (1,krep_rtimes.shape[0],krep_rtimes.shape[1])), Nodes, axis=0 )		
																			# k-array [Nodes,Redshift,k] dimensionality: used to pickle.
for r in range(len(Redshift)):
	print( "Read in Redshift number %s of %s" %(r,len(Redshift)) )
	for i in range(Nodes):
		try:
			ktmp1, NLPKtmp = np.loadtxt('%s/Predictions/z%s/NLPK_Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s%s.dat' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,i,Nodes_Label,sfx), usecols=(0,1), unpack=True) 
			ktmp2, LPKtmp = np.loadtxt('%s/Predictions/z%s/LPK_Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s%s.dat' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,i,Nodes_Label,sfx), usecols=(0,1), unpack=True) 
			NLPK[i,r,:] = np.interp(k, ktmp1, NLPKtmp)
			LPK[i,r,:] = np.interp(k, ktmp2, LPKtmp)
		except IOError:
			print( '%s/Predictions/z%s/LPK_Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s%s.dat' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,i,Nodes_Label,sfx) ) #"Node %s is missing." %i )
			weepers.append(i) 
			NLPK[i,r,:] = np.ones(len(k))*np.nan
			LPK[i,r,:] = np.ones(len(k))*np.nan
		Boost[i,r,:] = NLPK[i,r,:] / LPK[i,r,:]
		# save Boost
		np.savetxt('%s/Predictions/z%s/Ratio_Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s%s.dat' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,i,Nodes_Label,sfx), np.c_[k,Boost[i,r,:]], header='# k[h/Mpc], Boost (P^NLCDM_NL/P^NLCDM_L)')
		# and LogBoost
		np.savetxt('%s/Predictions/z%s/LogRatio_Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s%s.dat' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,i,Nodes_Label,sfx), np.c_[k,np.log(Boost[i,r,:])], header='# k[h/Mpc], logBoost (P^NLCDM_NL/P^NLCDM_L)')

	np.save('%s/Predictions/z%s/Ratio_Seed%sMx%s_CP%s_BF%s-Data%s_IDAllof%s%s' %(DIR,Redshift[r],Seed,Mx,CPrior,Basis,Data,Nodes_Label,sfx), np.stack((krep[:,r,:],Boost[:,r,:])) )
if len(Redshift) > 1:
	if not os.path.exists('%s/Predictions/zAll'%DIR):
		os.makedirs('%s/Predictions/zAll'%DIR)
	np.save('%s/Predictions/zAll/Ratio_Seed%sMx%s_CP%s_BF%s-Data%s_IDAllof%s%s' %(DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label,sfx), np.stack((krep, Boost)) )


# Read in physical models to over-plot...:
ModelWiggles = True
if ModelWiggles:
	WigglesLabel = 'w'
else:
	WigglesLabel = 'nw'
import glob
physmoddir = 'Physical_Models/'
F = open('%s/Nodes/Filenames_%s.txt'%(physmoddir,WigglesLabel))
AllModelNames = []
for line in F:
	AllModelNames.append(line.split('\n')[0])

kModels = np.loadtxt('%s/Predictions/z0.000/NLPK_%s_BF%s.dat'%(physmoddir,AllModelNames[0],Basis), usecols=(0,), unpack=True)
NLPK_Models = np.zeros([len(AllModelNames), len(Redshift), len(kModels) ]) 
LPK_Models = np.zeros_like(NLPK_Models)
Shape_Models = np.zeros_like(NLPK_Models)

for i in range(len(AllModelNames)):
	for r in range(len(Redshift)):
		zMod = Redshift[r]
		try:		# For the physical models, we only have z=0 & 1, so this will fail for other z's
			ktmp, mtmp = np.loadtxt('%s/Predictions/z%s/NLPK_%s_BF%s.dat'%(physmoddir,zMod,AllModelNames[i],Basis), usecols=(0,1), unpack=True)
		except IOError:
			print( " NOTE: z%s DOESNT EXIST FOR PHYSICAL MODEL %s, SO READING IN z=0 SHAPES AND PREDICTIONS." %(Redshift[r],AllModelNames[i],) )
			zMod = '0.000'		# If the shapes don't exist for the physical models, just read in z=0

		ktmp, mtmp = np.loadtxt('%s/Predictions/z%s/NLPK_%s_BF%s.dat'%(physmoddir,zMod,AllModelNames[i],Basis), usecols=(0,1), unpack=True)
		NLPK_Models[i,r,:] = np.interp(kModels, ktmp, mtmp)
		ktmp, mtmp = np.loadtxt('%s/Predictions/z%s/LPK_%s_BF%s.dat'%(physmoddir,zMod,AllModelNames[i],Basis), usecols=(0,1), unpack=True)
		LPK_Models[i,r,:] = np.interp(kModels, ktmp, mtmp)	
		# save boost
		np.savetxt('%s/Predictions/z%s/Ratio_%s_BF%s.dat' %(physmoddir,zMod,AllModelNames[i],Basis),
		           np.c_[kModels,NLPK_Models[i,r,:]/LPK_Models[i,r,:]],
		           header='# k[h/Mpc], Boost (P^NLCDM_NL/P^NLCDM_L)')
		# and the log-boost:
		np.savetxt('%s/Predictions/z%s/LogRatio_%s_BF%s.dat' %(physmoddir,zMod,AllModelNames[i],Basis),
		           np.c_[kModels, np.log(NLPK_Models[i,r,:]/LPK_Models[i,r,:])],
		           header='# k[h/Mpc], log-Boost (P^NLCDM_NL/P^NLCDM_L)')
                
		ktmp, mtmp = np.loadtxt('%s/Shapes/z%s/%s.dat'%(physmoddir,zMod,AllModelNames[i]), usecols=(0,1), unpack=True)
		Shape_Models[i,r,:] = np.interp(kModels, ktmp, mtmp)	


#print("Exiting before plotting...")
#sys.exit()


Rn = 0 # Which redshift to plot
lw = 2
f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex='col', figsize=(9,9))
for i in range(Nodes):
	ax1.loglog(k, LPK[i,Rn,:], color='dimgrey')
ax1.set_xlim([1.e-4, 10.])
ax1.set_ylim([1., 3e5])
#ax1.set_xlabel(r'$k$ [$h/$Mpc]')
ax1.set_xticks([])
ax1.set_ylabel(r'$P^{\rm{pseudo}}_{\rm{L}}(k)$') #%float(Redshift[Rn])
for i in range(len(AllModelNames)):
	ax1.loglog(kModels, LPK_Models[i,Rn,:], color='orange', linewidth=lw)

for i in range(Nodes):
	ax2.loglog(k, NLPK[i,Rn,:], color='dimgrey')
ax2.set_xlim([1.e-4, 10.])
ax2.set_ylim([1., 3e5])
ax2.set_xticks([])
#ax2.set_xlabel(r'$k$ [$h/$Mpc]')
ax2.set_ylabel(r'$P^{\rm{pseudo}}_{\rm{NL}}(k)$')
for i in range(len(AllModelNames)):
	ax2.loglog(kModels, NLPK_Models[i,Rn,:], color='orange')

for i in range(Nodes): 
	ax3.plot(k, np.log(NLPK[i,Rn,:] / LPK[i,Rn,:]), color='dimgrey')
for i in range(len(AllModelNames)):
	ax3.plot(kModels, np.log(NLPK_Models[i,Rn,:]/LPK_Models[i,Rn,:]), color='orange', linewidth=lw)
ax3.set_xscale('log')
ax3.set_ylim([-0.1, 4.8])
ax3.set_xlim([1.e-4, 10.])
ax3.set_xlabel(r'$k$ [$h/$Mpc]')
ax3.set_ylabel(r'$\ln B(k)$')	#(P^{\rm{N{\Lambda}CDM}}_{\rm{NL}} / P^{\rm{{\Lambda}CDM}}_{\rm{NL}})
legtrs = mlines.Line2D([],[],color='dimgrey', label=r'%s set'%DIR.split('/')[-1].split('_')[0], linewidth=lw )
legmod = mlines.Line2D([],[],color='orange', label=r'Physical', linewidth=lw)
ax3.legend(handles=[legtrs,legmod], loc='upper left', frameon=False)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('%s/Predictions/z%s/Plot_Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s%s.png' %(DIR,Redshift[Rn],Seed,Mx,CPrior,Basis,Data,Nodes_Label,sfx))
#plt.show() 


def Plot_PhysMods_Only():

	# Read in a single LCDM prediction to plot as well...
	ktmp, mtmp = np.loadtxt('%s/Predictions/z0.000/LVK_%s_BF%s.dat'%(physmoddir,AllModelNames[0],Basis), usecols=(0,1), unpack=True)
	LVK_Models = np.interp(kModels, ktmp, mtmp)

	ktmp, mtmp = np.loadtxt('%s/Predictions/z0.000/NLVK_%s_BF%s.dat'%(physmoddir,AllModelNames[0],Basis), usecols=(0,1), unpack=True)
	NLVK_Models = np.interp(kModels, ktmp, mtmp)

	plot_color = ['dimgrey', 'magenta']
	lw = 3
	f, ((ax2, ax3)) = plt.subplots(2, 1, sharex='col', figsize=(9,8))

	ax2.set_xlim([1.e-4, 10.])
	ax2.set_ylim([1., 9e4])
	ax2.set_xticks([])
	#ax2.set_xlabel(r'$k$ [$h/$Mpc]')
	ax2.set_ylabel(r'$P^{\rm{pseudo}}_{\rm{NL}}(k)$')
	for i in range(len(AllModelNames)):
		ax2.loglog(kModels, NLPK_Models[i,0,:], color=plot_color[0], linestyle='-', linewidth=lw)
	ax2.loglog(kModels, NLVK_Models, color='red', linestyle='-', linewidth=lw)

	for i in range(len(AllModelNames)):
		ax3.plot(kModels, np.log(NLPK_Models[i,0,:]/LPK_Models[i,0,:]), color=plot_color[0], linestyle='-', linewidth=lw)
	ax3.plot(kModels, np.log(NLVK_Models/LVK_Models), color='red', linestyle='-', linewidth=lw)
	
	ax3.set_xscale('log')
	ax3.set_ylim([-.4, 3.9])
	ax3.set_xlim([1.e-4, 10.])
	ax3.set_xlabel(r'$k$ [$h/$Mpc]')
	ax3.set_ylabel(r'$\ln B(k)$')	
	legtrs = mlines.Line2D([],[],color=plot_color[0], linestyle='-', label=r'$z=%.0f$'%float(Redshift[0]), linewidth=lw )
	legmod = mlines.Line2D([],[],color=plot_color[1], linestyle='-', label=r'$z=%.0f$'%float(Redshift[-1]), linewidth=lw)
	#ax3.legend(handles=[legtrs,legmod], loc='upper left', frameon=False)
	plt.subplots_adjust(wspace=0, hspace=0)
	#plt.savefig('%s/Predictions/z%s/Plot_AllModelPkBk_BF%s.png' %(physmoddir,Redshift[Rn],Basis))
	plt.show() 
	return
Plot_PhysMods_Only()













