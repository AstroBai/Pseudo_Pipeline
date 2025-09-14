# 28/08/2018, B. M. Giblin, PhD Student Edinburgh 
# Generate a Latin Hypercube (LHC) of cosmol & weight params for input to CAMB
# LHC has to satisfy condition shapes stay +ve: so generate 2*required number and pick those
# that satisy this condition

import os
import numpy as np
from pyDOE import lhs
import time
import sys
from scipy.spatial.distance import euclidean

import matplotlib
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#overall_DIR = 'Training_Set'
overall_DIR = 'Trial_Set'

Read_Optimal_LHC = False		#If False, it will make an LHC. If True it'll read in an Optimal
Optimise_ED = False 		# If True, it will cycle through experimental designs (EDs)
EDs = range(200)			# IDs of the EDs it will cycle though if Optimise_ED is True

# PARAMS TO SET
CPrior = "Final"			# Range of cosmology nodes. MiraTitan is wider than Planck18
		#MiraTitan
Basis = "Mixed"			# Read these basis functions: "Mixed" or "GPCurves"
Data = "Ultimate12.5"			# What scaling of the weight nodes to use. 
                                        # "LCDM" will use the "Ultimate" scaling, but manually set un-scaled weights dimensions to 0.
Narrow_Range = True			# THIS IS THE VARIABLE THAT THE CODE NOW USES TO FIGURE OUT IF IT SHOULD NARROW PRIOR RANGE

Seed=2 #int(sys.argv[1]) #1
Nodes = 200
Mx = 1.0   			# Generate this fraction X Nodes to overcompensate for shapes that may come out negative.
cosmol_dim = 5
weight_dim = 8 		
wiggle_dim = 0
dimensions = cosmol_dim + weight_dim + wiggle_dim


# 27/05/2019: Want to make a training set optimised in 13D, NOT 15D.
# So intoroduce a new variable to facilitate this:
Nodes_Label = "%s" %Nodes 
#if wiggle_dim == 2:
#	Nodes_Label = "%s" %Nodes
#elif wiggle_dim == 0:
	# IF you're reading in 13D LHC, WHICH ONE (there's 11)?
	# THE ID NEEDS TO BE GIVEN ON THE COMMAND LINE
#	try:
#		lhc_id = sys.argv[1]
#	except IndexError:
#		print( "NO ID GIVEN ON THE COMMAND LINE FOR 13D LHC (0-10)" )
#		print( "Either give an ID or set wiggle_dim back to 2." )
#		sys.exit()
#	Nodes_Label = "%sd%s_%s" %(Nodes,dimensions,lhc_id)
#else:
#	print( "wiggle_dim is not set to 0 or 2. Set it to something sensible." )
#	sys.exit()





#### !!! Priors to scale the nodes !!! ####				
# !GIBLIN NEW RANGES BELOW, SWITCH TO PHYSICAL PARAMS IN CAMB

# 1. Scaling of cosmol nodes.
# We need to be careful and make sure omegas sum to <= 1 
# (Flat LCDM; omega_neutrino is the missing piece to round up to 1)
if CPrior == "Planck18":
	# Roughly taken from Fig. 5 of https://arxiv.org/pdf/1807.06209.pdf
	omega_m = [0.26, 0.38] 		# omm = [0.12, 0.155]		# gib these are now little omh^2 
	omega_baryon = [0.035, 0.068]	# omb = [0.0215, 0.0235]
	hubble = [62., 74.] 		# h = [60., 80.]
	n_s = [0.92, 1.02]		# n_s = [0.9, 1.05]
	A_s = np.exp(np.array([2.9, 3.15])) / 1e10		#[2.92, 3.16]	
if CPrior == "MiraTitan":
	omm,omb,h,ns,omn = np.loadtxt('/home/bengib/MiraTitan/MT_Cosmol_Table.txt', usecols=(1,2,4,5,8), unpack=True)
	omega_m = [np.min(omm), np.max(omm)] 
	omega_baryon = [np.min(omb), np.max(omb)]
	hubble = [np.min(h)*100., np.max(h)*100.]
	n_s = [np.min(ns), np.max(ns)]
	A_s = np.exp(np.array([2.9, 3.15])) / 1e10 		#same as Planck
if CPrior == "Final":
	# Given to me by Matteo
	omega_m = [0.12, 0.155]		# !!! NOW little ommh^2 !!!
	omega_baryon = [0.0215, 0.0235]	# !!! NOW little ombh^2 !!!
	hubble = [60., 80.]
	n_s = [0.9, 1.05]
	A_s = np.exp(np.array([2.92, 3.16])) / 1e10			


Cosmol_Priors = np.vstack(( omega_m, omega_baryon, hubble, n_s, A_s ))		
#lhc = np.delete(lhc, -1,-1) # remove neutrino column of lhc


# 2. scaling of weight nodes.
import pickle
import glob
if Basis == "GPCurves" or Basis == "Mixed":
	# Get the basis functions
	T = np.array( pickle.load(open('%s/BasisFunctions%s_Amp5.0_lnp2.0.pkl'%(Basis,weight_dim),'rb'), encoding='latin1'))
else:
	T = np.array( pickle.load(open('%s/BasisFunctions%s.pkl'%(Basis,weight_dim),'rb'), encoding='latin1'))
k = T[0,:,0]
BFs = T[:,:,1]

# Get the curves that made the BFs: mean is required to do PCA of the "data"
fname = glob.glob('%s/*000_random_curves.pkl'%(Basis))
D = np.array( pickle.load(open(fname[0],'rb'), encoding='latin1'))
Curves = D[:,:,1]
Curves_MinusMean = np.empty_like(Curves)				
Curves_Mean = np.empty( len(k) )
for i in range(len(Curves[0,:])):
	Curves_Mean[i] = np.mean(Curves[:,i])
	Curves_MinusMean[:,i] = Curves[:,i] - Curves_Mean[i]					
W = np.dot(Curves_MinusMean,np.transpose(BFs))		
Weight_Priors = np.empty([weight_dim, 2])
for i in range(weight_dim):
	Weight_Priors[i,:] = [ np.min(W[:,i]), np.max(W[:,i]) ]

if Data == "Manual":		# Redefine acceptable weights
	bounds = [20., 10., 5., 1., 0.2, 0.05, 0.01, 0.005] # may still be a bit liberal (based on GPCurves narrowed a bit) 
	for i in range(weight_dim):
		Weight_Priors[i,:] = [ -1.*bounds[i], bounds[i] ] 
if Data == "Final":
	upp_bounds = [0., 3.0, 2., 1.5, 1., 0.25, 0.25, 0.25]
	low_bounds = [-20., -3.0, -2., -1.5, -1., -0.25, -0.25, -0.25]
	for i in range(weight_dim):
		Weight_Priors[i,:] = [ low_bounds[i], upp_bounds[i] ] 
elif "Ultimate" in Data or "LCDM" in Data:
#Data == "Ultimate" or Data == "LCDM" or len(Data.split('Ultimate')[-1]) > 0 or len(Data.split('LCDM')[-1]) > 0:
	upp_bounds = [0., 1.5, 1.2, 0.75, 0.5, 0.25, 0.2, 0.1]
	low_bounds = [-20., -1.5, -1.2, -0.75, -0.5, -0.25, -0.2, -0.1]
	for i in range(weight_dim):
		Weight_Priors[i,:] = [ low_bounds[i], upp_bounds[i] ] 
	# Keep the priors as these if doing a LCDM run: we'll just manually set weight nodes to 0.

Wiggle_Priors = np.array([ [0.,1.], [-0.65, 0.1] ])
if wiggle_dim == 2:
	Priors = np.vstack(( Cosmol_Priors, Weight_Priors, Wiggle_Priors ))
else:
	Priors = np.vstack(( Cosmol_Priors, Weight_Priors ))

if Narrow_Range:
	#len(Data.split('Ultimate')[-1]) > 0 or len(Data.split('LCDM')[-1]) > 0:
	if "Ultimate" in Data:
		scalefactor = float(Data.split('Ultimate')[-1]) / 100.
	elif "LCDM" in Data:
		scalefactor = float(Data.split('LCDM')[-1]) / 100.
	Priors_Orig = np.copy(Priors)
	# Narrow range
	Priors[:,0] += scalefactor * (Priors_Orig[:,1] - Priors_Orig[:,0])
	Priors[:,1] -= scalefactor * (Priors_Orig[:,1] - Priors_Orig[:,0])
	print( "Priors originally set to :" )
	for i in range(Priors.shape[0]):
		print( Priors_Orig[i,0], Priors_Orig[i,1] )
	print( "Narrowed Priors by factor %s:" %scalefactor )
	for i in range(Priors.shape[0]):
		print( Priors[i,0], Priors[i,1] )

#### !!! Priors to scale the nodes !!! ####



#### NOW EITHER MAKE OR READ IN A DIMENSIONLESS LHC, THEN SCALE IT BY THE PRIORS ####

def Check_If_LHC_Is_Okay(lhc,lhc_scaled):
	# Finally, scroll through and only keep the nodes which generate +ve shapes
	lhc_refined = np.empty([ Nodes, dimensions ])
	lhc_scaled_refined = np.empty([ Nodes, dimensions ])
	Shapes = np.empty([ Nodes, len(k) ])
	keepers = 0
	weepers = [] # store the elements of the neg shapes
	for i in range(lhc_scaled.shape[0]):
		if keepers == Nodes:
			break		# got enough nodes.
		# Make shape
		S = np.copy(Curves_Mean)
		for j in range(weight_dim):
			S += lhc_scaled[i,cosmol_dim+j] * BFs[j,:]
		if wiggle_dim == 2:
			S += (lhc_scaled[i,-2] + lhc_scaled[i,-1]*np.log10(k)) * BAO_shape


		if np.any( S <= 0.0 ) == False:
			lhc_refined[keepers,:] = lhc[i,:]
			lhc_scaled_refined[keepers,:] = lhc_scaled[i,:]		# save node
			Shapes[keepers,:] = S
			keepers+=1
		else:
			weepers.append(i)
			if Mx == "Optimal":
				# Save even the negative shapes, just to keep all the nodes
				lhc_refined[keepers,:] = lhc[i,:]
				lhc_scaled_refined[keepers,:] = lhc_scaled[i,:]		# save node
				Shapes[keepers,:] = S
				keepers+=1

	return lhc_refined, lhc_scaled_refined, Shapes, keepers, weepers



def Generate_Acceptable_LHC(Seed):
	np.random.seed(Seed)	# Make it reproducable 
	lhc = lhs(dimensions, samples=int(Mx*Nodes), criterion='maximin') #criterion=None)    # dimensionless nodes Nodes*[0,1]
					# See https://pythonhosted.org/pyDOE/randomized.html
					# for more options for criterion
					# generate 2*Nodes, and only keep Nodes which have +ve shapes
	
	lhc_scaled = np.empty_like(lhc)
	for i in range(dimensions):
		lhc_scaled[:,i] = Priors[i,0] + (Priors[i,1] - Priors[i,0])*lhc[:,i]

	if "LCDM" in Data:		
		# It's not correct to just set the weights equal to zero.
		# You need to set the weights such that W.BFs = 1-Curves_Mean
		# Since the Weights signify deviations about the Curves_Mean.
		# If you set Weights thus, all Shapes come out unity, and you recover LCDM.
		for i in range(Nodes):
			lhc_scaled[i,cosmol_dim:cosmol_dim+weight_dim] = np.dot((1.-Curves_Mean),np.transpose(BFs))
			# reset lhc to match
			for j in range(cosmol_dim,cosmol_dim+weight_dim):
				if len(Data.split('LCDM')[-1]) > 0: # If True, you've narrowed range of LCDM priors.
								 # but you'll later scale them by the original priors
								 # To keep things consistent, must do that here too.
					lhc[i,j] = (lhc_scaled[i,j]-Priors_Orig[j,0]) / (Priors_Orig[j,1]-Priors_Orig[j,0])
				else:
					lhc[i,j] = (lhc_scaled[i,j]-Priors[j,0]) / (Priors[j,1]-Priors[j,0])
			if wiggles_dim == 2:
				# And get rid of wiggles which deviate shape from unity
				lhc_scaled[i,-2:] = np.zeros(2)
				lhc[i,-2:] = np.zeros(2)

	lhc_refined, lhc_scaled_refined, Shapes, keepers, weepers = Check_If_LHC_Is_Okay(lhc,lhc_scaled)
	
	if "LCDM" in Data:
		# The Shapes recovered differ from unity by <0.1%, but let's set them to unity to be sure.
		Shapes = np.ones_like(Shapes)

	if keepers < Nodes:
		print( "There was not enough acceptable nodes. Only %s of %s generated +ve shapes. NOT SAVING THEM!!!" %(keepers,Nodes) )
		sys.exit()		
	return lhc_scaled_refined, lhc_refined, Shapes


# Read in the BAO-template for getting the shapes correct
# GIBLIN ! NOTE THIS IS THE CORRECT TEMPLATE TO READ IN. !
ktmp, BAOtmp = np.loadtxt('Physical_Models/BAO_shape.dat', usecols=(0,1),unpack=True)
BAO_shape = np.interp(k, ktmp, BAOtmp)

if Read_Optimal_LHC :
	print( "Reading in the Optimal LHC of %s Nodes..." %Nodes_Label )
	Mx="Optimal"
	lhc = np.loadtxt('Training_Set/Optimal_LHCs/maxpro_%s.csv'%Nodes_Label, unpack=True, skiprows=1, delimiter=',').transpose()
	lhc_scaled = np.empty_like(lhc)
	for i in range(dimensions):
		lhc_scaled[:,i] = Priors[i,0] + (Priors[i,1] - Priors[i,0])*lhc[:,i]
	lhc_refined, lhc_scaled_refined, Shapes, keepers, weepers = Check_If_LHC_Is_Okay(lhc,lhc_scaled)
	if len(weepers) > 0:
		print( "There was not enough acceptable nodes. Nodes %s of %s generated -ve shapes. But we'll leave in the neg shapes anyway." %(weepers,Nodes) )
	#	for i in range(len(weepers)):
	#		lhc_refined = np.delete(lhc_refined, -1, axis=0)	
	#		lhc_scaled_refined = np.delete(lhc_scaled_refined, -1, axis=0)
	#		Shapes = np.delete(Shapes, -1, axis=0)			

else:
	t1 = time.time()
	if Optimise_ED:
		min_distance = np.empty(len(EDs))
		for e in EDs:
			tmp_min_distance = 1e9 			# Initial massive Euclidean distance
			lhc_scaled_refined, lhc_refined, Shapes = Generate_Acceptable_LHC(e)
			print( "On ED %s" %e )
			for i in range(0,Nodes-1):
				for j in range(i+1,Nodes):
					#distance = 	euclidean(lhc_refined[i,:], lhc_refined[j,:])
					#distance = 	np.sqrt( np.sum( (lhc_refined[i,:]-lhc_refined[j,:])**2. ) ) # 1.8 times faster than scipy euclidean.
					distance = np.linalg.norm(lhc_refined[i,:]-lhc_refined[j,:]) 				 # 2.3 times faster than scipy euclidean.
					if distance < tmp_min_distance:
						tmp_min_distance = distance
			if np.isfinite(tmp_min_distance):
				min_distance[e] = tmp_min_distance
			else:
				min_distance[e] = 1.e9	
		best_ED = np.argmax(min_distance)
		print( "The best experimental design is number %s. Changing Seed to this." %best_ED )
		Seed = best_ED
		np.random.seed(best_ED)
		lhc_scaled_refined, lhc_refined, Shapes = Generate_Acceptable_LHC(best_ED)

	else:
		lhc_scaled_refined, lhc_refined, Shapes = Generate_Acceptable_LHC(Seed)

	t2 = time.time()
	print( "Time to make the LHC with Optimise_ED set to %s and %s EDs is %s s" %(Optimise_ED,len(EDs),(t2-t1)) )


# Assemble fmt string - NEED TO HAVE A_s COLUMN IN SCIENTIFIC FORMAT
FMT=''
for i in range(dimensions):
	if i == 4:
		FMT += '%.7e '
	else:
		FMT += '%.7f '

directory = overall_DIR
if not os.path.exists(overall_DIR + '/Nodes'):
        os.makedirs(directory + '/Nodes')
        os.makedirs(directory + '/Shapes')
        os.makedirs(directory + '/Predictions')

# Save the LHCs & Shapes
np.savetxt('%s/Nodes/Seed%sMx%s_Nodes%s_Dim%s.dat'%(overall_DIR,Seed,Mx,Nodes_Label,dimensions),
			lhc_refined, fmt='%.4f', header='# om_m, om_b, H, ns, As')		# Dimensionless

# Dimensions: w/wo header
np.savetxt('%s/Nodes/Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_Dim%s.dat'%(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label,dimensions),
			lhc_scaled_refined, fmt=FMT, header='# Om_m, Om_b, H, ns, As, Weights...')
np.savetxt('%s/Nodes/Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_Dim%s_NoHeader.dat'%(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label,dimensions),
			lhc_scaled_refined, fmt=FMT)

# Save a version with the cosmol nodes scaled to [0,1] 
# and the weight nodes scaled to range of first weight.
# Need to remove the scaling of Priors in this case...
if Narrow_Range:
#len(Data.split('Ultimate')[-1]) > 0 or len(Data.split('LCDM')[-1]) > 0:
	Priors = np.copy(Priors_Orig)
lhc_scaled_refined2 = np.copy(lhc_scaled_refined)
for i in range(dimensions):
	lhc_scaled_refined2[:,i] = (lhc_scaled_refined[:,i] - Priors[i,0]) / abs(Priors[i,1] - Priors[i,0])
	#if i >= cosmol_dim and i < cosmol_dim+weight_dim :
	#	lhc_scaled_refined2[:,i] = (lhc_scaled_refined[:,i] - Priors[cosmol_dim,0]) / abs(Priors[cosmol_dim,1] - Priors[cosmol_dim,0])

# 13/08/2024: rm'd scaling of weights to range of w1. They're now scaled to range in each indiv. weight dim.
# This is coz it makes no diff to emu acc, and makes optimisation algorithm simpler.
np.savetxt('%s/Nodes/Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_Dim%s_Scaled.dat'%(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label,dimensions),
			lhc_scaled_refined2,fmt='%.6f')


for ID in range(Shapes.shape[0]):
	np.savetxt('%s/Shapes/Seed%sMx%s_CP%s_BF%s-Data%s_ID%sof%s.dat'%(overall_DIR,Seed,Mx,CPrior,Basis,Data,ID,Nodes_Label), np.c_[k,Shapes[ID,:]])

def Visualise_LHC(coords):

	import pylab as plt
	import matplotlib
	from matplotlib import rc
	import matplotlib.gridspec as gridspec
	rc('text',usetex=True)
	rc('font',size=18)#18
	rc('legend',**{'fontsize':18})#18
	rc('font',**{'family':'serif','serif':['Computer Modern']})

	# Visualise the cosmologies of the trouble makers...
	fig = plt.figure(figsize = (14,14))
	gs1 = gridspec.GridSpec(5, 2)
	PO = np.array([ [0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4] ])	# Plot_Order
	LABELS = [ r'$\Omega_{\rm{m}}$', r'$\Omega_{\rm{b}}$', r'$H$', r'$n_s$', r'$\log_e[A_s \times 10^{10}]$' ] 
	#cmap = plt.get_cmap('jet')
	#colors = [cmap(i) for i in np.linspace(0, 1, count_inacc)]

	for i in range(PO.shape[0]):
		ax1 = plt.subplot(gs1[i])
		ax1.scatter(coords[:,PO[i,0]], coords[:,PO[i,1]], color='black')
		ax1.set_xlabel(LABELS[PO[i,0]])
		ax1.set_ylabel(LABELS[PO[i,1]])
	plt.savefig('%s//Predictions/Plot_Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_Cos.png' %(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label))
	#plt.show()
	return
	

# Rescale A_s for plotting purposes....
lhc_scaled_refined_plot = np.copy( lhc_scaled_refined )
lhc_scaled_refined_plot[:,4] = np.log(lhc_scaled_refined[:,4]*1e10)
#Visualise_LHC(lhc_scaled_refined_plot)


# Make corner plot
def Node_Corner(coords, labels,savename,limits):

	fig = plt.figure(figsize = (16,10)) #figsize = (20,14)
	gs1 = gridspec.GridSpec(coords.shape[1]-1,coords.shape[1]-1)
	p=0	# subplot number
	for i in range(coords.shape[1]-1):
		l=i+1 # which y-axis statistic is plotted on each row.	
		for j in range(coords.shape[1]-1):
			ax1 = plt.subplot(gs1[p])
			if j>i:
				ax1.axis('off')
			else:
				ax1.scatter(coords[:,j], coords[:,l], color='black',s=5)

				# Decide what the axis limits should be. If limits=None, it doesn't set any.
				# If limits is [a,b], limits are set to be a*min and b*max in each dimension.
				# Else, limits is interpreted as [ [x1,x2],[y1,y2],[z1,z2]...] for each dimension. 
				if limits != None:
					if len(np.array(limits).shape) == 1:
						ax1.set_xlim([ limits[0]*coords[:,j].min(),limits[1]*coords[:,j].max() ]) 
						ax1.set_ylim([ limits[0]*coords[:,l].min(),limits[1]*coords[:,l].max() ]) 
					else:
						ax1.set_xlim([ limits[j][0],limits[j][1] ]) 
						ax1.set_ylim([ limits[l][0],limits[l][1] ]) 

				# Set every other x/y-tick to be invisible
				#for thing in ax1.get_xticklabels()[::2]:
				#	thing.set_visible(False)
				#for thing in ax1.get_yticklabels()[::2]:
				#	thing.set_visible(False)
				
				# Get rid of x/y ticks completely for subplots not at the edge
				if j==0:
					ax1.set_ylabel(labels[l])
				else:
					ax1.set_yticks([])
				if i==coords.shape[1]-2:
					ax1.set_xlabel(labels[j])
				else:				
					ax1.set_xticks([])
			p+=1
	plt.savefig(savename)
	plt.show()
	return


rc('text',usetex=True)
rc('font',size=10)#18
rc('legend',**{'fontsize':18})#18
rc('font',**{'family':'serif','serif':['Computer Modern']})
LABELS = [ r'$\Omega_{\rm{m}}h^2$', r'$\Omega_{\rm{b}}h^2$', r'$H$ [kms$^{-1}/$Mpc]', r'$n_s$', r'$\ln[A_s \times 10^{10}]$', r'$w_1$', r'$w_2$', r'$w_3$', r'$w_4$', r'$w_5$', r'$w_6$', r'$w_7$', r'$w_8$', r'$a$', r'$b$']

limits = Priors.tolist()
limits[4][0] = np.log(limits[4][0]*1e10)
limits[4][1] = np.log(limits[4][1]*1e10)
#limits = [ [0.12,0.155], [0.0215,0.0235], [60,80], [0.92,1.02], [2.90,3.15],
#			[-20,0], [-1.5,1.5], [-1.2,1.2], [-0.75,0.75], [-0.25,0.25], [-0.25,0.25], [-0.2,0.2], [-0.1,0.1],
#			[0.,1.], [0.1, 0.65] ] 

savename = '%s/Predictions/Plot_Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_NodeCorner.png' %(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label)
savename2 = '%s/Predictions/Plot_Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_NodeCornerCosOnly.png' %(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label)
savename3 = '%s/Predictions/Plot_Seed%sMx%s_CP%s_BF%s-Data%s_Nodes%s_NodeCornerWeightOnly.png' %(overall_DIR,Seed,Mx,CPrior,Basis,Data,Nodes_Label)
#Node_Corner(lhc_scaled_refined_plot, LABELS,savename, limits)

rc('font',size=16)
rc('legend',**{'fontsize':16})
#Node_Corner(lhc_scaled_refined_plot[:,:cosmol_dim], LABELS,savename2,limits)
#Node_Corner(lhc_scaled_refined_plot[:,cosmol_dim:], LABELS[cosmol_dim:],savename3,limits[cosmol_dim:])

	

