# 29/08/2018, B. M. Giblin, PhD Student, Edinburgh
# Read in a Nodes files specified by Seed, CosmolPrior, BFs, Data, Node-Num, & Dimensions...
# ... and generate the CAMB predictions.
# SPEED = 1500 nodes (LCDM & NLCDM) PER HR

Seed=2 #2 #1
Mx="1.0"
CPrior="Final"		# Range of cosmology nodes. MiraTitan is wider than Planck18
Basis="Mixed"		# Read these basis functions... 
Data="Ultimate12.5"		# ...and project them with these curves,... 
						# ...to get acceptable range of weights

Redshift=(0.000) # 1.000)																	# USE THIS FOR A SINGLE REDSHIFT
#mapfile < Training_Set/Redshifts_0-2_41.dat		# USE THIS... 
#Redshift=${MAPFILE[*]}																# ...AND THIS FOR MULTIPLE REDSHIFTS

Nodes=200	
cosmol_dim=5
weight_dim=8
wiggle_dim=0
dimensions=$((cosmol_dim + weight_dim + wiggle_dim)) 

# 27/05/2019: Want to make a training set optimised in 13D, NOT 15D.
# So introduce a new variable to facilitate this:
Nodes_Label=${Nodes} 
#if [ "$wiggle_dim" -eq 2 ]; then
#	Nodes_Label=${Nodes}
#elif [ "$wiggle_dim" -eq 0 ]; then 
#	lhc_id=$1
#	if [ "$lhc_id" == "" ]; then
#		echo "NO ID GIVEN ON THE COMMAND LINE FOR 13D LHC (0-10)"
#		echo "Either give an ID or set wiggle_dim back to 2."
#		exit
#	fi
#	Nodes_Label="${Nodes}d${dimensions}_${lhc_id}"
#else
#	echo "wiggle_dim is not set to 0 or 2. Set it to something sensible."
#	exit
#fi


# !!! IMPORTANT LINE - WHERE THE OUTPUT IS GOING !!!
#DIR=${PWD}/Training_Set/ 
DIR=${PWD}/Trial_Set/
#DIR=${PWD}/Training_Set/Optimisation/Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}/nu0.19/

# Nodes file
# LHC nodes
mapfile < $DIR/Nodes/Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_Nodes${Nodes_Label}_Dim${dimensions}_NoHeader.dat 
# Optimised nodes!
#mapfile < $DIR/Optimised_Nodes.dat

CAMBDIR=../CAMB-0.1.6.1/
# hacked CAMB needs this library in path to work
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CAMBDIR}/lib


# Make sure we ARE using physical parameterisation (we changed), and non-linear
sed -i 's/^use_physical .*$/use_physical = T/' $CAMBDIR/params.ini
sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $CAMBDIR/params.ini

start=$SECONDS
# Cycle through the Nodes
LOOP=$Nodes
for i in `seq 0 $LOOP`; do

	pfile=$CAMBDIR/params_Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_ID${i}of${Nodes_Label}.ini
	cp $CAMBDIR/params.ini $pfile
	

	var=$(echo "${MAPFILE[$i]}")	# Reads in a line
	var2=($var) 					# Turns line into an array with whitespace delimiter
	# Cosmological params	
	omm=$(echo "${var2[0]}")		
	omb=$(echo "${var2[1]}")
	h=$(echo "${var2[2]}")
	ns=$(echo "${var2[3]}")
	As=$(echo "${var2[4]}")

	omn=0.
	oml=$(expr 1.-$omm | bc)	# Lambda
	omc=$(expr $omm-$omb | bc)	# CDM
	# weight params

	w1=$(echo "${var2[5]}")
	w2=$(echo "${var2[6]}")
	w3=$(echo "${var2[7]}")	
	w4=$(echo "${var2[8]}")	
	w5=$(echo "${var2[9]}")
	w6=$(echo "${var2[10]}")
	w7=$(echo "${var2[11]}")
	w8=$(echo "${var2[12]}")

	echo "Generating CAMB prediction $i of $Nodes ..."
	echo "$omm $omb $h $ns $As"
	

	# Editing the params.ini 
	sed -i "s/^ombh2 .*$/ombh2 = $omb/" $pfile
	sed -i "s/^omch2 .*$/omch2 = $omc/" $pfile
	#sed -i "s/^omega_lambda .*$/omega_lambda = $oml/" $pfile	# Now determined by combo of other omegas
	sed -i "s/^omnuh2 .*$/omnuh2 = 0./" $pfile	# Adding BAO-wiggles manually with (a,b) params.
	sed -i "s/^hubble .*$/hubble = $h/" $pfile
	sed -i "s/^scalar_spectral_index(1) .*$/scalar_spectral_index(1) = $ns/" $pfile
	sed -i "s/^scalar_amp(1) .*$/scalar_amp(1) = $As/" $pfile

	# !!!!!!!! MAKE SURE THIS LINE IS RIGHT GIBBO !!!!!!
	sf=$DIR/Shapes/Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_ID${i}of${Nodes_Label}.dat
	sed -i "s#shapefile .*#shapefile = \'$sf\'#" $pfile

	for z in ${Redshift[*]}; do
		
		echo "REDSHIFT $z ..."
		zlabel=`printf "%.3f" $z`
		OUTDIR=$DIR/Predictions/z$zlabel/
		if [ ! -d "$OUTDIR" ]; then
			mkdir -p $OUTDIR
		fi

		sed -i "s/^transfer_redshift(1) .*$/transfer_redshift(1) = $zlabel/" $pfile				

		# First run CAMB with nlcdm shape NON-LINEAR
		cd $CAMBDIR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 1/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/NLPK_Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_ID${i}of${Nodes_Label}.dat

		# Now with nlcdm shape LINEAR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 0/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = T/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/LPK_Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_ID${i}of${Nodes_Label}.dat


		# Now run with vanilla LCDM ("V") LINEAR
		sed -i 's/^do_nonlinear .*$/do_nonlinear = 0/' $pfile
		sed -i 's/^use_nlcdm_shape .*$/use_nlcdm_shape = F/' $pfile
		./camb $pfile
		mv $CAMBDIR/test_matterpower.dat $OUTDIR/LVK_Seed${Seed}Mx${Mx}_CP${CPrior}_BF${Basis}-Data${Data}_ID${i}of${Nodes_Label}.dat

		done
	rm -f $pfile
	done
cd $DIR

duration=$(( SECONDS - start ))
echo "It took $duration s to generate 2*$LOOP predictions"






