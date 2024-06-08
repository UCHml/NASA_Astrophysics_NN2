import sys
import numpy as np
import h5py
import hdf5plugin
import os
from pathlib import Path

import MAS_library as MASL
import Pk_library as PKL
import redshift_space_library as RSL




GRID = 200


#SUIT_NAME = "IllustrisTNG" 
SET_NAME = "CV"


res_dir = 'output_redshift_space' # directory of the results; Change it!!!

BASE_PATH_TO_SIMS = f"/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/" # change pathname
BASE_PATH_TO_HALO = f"/mnt/ceph/users/camels/PUBLIC_RELEASE/FOF_Subfind/"
# BASE_PATH_TO_SIMS = f"/home/musfer/work/Pylians/Sims/" # for test on local machine
# BASE_PATH_TO_HALO = f"/home/musfer/work/Pylians/FOF_Subfind/" # for test on local machine
# MIN_SNAP_ID = 0
# MAX_SNAP_ID = 33  # 33 for LH Set
#MIN_SIM_ID = 0
#MAX_SIM_ID = 999 # 999 for LH Set
snap_ids = [14, 18, 24, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56,
            58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
            90]


#RR = 2 # Mpc/h, for sigma8 calculations


def simsSnapName(simulation_id=0, snapshot_id=0):
    snapshot_id = ("0000" + str(snapshot_id))[-3:]
    return BASE_PATH_TO_SIMS + f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{simulation_id}/snap_{snapshot_id}.hdf5"


def haloSnapName(simulation_id=0, snapshot_id=0):
    snapshot_id = ("0000" + str(snapshot_id))[-3:]
    return BASE_PATH_TO_HALO + f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{simulation_id}/groups_{snapshot_id}.hdf5"


##### to plot delta filds
grid = GRID
MAS_dm     = 'CIC'  #mass-assigment scheme for DM
MAS_sub     = 'NGP'  #mass-assigment scheme for galaxies


verbose = True   #print information on progress

#####
axis = 2

if __name__ == "__main__":
	
	SUIT_NAME = sys.argv[1]
	

	       
    for sim_id in range(0,27):

        for snap_id in snap_ids:
            snap_name = simsSnapName(simulation_id=sim_id, snapshot_id=snap_id)

            with h5py.File(snap_name, 'r') as f:

            # Construct Dark Matter delta field
                BoxSize      = f['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
                redshift     = f['Header'].attrs[u'Redshift']
                h = f['Header'].attrs[u'HubbleParam'] #hubble parameter
                Omega_m  = f['Header'].attrs[u'Omega0']     #value of Omega_m
                Omega_l  = f['Header'].attrs['OmegaLambda']      #value of Omega_l


                Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)

                #                    print(f['Header'].attrs.keys())
                pos_dm = f['PartType1/Coordinates'][:]/1e3 #positions in Mpc/h
                pos_dm = pos_dm.astype('float32')
                pos_dm = np.array(pos_dm)
                Np = pos_dm.shape[0]



                vel_dm = f['PartType1/Velocities'] * np.sqrt(1 / (1 + redshift))
                vel_dm = vel_dm.astype('float32')
                vel_dm = np.array(vel_dm)

                mass_dm = np.ones(Np, dtype=np.float32)

                RSL.pos_redshift_space(pos_dm, vel_dm, BoxSize, Hubble, redshift, axis)

                # define 3D density field
                delta_dm = np.zeros((grid,grid,grid), dtype=np.float32)

                 # construct 3D density field
                MASL.MA(pos_dm, delta_dm, BoxSize, MAS_dm, W=mass_dm, verbose=verbose)

                delta_dm /= np.mean(delta_dm)
                delta_dm -= 1.

            subhalotab_name = haloSnapName(simulation_id=sim_id, snapshot_id=snap_id)

            with h5py.File(subhalotab_name, 'r') as f_sub:  

                BoxSize      = f_sub['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
                redshift     = f_sub['Header'].attrs[u'Redshift']
                SubHaloPos      = f_sub['Subhalo'][u'SubhaloCM'] [:] /1e3
                #                SubhaloMass      = f_sub['Subhalo'][u'SubhaloMass']

                mask = np.array(f_sub['Subhalo/SubhaloLenType'])[:,4] > 0
                #                print(f['Header'].attrs.keys())
                verbose = True   #print information on progress
                pos_subhalo = np.array(SubHaloPos)[mask]
                vel_subhalo = np.array(f_sub['Subhalo/SubhaloVel']  * (1 + redshift) )[mask]

                RSL.pos_redshift_space(pos_subhalo, vel_subhalo, BoxSize, Hubble, redshift, axis)


                Np_sub = pos_subhalo.shape[0]

                mass_subhalo = np.ones(Np_sub, dtype=np.float32)
                # define 3D density field
                delta_subhalo = np.zeros((grid,grid,grid), dtype=np.float32)

                # construct 3D density field
                MASL.MA(pos_subhalo, delta_subhalo, BoxSize, MAS_sub, W=mass_subhalo, verbose=verbose)

                delta_subhalo /= np.mean(delta_subhalo)
                delta_subhalo -= 1.


            # correlation function parameters
            threads = 32

            # compute cross-correlaton function of the two fields
            CCF_GG = PKL.XXi(delta_subhalo, delta_subhalo, BoxSize, [MAS_sub, MAS_sub], axis, threads)
            CCF_GM = PKL.XXi(delta_dm, delta_subhalo, BoxSize, [MAS_dm, MAS_sub], axis, threads)
            CCF_MM = PKL.XXi(delta_dm, delta_dm, BoxSize, [MAS_dm, MAS_dm], axis, threads)

            # get the attributes
            r      = CCF_GG.r3D      #radii in Mpc/h
            xxi0_GG   = CCF_GG.xi  
            xxi0_GM   = CCF_GM.xi  
            xxi0_MM   = CCF_MM.xi  

            ### Saving the output ###
            
            xi_res = [xxi0_GG, xxi0_GM, xxi0_MM ]  
            fn_res = Path(res_dir, f'corr_func_{SET_NAME}_{SUIT_NAME}_{i}.hdf5')
            
            try: 
                with h5py.File(fn_res, 'a') as hf:
                    dset_name_corr =  f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{sim_id}/snap_{snap_id}/corr_functions"
                    dset_name_rr = f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{sim_id}/snap_{snap_id}/rr"
                    hf.create_dataset(dset_name_corr, data = xi_res)
                    hf.create_dataset(dset_name_rr, data = r)

            except Exception as e:
                    print(e)
                    pass
            