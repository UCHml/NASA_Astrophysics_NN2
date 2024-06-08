import sys
import numpy as np
import h5py
import hdf5plugin
import os
from pathlib import Path

import MAS_library as MASL
#import matplotlib.pyplot as plt
import Pk_library as PKL


# In[17]:
threads = 8

GRID = 200


#SUIT_NAME = "IllustrisTNG" 
SET_NAME = "CV"


res_dir = 'output' # directory of the results; Change it!!!

BASE_PATH_TO_SIMS = f"/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/" # change pathname
BASE_PATH_TO_HALO = f"/mnt/ceph/users/camels/PUBLIC_RELEASE/FOF_Subfind/"
# BASE_PATH_TO_SIMS = f"/home/musfer/work/Pylians/Sims/" # for test on local machine
# BASE_PATH_TO_HALO = f"/home/musfer/work/Pylians/FOF_Subfind/" # for test on local machine
MIN_SNAP_ID = 0
MAX_SNAP_ID = 33  # 33 for LH Set
#MIN_SIM_ID = 0
#MAX_SIM_ID = 999 # 999 for LH Set


RR = 2 # Mpc/h, for sigma8 calculations


def simsSnapName(simulation_id=0, snapshot_id=0):
    snapshot_id = ("0000" + str(snapshot_id))[-3:]
    return BASE_PATH_TO_SIMS + f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{simulation_id}/snap_{snapshot_id}.hdf5"


def haloSnapName(simulation_id=0, snapshot_id=0):
    snapshot_id = ("0000" + str(snapshot_id))[-3:]
    return BASE_PATH_TO_HALO + f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{simulation_id}/fof_subhalo_tab_{snapshot_id}.hdf5"


# In[18]:


##### to plot delta filds
grid = GRID
MAS_dm     = 'CIC'  #mass-assigment scheme for DM
MAS_sub     = 'NGP'  #mass-assigment scheme for galaxies


verbose = True   #print information on progress

#####


if __name__ == "__main__":

    i, SUIT_NAME = sys.argv[1], sys.argv[2]

    i = int(i)

    for sim_id in range(0, 27):
        for snap_id in range(MIN_SNAP_ID, MAX_SNAP_ID+1):
                    
            # Construct Dark Matter delta field
            snap_name = simsSnapName(simulation_id=sim_id, snapshot_id=snap_id)

            with h5py.File(snap_name, 'r') as f:

                BoxSize      = f['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
                redshift     = f['Header'].attrs[u'Redshift']
                #                    print(f['Header'].attrs.keys())
                pos_dm = f['PartType1/Coordinates'][:]/1e3 #positions in Mpc/h
                pos_dm = pos_dm.astype('float32')
                pos_dm = np.array(pos_dm)
                Np = pos_dm.shape[0]
                mass_dm = np.ones(Np, dtype=np.float32)

                # define 3D density field
                delta_dm = np.zeros((grid,grid,grid), dtype=np.float32)

                # construct 3D density field
                MASL.MA(pos_dm, delta_dm, BoxSize, MAS_dm, W=mass_dm, verbose=verbose)

                delta_dm /= np.mean(delta_dm)
                delta_dm -= 1.0          

            # Construct SubHalo delta field

            subhalotab_name = haloSnapName(simulation_id=sim_id, snapshot_id=snap_id)

            with h5py.File(subhalotab_name, 'r') as f_sub:  

                BoxSize      = f_sub['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
                redshift     = f_sub['Header'].attrs[u'Redshift']
                SubHaloPos      = f_sub['Subhalo'][u'SubhaloCM'] [:] /1e3
                #                SubhaloMass      = f_sub['Subhalo'][u'SubhaloMass']

                mask = np.array(f_sub['Subhalo/SubhaloLenType'])[:,4] > 0
                #                print(f['Header'].attrs.keys())
                verbose = True   #print information on progress
                poss_subhalo = np.array(SubHaloPos)[mask]
                Np_sub = poss_subhalo.shape[0]

                mass_subhalo = np.ones(Np_sub, dtype=np.float32)
                # define 3D density field
                delta_subhalo = np.zeros((grid,grid,grid), dtype=np.float32)

                # construct 3D density field
                MASL.MA(poss_subhalo, delta_subhalo, BoxSize, MAS_sub, W=mass_subhalo, verbose=verbose)

                delta_subhalo /= np.mean(delta_subhalo)
                delta_subhalo -= 1.



        # correlation function parameters
            axis    = 0


            # compute cross-correlaton function of the two fields
            CCF_GG = PKL.XXi(delta_subhalo, delta_subhalo, BoxSize, [MAS_sub, MAS_sub], axis, threads)
            CCF_GM = PKL.XXi(delta_dm, delta_subhalo, BoxSize, [MAS_dm, MAS_sub], axis, threads)
            CCF_MM = PKL.XXi(delta_dm, delta_dm, BoxSize, [MAS_dm, MAS_dm], axis, threads)

            # get the attributes
            r      = CCF_GG.r3D      #radii in Mpc/h
            xxi0_GG   = CCF_GG.xi[:,0]  #monopole
            xxi0_GM   = CCF_GM.xi[:,0]  #monopole
            xxi0_MM   = CCF_MM.xi[:,0]  #monopole

        ### Saving the output ###

            xi_res = [r, xxi0_GG, xxi0_GM, xxi0_MM ]  
            fn_res = Path(res_dir, f'corr_func_{SET_NAME}_{SUIT_NAME}.hdf5')
            
            try: 
                with h5py.File(fn_res, 'a') as hf:
                    dset_name_corr =  f"{SUIT_NAME}/{SET_NAME}/{SET_NAME}_{sim_id}/snap_{snap_id}/corr_functions"

                    hf.create_dataset(dset_name_corr, data = xi_res)
            except Exception:
                    pass
