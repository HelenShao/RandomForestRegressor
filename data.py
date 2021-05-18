import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

# This function reads the rockstar file
def read_data(f_rockstar):
    # Getting the Halos
    PIDs = np.loadtxt(f_rockstar, usecols=41)      # Array of IDs (Halos have ID = -1)
    is_halo = np.array([x == -1 for x in PIDs])    # Identify halos from subhalos

    # Number of Particles Per Halo > 500 
    mass_per_particle = 6.56561e+11
    m_vir    = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo]
    n_particles = m_vir / mass_per_particle
    np_mask     = np.array([x>500 for x in n_particles])

    # Get the number of halos and properties
    n_halos = np.size(m_vir[np_mask])
    n_properties = 10

    #################################### LOAD DATA ###################################
    # Define container for data 
    data = np.zeros((n_halos, n_properties), dtype=np.float32)

    # v_rms
    data[:,0] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 4)[is_halo][np_mask]
    
    #v_max
    data[:,1] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 3)[is_halo][np_mask]

    #m_vir - forget this, same as r_vir
    #data[:,2] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 2)[is_halo][np_mask]
    
    # Ratio of kinetic to potential energies T/|U|
    data[:,2] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 37)[is_halo][np_mask]

    # r_vir
    data[:,3] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 5)[is_halo][np_mask]

    # r_s
    data[:,4] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 6)[is_halo][np_mask]

    # Velocities 
    v_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 11)[is_halo][np_mask]
    v_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 12)[is_halo][np_mask]
    v_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 13)[is_halo][np_mask]
    v_mag    = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))
    data[:,5] = v_mag

    # Angular momenta 
    J_x      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 14)[is_halo][np_mask]
    J_y      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 15)[is_halo][np_mask]
    J_z      = np.loadtxt(f_rockstar, skiprows = 16, usecols = 16)[is_halo][np_mask]
    J_mag    = np.sqrt((J_x**2) + (J_y**2) + (J_z**2))
    data[:,6] = J_mag

    # Spin
    data[:,7] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 17)[is_halo][np_mask]

    # b_to_a
    data[:,8] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 27)[is_halo][np_mask]

    # c_to_a
    data[:,9] = np.loadtxt(f_rockstar, skiprows = 16, usecols = 28)[is_halo][np_mask]

    ############################# NORMALIZE DATA ##############################
    # This function normalizes the input data
    def normalize_data(data):
        # Data_shape: (n_samples, n_features)
        n_halos = data.shape[0]       # n_samples
        n_properties = data.shape[1]  # n_features
        
        # Create container for normalized data
        data_norm = np.zeros((n_halos, n_properties), dtype=np.float32)
        
        for i in range(n_properties):
            mean = np.mean(data[:,i])
            std  = np.std(data[:,i])
            normalized = (data[:,i] - mean)/std
            data_norm[:,i] = normalized
        return(data_norm)

    # Take log10 of J_mag (m_vir removed)
    #data[:,2]  = np.log10(data[:,2]+1)
    data[:,6]  = np.log10(data[:,6]+1)

    # Normalize each property
    halo_data = normalize_data(data)

    # Convert to torch tensor
    halo_data = torch.tensor(halo_data, dtype=torch.float)
    
    return halo_data

###################################### Create Datasets ###################################
class make_Dataset(Dataset):
    
    def __init__(self, name, seed, f_rockstar):
        
        # Get the data
        halo_data = read_data(f_rockstar)
        n_halos = halo_data.shape[0] 
        
        # shuffle the halo number (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        # n_halos = 3674
        # halo_data shape = (n_halo, number of properties) = (3674, 10)
        
        np.random.seed(seed)
        indexes = np.arange(n_halos)
        np.random.shuffle(indexes)
        
        # Divide the dataset into train, valid, and test sets
        if   name=='train':  size, offset = int(n_halos*0.8), int(n_halos*0.0)
        elif name=='valid':  size, offset = int(n_halos*0.1), int(n_halos*0.8)
        elif name=='test' :  size, offset = int(n_halos*0.1), int(n_halos*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, 9), dtype=torch.float) # Each input has a shape of (9,) (flattened)
        self.output = torch.zeros((size, 1), dtype=torch.float)  # Each output has shape of (1,) 
        
        # do a loop over all elements in the dataset
        for i in range(size):
            j = indexes[i+offset]                 # find the halo index (shuffled)
            self.input[i] = halo_data[:,1:10][j]  # load input
            self.output[i] = halo_data[:,0][j]    # Load output (v_rms)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    
#This function creates datasets for train, valid, test
def create_datasets(seed, f_rockstar):
    
    train_Dataset = make_Dataset('train', seed, f_rockstar)
    valid_Dataset = make_Dataset('valid', seed, f_rockstar)
    test_Dataset  = make_Dataset('test',  seed, f_rockstar)
    
    return train_Dataset, valid_Dataset, test_Dataset
