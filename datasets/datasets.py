import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
import numpy as np
import os
import os.path as osp
import glob

class ECLDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 transform=None, 
                 pre_transform=None, 
                 refresh=False, 
                 usetime=False, 
                 usepsd=False, 
                 usemass=True):
        
        
        self.refresh = refresh
        self.usetime = usetime
        self.usepsd = usepsd
        self.usemass = usemass
        self.refresh = refresh
        self.root = root
        
        if self.refresh:
            print('ECLDataset: using PSD as input feature')
            pdir = self.processed_dir
            files = self.processed_file_names
            
            for f in files:
                try:
                    os.remove(osp.join(pdir, f))
                except OSError:
                    pass
             
            try:
                os.remove(osp.join(pdir, 'pre_filter.pt'))
            except OSError:
                pass
            
            try:
                os.remove(osp.join(pdir, 'pre_transform.pt'))
            except OSError:
                pass
     
            
        super(ECLDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_dir(self):
        return osp.join(self.root, '')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # we can use glob.glob later
        
        files = glob.glob(os.path.join(self.raw_dir, 'out_training-*.hdf5'))
        print('Read {} files.'.format(len(files)))
        return files

    @property
    def processed_file_names(self):
        return ['data.pt']    

    def process(self):
        list_data = []
        for filename in self.raw_file_names:

            filename = osp.join(self.raw_dir, filename)
            print(f'open {filename}')
            h5f = h5py.File(filename, 'r')

            # inputs (y)
            inputs_n = h5f['n'][:].T
            inputs_theta_local = h5f['input_theta'][:]
            inputs_phi_local = h5f['input_phi'][:]
            inputs_energy = h5f['input_energy'][:]
            inputs_theta_global = h5f['input_theta_global'][:]
            list_inputs = [inputs_theta_global, inputs_phi_local, inputs_energy]
            list_coord_local = [inputs_theta_local, inputs_phi_local]
            
            if self.usetime:
                print('ECLDataset: using crystal time as input feature')
                inputs_time = h5f['input_time'][:]
                list_inputs.append(inputs_time)

            if self.usepsd:
                print('ECLDataset: using PSD as input feature')
                inputs_psd = h5f['input_psd'][:]
                list_inputs.append(inputs_psd)

            if self.usemass:
                print('ECLDataset: using crystal weights as input feature')
                inputs_mass = h5f['input_mass'][:]
                list_inputs.append(inputs_mass)

            inputs = np.dstack(list_inputs)
            coord_local = np.dstack(list_coord_local)
            
            # energies
            targets_e0 = h5f['target_e0'][:] #true energy of particle 1
            targets_e1 = h5f['target_e1'][:]  #true energy of particle 2
            targets_ebkg = inputs_energy-(targets_e0+targets_e1)
            targets_ebkg_prime = np.where(targets_ebkg > 0, targets_ebkg, 0)
            targets_e =  np.dstack((targets_e0, targets_e1, targets_ebkg))
            targets_e_prime =  np.dstack((targets_e0, targets_e1, targets_ebkg_prime))
                        
            # t_ik
            x = np.sum(targets_e_prime, axis=2, keepdims=1)
            targets_zeroes = np.zeros_like(targets_e_prime)
            targets_t = np.divide(targets_e_prime, x, targets_zeroes, where=x>0)


            #monitoring (m)
            mon_uniqueid = h5f['mon_uniqueid'][:]
            mon_E0 = h5f['mon_E0'][:]
            mon_E1 = h5f['mon_E1'][:]
            mon_theta0 = h5f['mon_theta0'][:]
            mon_theta1 = h5f['mon_theta1'][:]
            mon_phi0 = h5f['mon_phi0'][:]
            mon_phi1 = h5f['mon_phi1'][:]
            mon_angle = h5f['mon_angle'][:]
            mon_nshared = h5f['mon_nshared'][:]
            mon_n0 = h5f['mon_n0'][:]
            mon_n1 = h5f['mon_n1'][:]
            mon_e0_sel = h5f['mon_e0_sel'][:]
            mon_e1_sel = h5f['mon_e1_sel'][:]
            mon_e0_tot = h5f['mon_e0_tot'][:]
            mon_e1_tot = h5f['mon_e1_tot'][:]
            mon_e0_overlap = h5f['mon_e0_overlap'][:]
            mon_e1_overlap = h5f['mon_e1_overlap'][:]
            
            monitors = np.dstack((mon_E0, mon_E1, mon_theta0, mon_theta1, mon_phi0, mon_phi1, mon_angle, mon_nshared, mon_n0, mon_n1, mon_e0_sel, mon_e1_sel, mon_e0_tot, mon_e1_tot, mon_e0_overlap, mon_e1_overlap))

            for (n, x, y, eik, m, uid, cl) in zip(inputs_n, inputs, targets_t, targets_e, monitors, mon_uniqueid, coord_local):
                # make variable length node lists - this is only for the multi-dimensional inputs
                x = x[:n[0],:] #inputs
                y = y[:n[0],:] #targets
                eik = eik[:n[0],:] #energies (for sqrt weights)
                cl = cl[:n[0],:] #local coordinates
                                
                data = Data(x=torch.from_numpy(x).to(torch.float), 
                            y=torch.from_numpy(y).to(torch.float))

                # data can be augmented with anything by just adding it
                # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
                data.uid = torch.tensor(uid, dtype=torch.int32)
                data.monitor = torch.tensor(m, dtype=torch.float)
                data.eik = torch.from_numpy(eik).to(torch.float)
                data.local = torch.from_numpy(cl).to(torch.float)

                # append this new data object to the list
                list_data.append(data)
                
                
        data, slices = self.collate(list_data)
        torch.save((data, slices), self.processed_paths[0])
        