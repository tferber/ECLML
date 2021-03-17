import torch
from torch_geometric.data import InMemoryDataset, Data
import h5py
import numpy as np
import os
import os.path as osp
import glob

class ECLDataset(InMemoryDataset):
    r"""Belle II ECL crystal level information dataset.
    .
    Args:
        root (string): Root directory where the input data is saved.
        name (string): The name of the dataset (one of :obj:`"TRAIN"`,
            :obj:`"ALL"`)
        transform (callable, optional): NOT IMPLEMENTED A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): NOT IMPLEMENTED A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        refresh (bool): Read raw input data again
        usetime (bool): Use timing information per crystal
        usepsd (bool): Use PSD information per crystal
        usemass (bool): Use crystal mass information per crystal
        maskfailedtimefits (bool): Only use crystals that have a valid time fit
    """
    names = ['train', 'all']

    def __init__(self, 
                 root,
                 name='train',
                 transform=None, 
                 pre_transform=None, 
                 refresh=False, 
                 usetime=False,
                 maskfailedtimefits=True,
                 usepsd=False, 
                 usemass=True):
        
        
        self.refresh = refresh
        self.usetime = usetime
        self.maskfailedtimefits = maskfailedtimefits
        self.usepsd = usepsd
        self.usemass = usemass
        self.refresh = refresh
        self.root = root
        
        self.name = name.lower()
        assert self.name in self.names
        
        if self.refresh:
            print('ECLDataset: refresh')
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
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # we can use glob.glob later
        
#         files = glob.glob(os.path.join(self.raw_dir, 'out_training-0.hdf5'))
        files = glob.glob(os.path.join(self.raw_dir, 'out_training-*.hdf5'))
        print('Read {} files.'.format(len(files)))
        return files

    @property
    def processed_file_names(self):
        if self.name == 'train':
            return ['data.pt']
        else:
            return ['data-all.pt']

    def process(self):
        
        list_data = []
        list_data_all = []
        
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
            
            input_timefittype = h5f['input_timefittype'][:]
            input_fittype = h5f['input_fittype'][:]

            if self.name != 'train' or self.usetime:
                inputs_time = h5f['input_time'][:]
                
            if self.name != 'train' or self.usepsd:
                inputs_psd = h5f['input_psd'][:]
                
            if self.name != 'train' or self.usemass:
                inputs_mass = h5f['input_mass'][:]
            
            if self.name != 'train':
                list_inputs_all = [inputs_theta_global, inputs_phi_local, inputs_energy, inputs_time, inputs_psd, inputs_mass]
                inputs_all = np.dstack(list_inputs_all)
                
            if self.usetime:
                print('ECLDataset: using crystal time as input feature in training')
                list_inputs.append(inputs_time)

            if self.usepsd:
                print('ECLDataset: using PSD as input feature in training')
                list_inputs.append(inputs_psd)

            if self.usemass:
                print('ECLDataset: using crystal weights as input feature in training')
                list_inputs.append(inputs_mass)

            inputs = np.dstack(list_inputs)
            coord_local = np.dstack(list_coord_local)
            
            if self.name != 'train':
                # baseline clustering info
                clstw0 = h5f['clstw0'][:]
                clstw1 = h5f['clstw1'][:]
                clstid0 = h5f['clstid0'][:]
                clstid1 = h5f['clstid1'][:]
                list_clst = [clstw0, clstw1, clstid0, clstid1]
                clst = np.dstack(list_clst)
                
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

            # training data list
            if self.name == 'train':
                for (n, x, y, eik, m, uid, cl, ff, ft) in zip(inputs_n, inputs, targets_t, targets_e, monitors, mon_uniqueid, coord_local, input_timefittype, input_fittype):

                    m = m.squeeze()
                
                    # make variable length node lists - this is only for the multi-dimensional inputs
                    x = x[:n[0],:] #inputs
                    cl = cl[:n[0],:] #local coordinates

                    y = y[:n[0],:] #targets                  
                    eik = eik[:n[0],:] #energies (for sqrt weights)
                    
                    ff = ff[:n[0]] #failed fits
                    ft = ft[:n[0]] #fit type
                    
                    # remove all digits that have failed time fits
                    if self.maskfailedtimefits:
                        mask_ff = (ff > 0)
                        x = x[mask_ff]    
                        cl = cl[mask_ff]    
                        y = y[mask_ff]    
                        eik = eik[mask_ff] 
                        
                    if x.shape[0] == 0:
                        print('no crystals left!')
                        continue

                    # sort targets by true phi
                    if m[3] > m[2] > 0:                    
                        y[:,[0, 1]] = y[:,[1, 0]]
                        eik[:,[0, 1]] = eik[:,[1, 0]]
                        m[[0, 1]] = m[[1, 0]]
                        m[[2, 3]] = m[[3, 2]]
                        m[[4, 5]] = m[[5, 4]]
                        m[[8, 9]] = m[[9, 8]]
                        m[[10, 11]] = m[[11, 10]]
                        m[[12, 13]] = m[[13, 12]]
                        m[[14, 15]] = m[[15, 14]]

                    data = Data(x=torch.from_numpy(x).to(torch.float), 
                                y=torch.from_numpy(y).to(torch.float))


                    # data can be augmented with anything by just adding it
                    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
                    data.uid = torch.tensor(uid, dtype=torch.int32)
                    data.monitor = torch.tensor(np.expand_dims(m, axis=0), dtype=torch.float)
                    data.eik = torch.from_numpy(eik).to(torch.float)
                    data.local = torch.from_numpy(cl).to(torch.float)

                    list_data.append(data)
            else:
                # all data list
                for (n, x, y, eik, m, uid, cl, inall, c, ff, ft) in zip(inputs_n, inputs, targets_t, targets_e, monitors, mon_uniqueid, coord_local, inputs_all, clst, input_timefittype, input_fittype):

                    m = m.squeeze()
                    
                    # make variable length node lists - this is only for the multi-dimensional inputs
                    x = x[:n[0],:] #inputs
                    cl = cl[:n[0],:] #local coordinates

                    y = y[:n[0],:] #targets                  
                    eik = eik[:n[0],:] #energies (for sqrt weights)
                    
                    time = inall[:n[0],[3]]
                    psd = inall[:n[0],[4]]
                    mass = inall[:n[0],[5]]
                    
                    ff = ff[:n[0]] #failed fits
                    ft = ft[:n[0]] #two component fit type

                    # sum the up to one cluster
                    cw0_ = np.nan_to_num(c[:n[0],[0]]).squeeze() #replace nans with zeros
                    cw1_ = np.nan_to_num(c[:n[0],[1]]).squeeze()
                    cid0_ = c[:n[0],[2]].squeeze()
                    cid1_ = c[:n[0],[3]].squeeze()
                    uid = np.unique((cid0_, cid1_)).astype(int) #list of unique cluster ids
                    uid = uid[uid>=0]
                    
                    if len(uid) < 2: # we have only one cluster, but two LM (i assume they fail energy and./or timing cuts)
                        continue
                        
                    sum0 = np.sum(np.where(cid0_==uid[0], x[:,2]*cw0_, 0)) + np.sum(np.where(cid1_==uid[0], x[:,2]*cw1_, 0))
                    sum1 = np.sum(np.where(cid0_==uid[1], x[:,2]*cw0_, 0)) + np.sum(np.where(cid1_==uid[1], x[:,2]*cw1_, 0))
                    
                    # which sum belongs to which true MC particle?
                    sum_c0_p0 = np.sum(np.where(cid0_==uid[0], eik[:,0]*cw0_, 0)) + np.sum(np.where(cid1_==uid[0], eik[:,0]*cw1_, 0))
                    sum_c0_p1 = np.sum(np.where(cid0_==uid[0], eik[:,1]*cw0_, 0)) + np.sum(np.where(cid1_==uid[0], eik[:,1]*cw1_, 0))
                    sum_c1_p0 = np.sum(np.where(cid0_==uid[1], eik[:,0]*cw0_, 0)) + np.sum(np.where(cid1_==uid[1], eik[:,0]*cw1_, 0))
                    sum_c1_p1 = np.sum(np.where(cid0_==uid[1], eik[:,1]*cw0_, 0)) + np.sum(np.where(cid1_==uid[1], eik[:,1]*cw1_, 0))
                    
                    # we need "baseline predicted" for each particle
                    eb0 = sum_c0_p0
                    eb1 = sum_c1_p1
                    if sum_c1_p0 > sum_c0_p0:
                        eb0 = sum_c1_p0
                    if sum_c0_p1 > sum_c1_p1:
                        eb1 = sum_c0_p1
                    eb = np.array([eb0, eb1])
                    
                    # sort targets by true phi
                    sw = np.array([0])
                    if m[3] > m[2] > 0:                    
                        y[:,[0, 1]] = y[:,[1, 0]]
                        eik[:,[0, 1]] = eik[:,[1, 0]]
                        eb[[0,1]] = eb[[1,0]]
                        sw[0]=1
                        m[[0, 1]] = m[[1, 0]]
                        m[[2, 3]] = m[[3, 2]]
                        m[[4, 5]] = m[[5, 4]]
                        m[[8, 9]] = m[[9, 8]]
                        m[[10, 11]] = m[[11, 10]]
                        m[[12, 13]] = m[[13, 12]]
                        m[[14, 15]] = m[[15, 14]]
                        
                    data = Data(x=torch.from_numpy(x).to(torch.float), 
                                y=torch.from_numpy(y).to(torch.float))

                    # data can be augmented with anything by just adding it
                    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
                    data.uid = torch.tensor(uid, dtype=torch.int32)
#                     print(eik.shape)
#                     print(m)
#                     print(np.expand_dims(m, axis=0).shape)
                    data.monitor = torch.tensor(np.expand_dims(m, axis=0), dtype=torch.float)
                    data.eik = torch.from_numpy(eik).to(torch.float)
                    data.local = torch.from_numpy(cl).to(torch.float)
                    data.time = torch.from_numpy(time).to(torch.float)
                    data.psd = torch.from_numpy(psd).to(torch.float)
                    data.mass = torch.from_numpy(mass).to(torch.float)
                    data.clstw0 = torch.from_numpy(cw0_).to(torch.float)
                    data.clstw1 = torch.from_numpy(cw1_).to(torch.float)
                    data.clstid0 = torch.from_numpy(cid0_).to(torch.float)
                    data.clstid1 = torch.from_numpy(cid1_).to(torch.float)
                    data.ff = torch.from_numpy(ff).to(torch.float)
                    data.ft = torch.from_numpy(ft).to(torch.float)
                    data.eb = torch.from_numpy(np.expand_dims(eb, axis=0)).to(torch.float)
                    data.swapped = torch.from_numpy(sw).to(torch.float)

                    list_data_all.append(data)
                
        if self.name == 'train':
            torch.save(self.collate(list_data), self.processed_paths[0])
        else:
            torch.save(self.collate(list_data_all), self.processed_paths[0])
            
            
        