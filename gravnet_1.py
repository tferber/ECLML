import argparse, time, random, os
from pathlib import Path
import h5py
import shutil

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
import numpy as np

from datasets.datasets import ECLDataset
from losses.losses import frac_loss
from models.models import GravNet
from metrics.metrics import clustermetric

# save models
directory_base = './saved_model/'
    
# ----------
def save_model(model, optimizer, dict_test, dict_train, epoch, lrate, directory, final=False):
    modelname = 'model-{}.pt'.format(str(epoch).zfill(5))
    dict_test_name = 'dict_test-{}.npy'.format(str(epoch).zfill(5))
    dict_train_name = 'dict_train-{}.npy'.format(str(epoch).zfill(5))
    
    if final:
        modelname = 'model-final.pt'
        dict_test_name = 'dict_test-final.npy'
        dict_train_name = 'dict_train-final.npy'
    
    torch.save({
        'epoch': epoch,
        'lr': lrate,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(directory, modelname))
        
    np.save(os.path.join(directory, dict_test_name), dict_test) 
    np.save(os.path.join(directory, dict_train_name), dict_train) 

# ----------
def load_model(model, optimizer, dict_test, dict_train, epoch, directory):
    
    # load model and dictionary
    checkpoint = torch.load(os.path.join(directory, 'model-final.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    dict_train = np.load(os.path.join(directory, 'dict_train-final.npy'), allow_pickle='TRUE').item()
    dict_test = np.load(os.path.join(directory, 'dict_test-final.npy'), allow_pickle='TRUE').item() 
    
    return model, optimizer, dict_test, dict_train, epoch
                            
# ----------
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# ----------
def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
    print("-------------------------------------------------------------------------")
    line_new = "{:>30}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("-------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("-------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    print("-------------------------------------------------------------------------")

# ----------
def train(device, model, optimizer, loader, epoch, lrate, dict_train, debug=False):
    model.train()
    
    t0 = time.time()
    loss_in_epoch = 0
    ngraphs_in_epoch = 0
    nbatches = len(loader)
    r_in_epoch = torch.tensor(()).to(device)
    a_in_epoch = torch.tensor(()).to(device)
   
    for i, batch in enumerate(loader):
                    
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)
        
        loss = frac_loss(batch, pred, usesqrt=True)
            
        loss_in_epoch += loss.item()
        ngraphs_in_epoch += batch.num_graphs
        
        loss.backward()

        optimizer.step()
        
        # metric calculation
        metric_r, metric_a1, metric_a2 = clustermetric(batch, pred, amin1=0.7, amax1=1.2, amin2=0.85, amax2=1.1)
        r_in_epoch = torch.cat((r_in_epoch, metric_r), 0)
        a1_in_epoch = torch.cat((a_in_epoch, metric_a1), 0)
        a2_in_epoch = torch.cat((a_in_epoch, metric_a2), 0)
        
        if debug:
            print('train: batch: {}/{} -> loss: {:10.8f} (avg. loss per graph: {:10.8f})'.format(str(i).zfill(5), 
                                                                                       nbatches, 
                                                                                       loss.item(),
                                                                                       loss_in_epoch/ngraphs_in_epoch), end="\r", flush=True)

    
    if debug:
        print()
        
    # get some values from metrics
    means = torch.mean(r_in_epoch, dim=0).detach().cpu().numpy()
    stds = torch.std(r_in_epoch, dim=0).detach().cpu().numpy()
    a1 = (torch.sum(a1_in_epoch, dim=0)/ngraphs_in_epoch).detach().cpu().numpy()
    a2 = (torch.sum(a2_in_epoch, dim=0)/ngraphs_in_epoch).detach().cpu().numpy()
    
    # add to dict for later plotting
    dict_train[epoch] = [loss_in_epoch/ngraphs_in_epoch, 
                         (time.time()-t0)/ngraphs_in_epoch, 
                         ngraphs_in_epoch, 
                         lrate, 
                         means, 
                         stds, 
                         a1,
                         a2]

# ----------
def test(device, model, loader, epoch, dict_test, debug=False):
    model.eval()
    
    t0 = time.time()
    loss_in_epoch = 0
    ngraphs_in_epoch = 0
    nbatches = len(loader)
    r_in_epoch = torch.tensor(()).to(device)
    a_in_epoch = torch.tensor(()).to(device)
   
    for i, batch in enumerate(loader):
    
        batch = batch.to(device)

        with torch.no_grad():

            pred = model(batch)

            loss = frac_loss(batch, pred)
            loss_in_epoch += loss.item()
            ngraphs_in_epoch += batch.num_graphs
            
            # metric calculation
            metric_r, metric_a1, metric_a2 = clustermetric(batch, pred, amin1=0.7, amax1=1.2, amin2=0.85, amax2=1.1)
            r_in_epoch = torch.cat((r_in_epoch, metric_r), 0)
            a1_in_epoch = torch.cat((a_in_epoch, metric_a1), 0)
            a2_in_epoch = torch.cat((a_in_epoch, metric_a2), 0)
                
            if debug:
                print('test: batch: {}/{}, loss:{:10.8f}'.format(str(i).zfill(5), nbatches, loss.item()), end="\r", flush=True)

    # get some values from metrics
    means = torch.mean(r_in_epoch, dim=0).detach().cpu().numpy()
    stds = torch.std(r_in_epoch, dim=0).detach().cpu().numpy()
    a1 = (torch.sum(a1_in_epoch, dim=0)/ngraphs_in_epoch).detach().cpu().numpy()
    a2 = (torch.sum(a2_in_epoch, dim=0)/ngraphs_in_epoch).detach().cpu().numpy()
        
    if debug:
        print()

    dict_test[epoch] = [loss_in_epoch/ngraphs_in_epoch, 
                        (time.time()-t0)/ngraphs_in_epoch, 
                        ngraphs_in_epoch,
                        -1,
                        means, 
                        stds, 
                        a1,
                        a2]

# ----------
def infer(device, model, loader, outfile):
    model.eval()

    list_phi = []
    list_theta = []
    list_energy = []
    list_t0 = []
    list_t1 = []
    list_tbkg = []
    list_p0 = []
    list_p1 = []
    list_pbkg = []
    list_monitor = []

    pad = 100

    for i, batch in enumerate(loader):
    
        batch = batch.to(device)

        with torch.no_grad():

            pred = model(batch)
            
            pred_batch = to_dense_batch(pred, batch.batch)[0].detach().cpu().numpy()
            x_batch = to_dense_batch(batch.x, batch.batch)[0].detach().cpu().numpy()
            y_batch = to_dense_batch(batch.y, batch.batch)[0].detach().cpu().numpy()
            monitor = batch.monitor.detach().cpu().numpy()
            local_batch = to_dense_batch(batch.local, batch.batch)[0].detach().cpu().numpy()
            
            for idx in range(pred_batch.shape[0]):
                p0 = pred_batch[idx,:,0]
                p1 = pred_batch[idx,:,1]
                pbkg = pred_batch[idx,:,2]
                
                t0 = y_batch[idx,:,0]
                t1 = y_batch[idx,:,1]
                tbkg = y_batch[idx,:,2]
                
                theta = local_batch[idx,:,0]
                phi = local_batch[idx,:,1]
                energy = x_batch[idx,:,2]
                
                list_theta.append(np.float32(np.pad(theta, pad_width=(pad - len(theta)), mode='constant', constant_values=0.))[pad - len(theta):])
                list_phi.append(np.float32(np.pad(phi, pad_width=(pad - len(phi)), mode='constant', constant_values=0.))[pad - len(phi):])
                list_energy.append(np.float32(np.pad(energy, pad_width=(pad - len(energy)), mode='constant', constant_values=0.))[pad - len(energy):])
                
                list_t0.append(np.float32(np.pad(t0, pad_width=(pad - len(t0)), mode='constant', constant_values=0.))[pad - len(t0):])                
                list_t1.append(np.float32(np.pad(t1, pad_width=(pad - len(t1)), mode='constant', constant_values=0.))[pad - len(t1):])
                list_tbkg.append(np.float32(np.pad(tbkg, pad_width=(pad - len(tbkg)), mode='constant', constant_values=0.))[pad - len(tbkg):])    
                
                list_p0.append(np.float32(np.pad(p0, pad_width=(pad - len(p0)), mode='constant', constant_values=0.))[pad - len(p0):])                
                list_p1.append(np.float32(np.pad(p1, pad_width=(pad - len(p1)), mode='constant', constant_values=0.))[pad - len(p1):])
                list_pbkg.append(np.float32(np.pad(pbkg, pad_width=(pad - len(pbkg)), mode='constant', constant_values=0.))[pad - len(pbkg):])
                
            list_monitor.append(np.float32(monitor))
            break
            
    
    hdf5file = 'tmp-xxx.hdf'
    with h5py.File(hdf5file, 'w') as f:

        arr_phi = np.asmatrix(list_phi)
        arr_theta = np.asmatrix(list_theta)
        arr_energy = np.asmatrix(list_energy)
        arr_t0 = np.asmatrix(list_t0)
        arr_t1 = np.asmatrix(list_t1)
        arr_tbkg = np.asmatrix(list_tbkg)
        arr_p0 = np.asmatrix(list_p0)
        arr_p1 = np.asmatrix(list_p1)
        arr_pbkg = np.asmatrix(list_pbkg)

        f.create_dataset('input_phi', data=arr_phi)
        f.create_dataset('input_theta', data=arr_theta)
        f.create_dataset('input_energy', data=arr_energy)
        f.create_dataset('target_t0', data=arr_t0)
        f.create_dataset('target_t1', data=arr_t1)
        f.create_dataset('target_tbkg', data=arr_tbkg)
        f.create_dataset('target_p0', data=arr_p0)
        f.create_dataset('target_p1', data=arr_p1)
        f.create_dataset('target_pbkg', data=arr_pbkg)

        arr_monitoring = np.asmatrix(np.concatenate(list_monitor, 0))
        f.create_dataset('mon_E0', data=arr_monitoring[:,0])
        f.create_dataset('mon_E1', data=arr_monitoring[:,1])
        f.create_dataset('mon_theta0', data=arr_monitoring[:,2])
        f.create_dataset('mon_theta1', data=arr_monitoring[:,3])
        f.create_dataset('mon_phi0', data=arr_monitoring[:,4])
        f.create_dataset('mon_phi1', data=arr_monitoring[:,5])
        f.create_dataset('mon_angle', data=arr_monitoring[:,6])
        f.create_dataset('mon_nshared', data=arr_monitoring[:,7])
        f.create_dataset('mon_n0', data=arr_monitoring[:,8])
        f.create_dataset('mon_n1', data=arr_monitoring[:,9])
        f.create_dataset('mon_e0_sel', data=arr_monitoring[:,10])
        f.create_dataset('mon_e1_sel', data=arr_monitoring[:,11])
        f.create_dataset('mon_e0_tot', data=arr_monitoring[:,12])
        f.create_dataset('mon_e1_tot', data=arr_monitoring[:,13])
        f.create_dataset('mon_e0_overlap', data=arr_monitoring[:,14])
        f.create_dataset('mon_e1_overlap', data=arr_monitoring[:,15])
        
    shutil.move(hdf5file, outfile)
    
# ----------
def main():        
    #---------------------
    # show pytorch version
    print('Using pytorch version: {}'.format(torch.__version__))
            
    #---------------------
    # user input via commandline
    parser = argparse.ArgumentParser(description='PyTorch ECL ML clustering')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs (default: 1)')
    parser.add_argument('--trainfrac', type=float, default=0.9, help='fraction of events used for training (default: 0.9)')
    parser.add_argument('--ncpu', type=int, default=1, help='how many CPUs are used in loaders (default: 1)')
    parser.add_argument('--seed', type=int, default=0, help='set random seed for all random generators')
    parser.add_argument('--modeldir', type=str, default=None, help='directory with pretrained model')
    parser.add_argument('--inferonly', dest='inferonly', default=False, action='store_true', help='only run inference, no train or test.')
    parser.add_argument('--nsave', type=int, default=5, help='save model and status every nsave epochs')
    parser.add_argument('--ninference', type=int, default=10, help='save inference ntuples for one batch every ninference epochs')
    parser.add_argument('--inferencedir', type=str, default='.', help='directory to store inference ntuples for one batch')
    parser.add_argument('--overtrain', dest='overtrain', default=False, action='store_true', help='only use one event to force overtraining')
    parser.add_argument('--nodecay', dest='nodecay', default=False, action='store_true', help='do not decay learning rate (LR)')
    parser.add_argument('--refresh', dest='refresh', default=False, action='store_true', help='do not load prepocessed datasets.')
    parser.add_argument('--debug-train', dest='debug_train', default=False, action='store_true', help='print loss for every training batch')
    parser.add_argument('--debug-test', dest='debug_test', default=False, action='store_true', help='print loss for every test batch')
    parser.add_argument('--no-test', dest='no_test', default=False, action='store_true', help='skip testing')
    parser.add_argument('--print-model', dest='print_model', default=False, action='store_true', help='print model')
    parser.add_argument('--use-cpu', dest='use_cpu', default=False, action='store_true', help='do not use GPU even if it is available')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--lrdecay', type=float, default=0.95, help='exponential decay of learning rate')
    args = parser.parse_args()

    # ----------------
    # set random seeds
    if args.seed > 0:
        set_seed(args.seed)

    #create savedir for model
    timestr = time.strftime("%Y%m%d-%H%M%S")
    randomstr = ''.join(["{}".format(random.randint(0, 9)) for num in range(0, 5)])
    targetdir = os.path.join(directory_base, timestr + '-'+ randomstr)
    Path(targetdir).mkdir(parents=True, exist_ok=True)

    # ----------------
    # get data
    indir = '/pnfs/desy.de/belle/local/user/ferber/mc/training-ML-BGx1/pair_gammas-0.1-withsec-7961763531/output_0054255137/'
    dataset = ECLDataset(root=indir, refresh=args.refresh, usetime=False, usemass=True, usepsd=False)
    dataset = dataset.shuffle()
    
    ntrain = int(args.trainfrac * len(dataset))
    ntest = len(dataset) - ntrain
    
    # split into test and train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [ntrain, ntest])
    print('#entries in train dataset: ', len(train_dataset))
    print('#entries in test dataset:  ', len(test_dataset))
    
    # force into overtraining by learning a single event
    if args.overtrain:
        nn = 10000 #range(0, args.batch_size)
        eventslist = list(range(0,nn))
        train_dataset = torch.utils.data.Subset(train_dataset, eventslist)
        test_dataset = torch.utils.data.Subset(test_dataset, eventslist)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.ncpu, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.ncpu, drop_last=True)
    
    # -----
    device = 'cpu'
    devicename = 'cpu'
    if not args.use_cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cuda'):
            devicename = torch.cuda.get_device_name(0)

    print('using device: {}, {}'.format(device, devicename))

    print('dataset.num_features: ', dataset.num_features)
    
    # --------------------- 
    # CREATE MODEL
    # ---------------------
    modelparams = {}
    model = GravNet(in_channels = dataset.num_features).to(device)
#     model = SimpleNet(in_channels = dataset.num_features).to(device)
    
    # just some print out of the model parameters
    if args.print_model:
        print(model)
        gnn_model_summary(model)

    # --------------------- 
    # CREATE OPTIMIZER
    # ---------------------
    print('learning rate: {}'.format(args.lr))
    print('learning rate decay: {}'.format(args.lrdecay))
    
    optparams = {'lr': args.lr}
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=optparams['lr'])

    # FIXME: Add second scheduler for stuck metric?
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lrdecay, verbose=False)
    
    # --------------------- 
    dict_train = {}
    dict_test = {}
    epoch_start = 0
    
    # --------------------- 
    # LOAD MODEL
    # ---------------------
    if args.modeldir is not None:
        print('loading previous model: {}'.format(args.modeldir))
        model, optimizer, dict_test, dict_train, epoch_start = load_model(model, optimizer, dict_test, dict_train, epoch_start, args.modeldir)
        
    # --------------------- 
    # CREATE DIRECTORY FOR INFERENCE
    # ---------------------
    if args.inferencedir is not None:
        
        # create path if it does not exist
        Path(args.inferencedir).mkdir(parents=True, exist_ok=True)

    # --------------------- 
    # TRAINING AND TESTING
    # ---------------------
    if not args.inferonly:
        lastepoch = args.epochs + 1 + epoch_start
        for epoch in range(1 + epoch_start, lastepoch):

            print('epoch:{}/{}'.format(str(epoch).zfill(5), str(lastepoch-1).zfill(5)))
            lrate = scheduler.get_last_lr()[0]

            # train
            train(device=device,
                  model=model,
                  optimizer=optimizer,
                  loader=train_loader,
                  epoch=epoch,
                  lrate=lrate,
                  dict_train=dict_train,
                  debug=args.debug_train)

            #decay learning rate
            if not args.nodecay:
                scheduler.step()

            # test
            if not args.no_test:
                test(device=device,
                      model=model,
                      loader=test_loader,
                      epoch=epoch,
                      dict_test=dict_test,
                      debug=args.debug_test)

            # FIXME: move to function!
            if not args.no_test:
                print('    train time: {:10.4f}s, test time: {:10.4f}s'.format(dict_train[epoch][1]*dict_train[epoch][2], dict_test[epoch][1]*dict_test[epoch][2]))
                print('    train loss (per graph): {:10.8f}, test loss (per graph): {:10.8f}'.format(dict_train[epoch][0], dict_test[epoch][0]))
                print('    train metrics: k=1   mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][0], dict_train[epoch][5][0], dict_train[epoch][6][0]))
                print('    train metrics: k=2   mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][1], dict_train[epoch][5][1], dict_train[epoch][6][1]))
                print('    train metrics: k=bkg mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][2], dict_train[epoch][5][2], dict_train[epoch][6][2]))
            else: #no testing requested
                print('    train time: {:10.4f}s'.format(dict_train[epoch][1]*dict_train[epoch][2]))
                print('    train loss (per graph): {:10.8f}'.format(dict_train[epoch][0]))
                print('    train metrics: k=1   mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][0], dict_train[epoch][5][0], dict_train[epoch][6][0]))
                print('    train metrics: k=2   mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][1], dict_train[epoch][5][1], dict_train[epoch][6][1]))
                print('    train metrics: k=bkg mean = {:10.8f}, sigma = {:10.8f}, a1 = {:10.8f}'.format(dict_train[epoch][4][2], dict_train[epoch][5][2], dict_train[epoch][6][2]))


            # save model
            if epoch%(int(args.nsave))==0:
                save_model(model, optimizer, dict_test, dict_train, epoch, lrate, targetdir)
                
            # store inference ntuples
            if epoch%(int(args.ninference))==0:
                # delete outfile if it exists        
                outfile = os.path.join(args.inferencedir, 'inference-{}.hdf5'.format(str(epoch).zfill(5)))
                try:
                    os.remove(outfile)
                except OSError:
                    pass
        
                infer(device=device,
                      model=model,
                      loader=train_loader,
                      outfile=outfile)


        # in any case, save the model at the end!
        save_model(model, optimizer, dict_test, dict_train, epoch, lrate, targetdir, final=True)
        
    else:
        # delete outfile if it exists        
        outfile = os.path.join(args.inferencedir, 'inference.hdf5')
        try:
            os.remove(outfile)
        except OSError:
            pass

        infer(device=device,
              model=model,
              loader=train_loader,
              outfile=outfile)

if __name__ == '__main__':
    main()
    