# basf2 script to select and truth match ECL information
#
# (c) Torben Ferber, 2020 (torben.ferber@desy.de)
#
# Set the correct global_tag if you run on data! Requires cDST format data and MC!

import basf2
from modularAnalysis import inputMdstList, fillParticleList, variablesToNtuple
from variables import variables
from array import array
import ROOT
from ROOT import Belle2
from ROOT import gSystem
gSystem.Load('libecl.so')
import numpy as np
import sys, h5py, shutil, argparse, os

import networkx as nx
from networkx.algorithms.components.connected import connected_components

parser = argparse.ArgumentParser(description='basf2 script to extract ECL based ML information')
parser.add_argument('--minweightfrac', action="store", type=float, default=0.0) #weight/mcparticle energy
parser.add_argument('--minweightabs', action="store", type=float, default=0.001) #weight energy (in GeV)
parser.add_argument('--energyfraction', action="store", type=float, default=0.001) #weight/total deposited energy
parser.add_argument('--outfile', action="store", type=str, default='out.hdf5')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
args = parser.parse_args()

# --------------------------------
# create path
main_path = basf2.create_path()

# --------------------------------
# read input file
inputMdstList('default', '', path=main_path)

# --------------------------------
# 7x7 neighbour maps, cast to integers
nmap = np.genfromtxt('/nfs/dust/belle2/user/ferber/git/machinelearning/crystals/crystals_neighbours3.csv', dtype=np.int32, delimiter=',')
if nmap.shape[0] != 8736:
    print('selection.py, ERROR: ', nmap.shape[0])
    print('selection.py, ERROR: nmap has invalid number of crystals. Check if the textfile contains a header.')
    exit()
crystals = np.genfromtxt('/nfs/dust/belle2/user/ferber/git/machinelearning/crystals/crystals_coordinates_all.csv', delimiter=',')


# --------------------------------
# crystal masses
cmass = np.genfromtxt('/nfs/dust/belle2/user/ferber/git/machinelearning/crystals/crystal_masses.csv', dtype=np.float32, delimiter=',')

# --------------------------------
def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

# --------------------------------
def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current    

# --------------------------------
def zeropad(x, padlength=100, padfill = 0.0):
    return np.float32(np.pad(x, pad_width=(padlength - len(x)), mode='constant', constant_values=padfill))[padlength - len(x):]

# --------------------------------
def zeropadlist(x, padlength=100, padfill = [0.0, 0.0]):
    return x[:padlength] + [padfill]*(padlength - len(x))

# --------------------------------
def findConversion(particle):
    
    if particle.getSecondaryPhysicsProcess()==14:
        return particle.getMother().getArrayIndex()
    elif particle.getNDaughters()>0:
        for d in particle.getDaughters():
            return findConversion(d)
    else:
        return -1

# --------------------------------
def isConversionBeforeECL(vtx):
    if vtx.Z() < -102.0 or vtx.Z() > 196.0 or vtx.Perp()>118.0: #TOP start at 118cm)
        return False
    return True

# --------------------------------
def getAllDaughters(particle, dlist = []):
    for d in particle.getDaughters():
        dlist.append(d.getArrayIndex())
        getAllDaughters(d, dlist)
    return set(dlist) #making it a set is faster than looping over all particles again and again

# --------------------------------
def phiSubstract(angles, a):
    angles -= a
    return np.arctan2(np.sin(angles), np.cos(angles))    

# --------------------------------
def thetaSubstract(angles, a):
    return angles - a

# --------------------------------
list_m_uniqueid = []
list_m_exp = []
list_m_run = []
list_m_event = []
list_m_cluster_w = []
list_m_cluster_sum = []
list_m_e_sum = []

list_mc_energy = []
list_mc_theta = []
list_mc_phi = []
list_mc_ispairconversion = []
list_mc_pdg = []
list_mc_index = []
list_mc_angle = []

list_x_n = []
list_x_n_lm = []
list_x_cellid = []
list_x_energy = []
list_x_time = []
list_x_psd = []
list_x_theta = []
list_x_phi = []
list_x_theta_local = []
list_x_phi_local = []
list_x_timefittype = []
list_x_fittype = []
list_x_mass = []

list_y_class = []
list_y_e = []
list_y_frac = []

# --------------------------------
keepnmc = 3 #keep n highest true MC (via total sum per graph) per graph
keepnclst = 4 #keep n highest baseline clusters
keeplm = [1,2,3,4] # [1, 2] #keep LM overlaps with that many LMs

# --------------------------------
class getECLMLInfo(basf2.Module):
    """Module to get ECL ML information"""
    def __init__(self, minweightfrac, minweightabs, energyfraction, debug=False):
        super().__init__() # don't forget to call parent constructor
        
        self.debug = debug
        self.obj_eclgeometrypar = Belle2.ECL.ECLGeometryPar.Instance()
        self.minweightfrac = float(minweightfrac)
        self.minweightabs = float(minweightabs)
        self.energyfraction = float(energyfraction)
        
    def initialize(self):
        self.tracks = Belle2.PyStoreArray('Tracks')
        self.exthits = Belle2.PyStoreArray('ExtHits')
        self.mcparticles = Belle2.PyStoreArray('MCParticles')
        self.eclcaldigits = Belle2.PyStoreArray('ECLCalDigits')
        self.eclclusters = Belle2.PyStoreArray('ECLClusters')
        self.ecllms = Belle2.PyStoreArray('ECLLocalMaximums')
        self.eventinfo = Belle2.PyStoreObj('EventMetaData')
        self.cellidmap = Belle2.PyStoreObj('ECLCellIdMapping')
        
    def event(self):
        """Event loop"""

        # -----
        # set event variables
        m_exp = self.eventinfo.getExperiment()
        m_run = self.eventinfo.getRun()
        m_event = self.eventinfo.getEvent()
        
        cellIdToStoreArray = [None] * 8737
        for idx, eclcaldigit in enumerate(self.eclcaldigits):
            cellIdToStoreArray[eclcaldigit.getCellId()] = idx            
        
        # -----
        # loop over all primary MC particles and collect the daughters...
        edep_dict = {} # (caldigit id, particle id) -> energy
        etot_dict = {} # (particle id) -> energy
        true_energy_dict = {} # (particle id) -> true MC energy
        true_pdg_dict = {} # (particle id) -> true MC pdg
        
        pairconversion_dict = {}
        for idx, mcparticle in enumerate(self.mcparticles):
            
            if not mcparticle.isPrimaryParticle():
                continue
                
            mcparticles_temp = []
            mcparticles_daughters_temp = []

            isConversion = 0
            arrayindex = findConversion(mcparticle)        
            if(arrayindex>=0):
                convvtx = self.mcparticles[arrayindex].getDecayVertex()
                isConversion = isConversionBeforeECL(convvtx)
                
                if isConversion: #there is a pair conversion outside the ECL, keep the daughters instead. energy cuts in G4 can lead to "missing" electrons
                    
                    motherpdg = 22
                    motherarrayindex = arrayindex
                    for pp in self.mcparticles[arrayindex].getDaughters():
                        pairconversion_dict[pp.getArrayIndex()] = 1
                        mcparticles_temp.append(pp)
                        mcparticles_daughters_temp.append(getAllDaughters(pp, []))
                else: #nothing special, conversion happens inside ECL, just continue with the original particle
                    pairconversion_dict[arrayindex] = 0
                    mcparticles_temp.append(self.mcparticles[arrayindex])
                    mcparticles_daughters_temp.append(getAllDaughters(self.mcparticles[arrayindex], []))
                
            else: #no pair conversion at all
                pairconversion_dict[mcparticle.getArrayIndex()] = 0
                mcparticles_temp = [mcparticle]
                mcparticles_daughters_temp = [[mcparticle.getArrayIndex()]]
               
            for mcp, list_daughters in zip(mcparticles_temp, mcparticles_daughters_temp):
                    
                for ppidx in list_daughters:
                    mcrelations = self.mcparticles[ppidx].getRelationsWith('ECLCalDigits')
                    nmcrelations = mcrelations.size()
                    for mc_idx in range(nmcrelations):
                        w = mcrelations.weight(mc_idx)
                        cellid = mcrelations.object(mc_idx).getCellId() 
                        
                        edep_dict[(cellid, mcp.getArrayIndex())] = edep_dict.get((cellid, mcp.getArrayIndex()), 0) + w
                        etot_dict[mcp.getArrayIndex()] = etot_dict.get(mcp.getArrayIndex(), 0) + w
                        true_energy_dict.setdefault(mcp.getArrayIndex(), mcp.getEnergy())
                        true_pdg_dict.setdefault(mcp.getArrayIndex(), mcp.getPDG())
                        
        particle_keys = list(true_pdg_dict.keys())

        
        # -----
        # loop over all LMs and find potential overlaps
        l = []
        lid = []
        for idx, lm in enumerate(self.ecllms):
            lmid = lm.getCellId()-1
            nm = nmap[lmid] #crystal maps are 0..n
            l.append(set(nm[nm>=0]))
            lid.append(lmid+1) #actual cell ids are 1..n+1
        
        #get all (connected) subgraphs 
        G = to_graph(l)
        graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        lg = []
        for g in graphs:
            lg.append(list(g.nodes))
            
        # -----
        #graph id to LMId mapping (gid) -> [list of LMId]    
        gid_dict = {}
        for l in lid:
            a = [i for i, lst in enumerate(lg) if l in lst]
            if len(a) == 0:
                print(l)
                print(lg)
            gid_dict.setdefault(a[0], []).append(l)
            
        # -----
        # loop over all graphs
        for key in gid_dict:
            
            # check that the graph has one or two LMs
            if len(gid_dict[key]) in keeplm:
                
                lm_idx = [x - 1 for x in gid_dict[key]]
                lm_theta_global = (crystals[lm_idx, 3]) 
                lm_phi_global = (crystals[lm_idx, 4])
                
                # phi
                dphi = lm_phi_global[0]
                if len(gid_dict[key]) == 2:
                    dphi = (lm_phi_global[1]-lm_phi_global[0])
                shift_phi = np.pi if np.abs(dphi) > np.pi else 0.0
            
                tmp_phi = phiSubstract(lm_phi_global, shift_phi)
                mean_phi = np.mean(tmp_phi)
                shift_phi += mean_phi
                lm_phi_local = phiSubstract(tmp_phi, shift_phi)
                
                #theta
                shift_theta = np.mean(lm_theta_global)
                lm_theta_local = thetaSubstract(lm_theta_global, shift_theta)

                node_dict = {}
                cluster_dict = {}
                clustersum_dict = {}
                shower_dict = {}
                showersum_dict = {}
                for node in lg[key]:
                    aid = cellIdToStoreArray[node]
                    if aid is not None:
                        ene = [edep_dict.get((node, pidx)) for pidx in particle_keys]

                        for pidx, ee in zip(particle_keys, ene):
                            if ee is not None:
                                node_dict[pidx] = node_dict.get(pidx, 0) + ee    
                        
                        relcluster = self.eclcaldigits[aid].getRelationsWith('ECLClusters')
                        nrelcluster = relcluster.size()
                        for relidx in range(nrelcluster):
                            cluster = relcluster.object(relidx)
                            if not cluster.hasHypothesis(Belle2.ECLCluster.EHypothesisBit.c_nPhotons):
                                continue
                            w = relcluster.weight(relidx)
                            clustersum_dict[cluster.getArrayIndex()] = clustersum_dict.get(cluster.getArrayIndex(), 0) + w*self.eclcaldigits[aid].getEnergy()
                            cluster_dict[aid, cluster.getArrayIndex()] = w
                                                    
#                         relshower = self.eclcaldigits[aid].getRelationsWith('ECLShowers')
#                         nrelshower = relshower.size()
#                         for relidx in range(nrelshower):
#                             shower = relshower.object(relidx)
#                             if shower.getHypothesisId()!=5:
#                                 continue
#                             w = relshower.weight(relidx)
#                             showersum_dict[shower.getArrayIndex()] = showersum_dict.get(shower.getArrayIndex(), 0) + w*self.eclcaldigits[aid].getEnergy()
#                             shower_dict[aid, shower.getArrayIndex()] = w
                                                    
                        
                node_dict = dict(sorted(node_dict.items(), key=lambda item: item[1], reverse=True)) #sort, highest first
                node_keys = list(node_dict.keys())[:keepnmc] # keep up to two elements
                
                # we will only keep those clusters that deposit the most total energy
                clustersum_dict = dict(sorted(clustersum_dict.items(), key=lambda item: item[1], reverse=True)) #sort
#                 showersum_dict = dict(sorted(showersum_dict.items(), key=lambda item: item[1], reverse=True)) #sort
                cluster_keys = list(clustersum_dict.keys())[:keepnclst] # keep up to two elements      
#                 shower_keys = list(showersum_dict.keys())[:keepnclst] # keep up to two elements      
#                 print(clustersum_dict)    
#                 print(cluster_keys)
#                 print(showersum_dict)    
#                 print(shower_keys)
#                 print('---')
                    
                # store all information per node
                x_cellid = []
                x_energy = []
                x_time = []
                x_theta = []
                x_phi = []
                x_mass = []
                x_theta_local = []
                x_phi_local = []
                x_psd = []
                x_timefittype = []
                x_fittype = []
                m_cluster_w = []
                
                y_e = []
                y_eprime = []
                y_frac = []
                
                # per graph (we need to loop over all nodes for this one)
                m_e_sum = np.zeros(keepnmc)
                
                for node in lg[key]:
                    aid = cellIdToStoreArray[node]
                    if aid is not None:
                        caldigit = self.eclcaldigits[aid]
                        
                        ecldigit = eclcaldigit.getRelationsWith('ECLDigits').object(0)
                        ecldigit_timefittype = 1
                        if ecldigit.getTimeFit() == -2048: #needed in release-04, use status getters for release-05+
                            ecldigit_timefittype = 0
            
                        x_cellid.append(node)
                        x_theta.append(crystals[node-1, 3])
                        x_phi.append(crystals[node-1, 4])
                        x_theta_local.append(thetaSubstract(crystals[node-1, 3], shift_theta))
                        x_phi_local.append(phiSubstract(crystals[node-1, 4], shift_phi))
                        x_energy.append(caldigit.getEnergy())
                        x_time.append(caldigit.getTime())
                        x_psd.append(eclcaldigit.getTwoComponentHadronEnergy()/eclcaldigit.getTwoComponentTotalEnergy())
                        x_timefittype.append(ecldigit_timefittype)
                        x_fittype.append(caldigit.getTwoComponentFitType())
                        x_mass.append(cmass[node-1])
                               
                        # baseline clustering
                        wclst = [cluster_dict.get((aid, clstidx)) for clstidx in cluster_keys]
                        cluster_w = np.zeros(keepnclst)
                        for clstidx, _ in enumerate(wclst):
                            cluster_w[clstidx] = wclst[clstidx] if wclst[clstidx] is not None else 0.0
                        m_cluster_w.append(cluster_w)
                        
                        #target fractions
                        ene = [edep_dict.get((node, pidx)) for pidx in node_keys]
                        target_e = np.zeros(keepnmc+1)             
                        for eneidx, _ in enumerate(ene):
                            target_e[eneidx] = ene[eneidx] if ene[eneidx] is not None else 0.0
                            m_e_sum[eneidx] += target_e[eneidx]
                            
                        #FIXME: DOES NOT WORK FOR ANY NUMBER OF TARGETS YET
                        targetsum = np.sum(target_e[:keepnmc])
                        target_e[-1] = caldigit.getEnergy() - targetsum #add beam background (can be negative)
 
                        target_eprime = np.zeros(keepnmc+1)
                        target_eprime = np.where(target_e > 0., target_e, 0.)

                        x = np.sum(target_eprime, keepdims=1)
                        target_zeroes = np.zeros_like(target_eprime)
                        target_frac = np.divide(target_eprime, x, target_zeroes, where=x>0)
                        
                        y_frac.append(target_frac)
                        y_e.append(target_e)
                        y_eprime.append(target_eprime)
                                               
                
                # per graph
                list_x_n.append(len(x_cellid))
                list_x_n_lm.append(len(gid_dict[key]))
                
                # clusters
                m_cluster_sum = np.zeros(keepnclst)
                wclst = [clustersum_dict.get((clstidx)) for clstidx in cluster_keys]
                for clstidx, _ in enumerate(wclst):
                    m_cluster_sum[clstidx] = wclst[clstidx] if wclst[clstidx] is not None else 0.0

                # MC edep
                
                # MC truth: zeropad keepnmc
                mcparticles = [self.mcparticles[pidx] for pidx in node_keys]
                mce = zeropad([self.mcparticles[pidx].getEnergy() for pidx in node_keys], padlength=keepnmc)
                mctheta = zeropad([self.mcparticles[pidx].getMomentum().Theta() for pidx in node_keys], padlength=keepnmc)
                mcphi = zeropad([self.mcparticles[pidx].getMomentum().Phi() for pidx in node_keys], padlength=keepnmc)
                mcpdg = zeropad([self.mcparticles[pidx].getPDG() for pidx in node_keys], padlength=keepnmc)
                mcindex = zeropad([self.mcparticles[pidx].getArrayIndex() for pidx in node_keys], padlength=keepnmc)
                mconversion = zeropad([pairconversion_dict.get(self.mcparticles[pidx].getArrayIndex()) for pidx in node_keys], padlength=keepnmc)
                
                #FIXME
                mcangle = [-1.]
                if keepnmc == 2 and len(mcparticles) == 2:
                    mcangle = [(mcparticles[0].getMomentum()).Angle(mcparticles[1].getMomentum())]
                
                
                # zero padding is needed
                list_x_cellid.append(zeropad(x_cellid))
                list_x_energy.append(zeropad(x_energy))
                list_x_time.append(zeropad(x_time))
                list_x_psd.append(zeropad(x_psd))
                list_x_theta.append(zeropad(x_theta))
                list_x_phi.append(zeropad(x_phi))
                list_x_theta_local.append(zeropad(x_theta_local))
                list_x_phi_local.append(zeropad(x_phi_local))
                list_x_timefittype.append(zeropad(x_timefittype))
                list_x_fittype.append(zeropad(x_fittype))
                list_x_mass.append(zeropad(x_mass))

                list_y_e.append(np.asmatrix(zeropadlist(y_e, padfill = [0] * (keepnmc+1))))
                list_y_frac.append(np.asmatrix(zeropadlist(y_frac, padfill = [0] * (keepnmc+1))))
                
                list_m_uniqueid.append(m_run*1000000 + m_event)
                list_m_exp.append(m_exp)
                list_m_run.append(m_run)
                list_m_event.append(m_event)
                
                list_m_cluster_w.append(np.asmatrix(zeropadlist(m_cluster_w, padfill = [0] * keepnclst)))
                list_m_cluster_sum.append(m_cluster_sum)
                list_m_e_sum.append(m_e_sum)
                
                list_mc_energy.append(mce) #FIXME
                list_mc_theta.append(mctheta)
                list_mc_phi.append(mcphi)
                list_mc_ispairconversion.append(mconversion)
                list_mc_pdg.append(mcpdg)
                list_mc_index.append(mcindex)
                list_mc_angle.append(mcangle)

# --------------------------------
main_path.add_module('Gearbox')
main_path.add_module('Geometry')
main_path.add_module('ECLFillCellIdMapping')

# run python module
getECLMLInfo = getECLMLInfo(args.minweightfrac, args.minweightabs, args.energyfraction, debug=args.debug)
main_path.add_module(getECLMLInfo)

basf2.process(main_path)
print(basf2.statistics)

# --------------------------------
# write hdf5 file
hdf5file = 'tmp-xxx.hdf5'
with h5py.File(hdf5file, 'w', ) as f:

    # inputs (reconstructed)
    arr_x_n = np.asmatrix(list_x_n).T #need to transpose this one!
    arr_x_n_lm = np.asmatrix(list_x_n_lm).T #need to transpose this one!
    arr_x_cellid = np.asmatrix(list_x_cellid)
    arr_x_energy = np.asmatrix(list_x_energy)
    arr_x_time = np.asmatrix(list_x_time)
    arr_x_psd = np.asmatrix(list_x_psd)
    arr_x_theta = np.asmatrix(list_x_theta)
    arr_x_phi = np.asmatrix(list_x_phi)
    arr_x_theta_local = np.asmatrix(list_x_theta_local)
    arr_x_phi_local = np.asmatrix(list_x_phi_local)
    arr_x_timefittype = np.asmatrix(list_x_timefittype)
    arr_x_fittype = np.asmatrix(list_x_fittype)
    arr_x_mass = np.asmatrix(list_x_mass)

    # monitoring
    arr_m_exp = np.asmatrix(list_m_exp).T #need to transpose this one!
    arr_m_run = np.asmatrix(list_m_run).T #need to transpose this one!
    arr_m_evt = np.asmatrix(list_m_event).T #need to transpose this one!
    arr_m_uniqueid = np.asmatrix(list_m_uniqueid).T #need to transpose this one!

    # write to file
    f.create_dataset('n', data=arr_x_n)
    f.create_dataset('n_lm', data=arr_x_n_lm)
    f.create_dataset('x_cellid', data=arr_x_cellid, compression="gzip", compression_opts=9)
    f.create_dataset('x_energy', data=arr_x_energy, compression="gzip", compression_opts=9)
    f.create_dataset('x_time', data=arr_x_time, compression="gzip", compression_opts=9)
    f.create_dataset('x_psd', data=arr_x_psd, compression="gzip", compression_opts=9)
    f.create_dataset('x_theta', data=arr_x_theta, compression="gzip", compression_opts=9)
    f.create_dataset('x_phi', data=arr_x_phi, compression="gzip", compression_opts=9)
    f.create_dataset('x_theta_local', data=arr_x_theta_local, compression="gzip", compression_opts=9)
    f.create_dataset('x_phi_local', data=arr_x_phi_local, compression="gzip", compression_opts=9)
    f.create_dataset('x_timefittype', data=arr_x_timefittype, compression="gzip", compression_opts=9)
    f.create_dataset('x_fittype', data=arr_x_fittype, compression="gzip", compression_opts=9)
    f.create_dataset('x_mass', data=arr_x_mass, compression="gzip", compression_opts=9)

    f.create_dataset('y_e', data=list_y_e, compression="gzip", compression_opts=9)
    f.create_dataset('y_frac', data=list_y_frac, compression="gzip", compression_opts=9)

    f.create_dataset('m_cluster_w', data=list_m_cluster_w, compression="gzip", compression_opts=9)
    f.create_dataset('m_cluster_sum', data=list_m_cluster_sum, compression="gzip", compression_opts=9)
    f.create_dataset('m_e_sum', data=list_m_e_sum, compression="gzip", compression_opts=9)

    f.create_dataset('m_exp', data=arr_m_exp)
    f.create_dataset('m_run', data=arr_m_run)
    f.create_dataset('m_evt', data=arr_m_evt)
    f.create_dataset('m_uniqueid', data=arr_m_uniqueid)
    
    f.create_dataset('mc_energy', data=list_mc_energy)
    f.create_dataset('mc_theta', data=list_mc_theta)
    f.create_dataset('mc_phi', data=list_mc_phi)
    f.create_dataset('mc_pdg', data=list_mc_pdg)
    f.create_dataset('mc_index', data=list_mc_index)
    f.create_dataset('mc_ispairconversion', data=list_mc_ispairconversion)
    f.create_dataset('mc_angle', data=list_mc_angle)

    shutil.move(hdf5file, args.outfile)
    