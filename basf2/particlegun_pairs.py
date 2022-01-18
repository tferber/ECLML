#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################
#
# (c) Torben Ferber 2021
#
########################################################

# imports
from basf2 import *
import ROOT
from ROOT import Belle2
import os
import glob
import numpy as np
import math
import random

# Create main path
main = create_path()

# Add event info setter
main.add_module("EventInfoSetter", expList=int(1003), runList=int(0), evtNumList=100)

# Prepare particle gun
pdg_code_dict = {
    "pair_gammas": [22, 22],
    "pair_pions": [211, -211], #careful, the charged particle(s) will be curved in phi!
    "pair_piongamma": [211, 22] #careful, the charged particle(s) will be curved in phi!
    "pair_muongamma": [13, 22] #careful, the charged particle(s) will be curved in phi!
}

pdgcodes = pdg_code_dict[ "pair_gammas" ]
thetalow = 100. #degrees
thetahigh = 110. #degrees
philow = 0.0 #degrees
phihigh = 360.0 #degrees
elow = 0.5 #GeV
ehigh = 1.0 #GeV
maxopeningangle = 0.25 #radians

class CloseByParticleGenerator(Module):
    particle1_pdg_generator = lambda self: random.choice([int(pdgcodes[0])])
    particle2_pdg_generator = lambda self: random.choice([int(pdgcodes[1])])
    particle1_mom_generator = lambda self: random.uniform(float(elow), float(ehigh))
    particle2_mom_generator = lambda self: random.uniform(float(elow), float(ehigh))
    theta_generator = lambda self: math.radians(random.uniform(float(thetalow), float(thetahigh)))
    phi_generator = lambda self: math.radians(random.uniform(float(philow), float(phihigh)))
    opening_angle_generator = lambda self: random.uniform(0, 0.maxopeningangle)
    opening_rotation_generator = lambda self: random.uniform(0, 2) * math.pi

    def initialize(self):
        self.mcp = Belle2.PyStoreArray("MCParticles")
        self.mcp.registerInDataStore()

    def event(self):
        pdg1 = self.particle1_pdg_generator()
        pdg2 = self.particle2_pdg_generator()
        mom1 = self.particle1_mom_generator()
        mom2 = self.particle2_mom_generator()
        theta = self.theta_generator()
        phi = self.phi_generator()
        opening_angle = self.opening_angle_generator()
        opening_rotation = self.opening_rotation_generator()
        p1 = ROOT.TVector3()
        p1.SetMagThetaPhi(mom1, theta, phi)
        p2 = ROOT.TVector3()
        p2.SetMagThetaPhi(mom2, theta, phi)
        p2.Rotate(opening_angle, p1.Orthogonal())
        p2.Rotate(opening_rotation, p1)

        for pdg, p in [(pdg1, p1), (pdg2, p2)]:
            mcp = self.mcp.appendNew()
            mcp.setStatus(Belle2.MCParticle.c_PrimaryParticle | Belle2.MCParticle.c_StableInGenerator)
            mcp.setPDG(pdg)
            mcp.setMassFromPDG()
            m = mcp.getMass()
            mcp.setMomentum(p.x(), p.y(), p.z())
            mcp.setEnergy((p.Mag()**2 + m**2)**.5)
            mcp.setDecayTime(float("inf"))
            mcp.setProductionTime(0)
            mcp.setProductionVertex(0,0,0)

main.add_module(CloseByParticleGenerator())

# add simulation, reconstruction, and output....
