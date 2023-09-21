# args[1] - number of nodes ('512','1024','2048')
# args[2] - network index
# args[3] - random seed
import sys, os
sys.path.insert(0, 'PySources/')
sys.path.insert(0, 'NetworkStates/')
args = sys.argv

import numpy as np
import numpy.linalg as la
import numpy.random as rand
import pickle

import network_generation as ngen
import network_util as nutil
import network_plot as nplot

# make network objects. net3 is the one usable for cpp lin-solver
NNodes = int(args[1])
XFlag = False
if args[1] == '64':
    NS = '00064'
    XFlag = True
if args[1] == '128':
    NS = '00128'
    XFlag = True
if args[1] == '256':
    NS = '00256'
    XFlag = True
if args[1] == '512':
    NS = '00512'
    XFlag = True
if args[1] == '1024':
    NS = '01024'
    XFlag = True
if args[1] == '2048':
    NS = '02048'
    XFlag = True
if args[1] == '4096':
    NS = '04096'
    XFlag = True
assert(XFlag)
NNet = int(args[2])



if args[1] in ['512','1024','2048','4096']:
    fn = 'NetworkStates/statedb_N' + NS + '_Lp-4.0000'
    net = ngen.convert_jammed_state_to_network(fn,NNet) # choose number of nodes
    LL = net['box_mat'].diagonal()
    net['node_pos'] = net['node_pos']/LL[0] - 0.5
    net2 = ngen.convert_to_network_object(net)
    #net3 = ngen.convertToFlowNetwork(net2)
else:
    if args[1] == '128':
        NNet = NNet + 50
    if args[1] == '256':
        NNet = NNet + 100
    fn = 'NetworkStates/data_N' + NS + '_Lp0.0100_r' + str(NNet) + '.txt'
    f = open(fn)
    Data = f.readlines()
    Data = [d.rsplit() for d in Data]
    f.close()
    NN = int(Data[0][0])
    NE = int(Data[0][1])
    L = float(Data[0][2])
    pos = Data[1:NN+1]
    pos = np.asarray(pos, float).flatten()
    edges = np.asarray(Data[NN+1:],int)
    EI = edges.T[0]
    EJ = edges.T[1]
    
    net = {}
    net['DIM'] = 2
    net['box_L'] = np.array([L, L])
    net['NN'] = NN
    net['node_pos'] = pos
    net['NE'] = NE
    net['edgei'] = EI
    net['edgej'] = EJ
    net['box_mat'] = np.array([[L, 0.], [0., L]])
    LL = net['box_mat'].diagonal()
    net['node_pos'] = net['node_pos']/LL[0] - 0.5
    net2 = ngen.convert_to_network_object(net)
    #net3 = ngen.convertToFlowNetwork(net2)


import numba
from numpy import zeros, ones, diag, array, where, dot, c_, r_, arange, sum

#setup useful network objects
NRandSeed = int(args[3])
rand.seed(NRandSeed)
DIM = 1
NN = net2.NN
NE = net2.NE
EI = array(net2.edgei)
EJ = array(net2.edgej)

DM = zeros([NE, NN])
for i in range(NE):
    DM[i,EI[i]] = +1.
    DM[i,EJ[i]] = -1.

from scipy.linalg import solve

# setup matrices

D1 = array([where(EI==i,1,0) for i in range(NN)])
D2 = array([where(EJ==j,1,0) for j in range(NN)])
DD = D1 + D2

ids = c_[EI,EJ]
UT = zeros([NN,NN])

# flow solver

# Maked Constraint matrices
@numba.jit()
def ConstraintMatrix(NodeData, Nodes, EdgeData, Edges, GroundNodes, NN, EI, EJ):
    csg = len(GroundNodes)
    idg = arange(csg)
    CStr = zeros([csg, NN+1])
    CStr[idg, GroundNodes] = +1.
    CStr[:, NN] = 0.
    
    if len(Nodes):
        csn = len(Nodes)
        idn = arange(csn)
        SN = zeros([csn, NN+1])
        SN[idn, Nodes] = +1.
        SN[:, NN] = NodeData
        CStr = r_[CStr, SN]
    
    if len(Edges):
        cse = len(Edges)
        ide = arange(cse)
        SE = zeros([cse, NN+1])
        SE[ide, EI[Edges]] = +1.
        SE[ide, EJ[Edges]] = -1.
        SE[:, NN] = EdgeData
        CStr = r_[CStr, SE]
    return CStr
