import steps.utilities.meshio as smeshio
import steps.utilities.meshctrl as meshctrl
import steps.solver as solvmod
import steps.model as smodel
import steps.geom as sgeom
import steps.rng as srng

import numpy as np
import mmap
import copy
import re

import misc

def model(tissue):
    """ initialises model using tissue array """

    mdl = smodel.Model()
    unique_p = []
    for cell in tissue:
        [unique_p.append(p) for p in cell.prtcl_names if p not in unique_p]
    
    NP = []
    NPi = []
    # Create particles and corresponding species
    for p in unique_p:
        # Free NPs
        NP.append(smodel.Spec('N{}'.format(p), mdl))
        # internalised NPs
        NPi.append(smodel.Spec('N{}i'.format(p), mdl))
        
    # receptor state: 'naive' state (no bound NPs)
    R = smodel.Spec('R', mdl)
    # complexes state: NPs bound to a cell receptor
    NPR = smodel.Spec('NPR', mdl)
    d = {}
    rxn_ = {}

    # Create volume for diffusion of NPs
    vsys1 = smodel.Volsys('tissue', mdl)
    vsys2 = smodel.Volsys('vessel', mdl)

    dfsn_ = {} 
    # Loop where cell and particle properties are connected to reactions   
    for n,cell in enumerate(tissue):    
        for p_idx, p in enumerate(unique_p):
            tag = str(n) + p
            prtcl = getattr(cell, p)
            d["memb{}".format(tag)] = smodel.Surfsys("memb{}".format(tag), mdl)
            k_diff = prtcl['D']

             # binding reactions:
            if 'k_a' in prtcl:
                k_bind = prtcl['k_a']
                k_unbind = prtcl['k_d']
                k_intern = prtcl['k_i']
                rxn_["bind_{}".format(tag)] = smodel.SReac("bind_{}".format(tag), d["memb{}".format(tag)], 
                    olhs=[NP[p_idx]], slhs =[R], srhs=[NPR], kcst=k_bind)
                rxn_["unbind_{}".format(tag)] = smodel.SReac("unbind_{}".format(tag), d["memb{}".format(tag)], 
                    slhs=[NPR], orhs=[NP[p_idx], R], kcst=k_unbind)
                rxn_["intern_{}".format(tag)] = smodel.SReac("intern_{}".format(tag), d["memb{}".format(tag)], 
                    slhs=[NPR], irhs=[NPi[p_idx], R], kcst=k_intern)

            # Diffusion 
            dfsn1 = smodel.Diff('dfsn1', vsys1, NP[p_idx], dcst = k_diff)    
            dfsn2 = smodel.Diff('dfsn2', vsys2, NP[p_idx], dcst = k_diff)    

    return mdl


def geom(tissue, sim_opt):
    unique_p = []
    for cell in tissue:
        [unique_p.append(p) for p in cell.prtcl_names if p not in unique_p]

    print('parameters: mesh_name = %s' %  sim_opt['opt'].mesh)
    cell = tissue[0]
    meshf = 'meshes/{0}_{1}.inp'.format(sim_opt['opt'].scenario, sim_opt['opt'].mesh)
    #scale is 1um (domain size is 40um)

    mesh, nodeproxy, tetproxy, triproxy  = smeshio.importAbaqus(meshf, 1e-6,verbose=False)
    if sim_opt['opt'].savemesh == 1:
        smeshio.saveMesh(sim_opt['opt'].mesh, mesh)

    tet_groups = tetproxy.blocksToGroups()
    f = open(meshf)
    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    vol_tag = []
    type_tags = []
    while s.find(b'Volume') != -1:
        m = re.search(b'Volume', s)
        if len(vol_tag) < 30:
            volume = s[m.start(): m.end() + 3]
        else:
            volume = s[m.start(): m.end() + 4]
        vol_tag.append(''.join([n for n in str(volume) if n.isdigit()]))
        s = s[m.end():-1]

    domain_tets = tet_groups["Volume186"]# % 186 is specified in the .geo file
    tissue = sgeom.TmComp('tissue', mesh, domain_tets)
    tissue.addVolsys('tissue')
    vol_tag.remove('186')
    tissue_vessel_db = []
    type_tags = [t.split("_") for t in type_tags]
    print('Successfully found %i cells' %  len(vol_tag))
    msh_ = {}
    memb_ = {}
    cell_ = {}
    for n,vtag in enumerate(vol_tag): 
        for p_idx, p in enumerate(unique_p):
            tag = str(n) + p
            msh_['c_tets{}'.format(tag)] = tet_groups["Volume%s" % vtag]
            msh_['memb_tets{0}'.format(tag)] = meshctrl.findOverlapTris(mesh, domain_tets, msh_['c_tets{}'.format(tag)])
            cell_['cell{}'.format(tag)] = sgeom.TmComp('cell{}'.format(tag), mesh, msh_['c_tets{}'.format(tag)])
            cell_['memb{}'.format(tag)] = sgeom.TmPatch('memb{}'.format(tag), mesh, msh_['memb_tets{}'.format(tag)], \
                icomp = cell_['cell{}'.format(tag)], ocomp = tissue)
            cell_['memb{}'.format(tag)].addSurfsys('memb{}'.format(tag))
    geom = [mesh, domain_tets, type_tags]
    return geom

def boundary_tets(geom):

    ntets = geom[0].countTets()
    boundary_tets = []
    boundary_tris = set()

    z_max = geom[0].getBoundMax()[2]
    eps = 1e-6
    for t in range(ntets):
        # Fetch the z coordinate of the barycenter
        barycz = geom[0].getTetBarycenter(t)[2]
        # Fetch the triangle indices of the tetrahedron, a tuple of length 4:
        tris = geom[0].getTetTriNeighb(t)
        if barycz > z_max - eps :
            boundary_tets.append(t)
            boundary_tris.add(tris[0])
            boundary_tris.add(tris[1])
            boundary_tris.add(tris[2])
            boundary_tris.add(tris[3])
    return boundary_tets

def solver(mdl, geom, tissue, sim_opt):

    treated_tissue = copy.deepcopy(tissue) 
    reactants = ['NP','NPi', 'NPR', 'R']
    NT = int(sim_opt['general'].NT)
    t_final = int(sim_opt['general'].t_final)
    NP0 = float(tissue[0].P0['NP0'])
    NR = float(tissue[0].NR)

    N_cells = int(sim_opt['general'].N_cells)
    mesh = geom[0]
    domain_tets = geom[1]
    type_tags = geom[2]

    # Create rnadom number generator object
    rng = srng.create('mt19937', 512)
    rng.initialize(np.random.randint(1, 10000))

    # Create solver object
    sim = solvmod.Tetexact(mdl, mesh, rng)

    sim.reset()
    tpnt = np.linspace(0.0, t_final, NT)
    ntpnts = tpnt.shape[0]

    # Create the simulation data structures
    ntets = mesh.countTets()
    resi = np.zeros((ntpnts,1,  len(reactants)))

    c_type = ''
    if sim_opt['opt'].unif == '1':
        sim.setCompCount('tissue', 'NP0', NP0)
    else:
        tetX = boundary_tets(geom)

        for n in tetX:
            sim.setTetCount(n, 'NP0', NP0/len(tetX))    
    
    sim.setPatchCount('memb0P0', 'R', NR)

    if sim_opt['opt'].visualise == '1':
        raise Exception("Plotting option is selected but not plotting scripts included")
        
    else:
        misc.printProgressBar(0, NT, prefix = 'Progress:', suffix = 'Complete', length = 40)                
        for t in range(NT):
            sim.run(tpnt[t])
            misc.printProgressBar(t, NT, prefix = 'Progress:', suffix = 'Complete', length = 40)
            n = 0
            for nc,cell in enumerate(treated_tissue):
                for p_idx, p in enumerate(cell.prtcl_names):
                    tag = str(nc) + p
                    prtcl = getattr(cell, p)
                    if hasattr(cell,'NR'):
                        resi[t, n, 0] = sim.getCompCount('tissue', 'N{}'.format(p))   
                        resi[t, n, 1] = sim.getPatchCount("memb{}".format(tag), 'R')                                    
                        resi[t, n, 2] = sim.getPatchCount("memb{}".format(tag), 'NPR')                                    
                        resi[t, n, 3] = sim.getCompCount("cell{}".format(tag), 'N{}i'.format(p))                                    
                    if 'T' in prtcl:
                        if (tpnt[t] > prtcl['T']) or (prtcl['T'] == 1):
                            sim.setCompClamped("tissue", 'N{}'.format(p), False)
                    if ('NP_max' in prtcl) and (resi[t, n, 3] > prtcl['NP_max']):
                        cell.type = 'dead'
                    n += 1
        return treated_tissue, resi   
