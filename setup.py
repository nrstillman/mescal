import xml.etree.ElementTree as ET
import numpy as np
import copy

class Cell(object):
    """ Class for individual cell data - see config for options"""

    def __init__(self, dictionary):
        prtcl_names = []
        for key in dictionary:   
            if key[0] == 'P':
                self.key = dictionary[key]
                prtcl_names.append(key) 
            setattr(self, key, dictionary[key])
        self.prtcl_names = prtcl_names
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<cell: %s>" % self.__dict__

class Options(object):
    """ Class for simulation options such as simulated time, parallelisation """
    
    def __init__(self, dictionary):
        for key in dictionary:    
            setattr(self, key, dictionary[key])
    #----------------------------------------------------------------------
    def __repr__(self):
        return "<cell: %s>" % self.__dict__


def get_sim_data(path):
    """ Simulation options from config and cast into Options object"""
    tree = ET.parse(path)
    root = tree.getroot()
    
    #find all cell data in config file
    sim_category = root.findall( './/category' )

    #initiate dictionary of cell data
    sim_data = {}
    opt = []

    for c in sim_category:
        d = {}    
        for child in c:
            d[child.attrib.get("name")] = child.text 
        sim_data[c.attrib.get("name")] = Options(d)

    return sim_data

def update_config(update_data, path):
    """ Update config file from input dict"""
    tree = ET.parse(path)
    root = tree.getroot()

    for opt in list(update_data):
        data = update_data[opt]
        if opt == 'cell':

            #find all cell data in config file
            cell_types = root.findall( './/cell' )

            #initiate dictionary of cell data
            cell_data = {}

            # loop through input data and update config file
            for d,c in zip(data,cell_types):
                if d != c[0].text:
                    assert('Error: data is not ordered as config file')
                for tag in data[d].keys():
                    if type(data[d][tag]) == dict:                    
                        for p in data[d][tag].keys():
                            elem = tree.findall('.//cell[@name="{}"]/particle[@name="{}"]/data[@name="{}"]'.format(d, tag,p))
                            if len(elem) > 1:
                                print('Error: attempted format over range - check input')
                                break   
                            elem[0].text = str(data[d][tag][p])
                    else:
                        elem = tree.findall('.//cell[@name="{}"]/data[@name="{}"]'.format(d,tag))
                        elem[0].text = str(data[d][tag])
        else:
            #find all cell data in config file
            cat_types = root.findall( './/{}'.format(opt) )
            #initiate dictionary of cell data
            cat_data = {}
            # loop through input data and update config file
            for d,c in zip(data,cat_types):
                if d != c[0].text:
                    assert('Error: data is not ordered as config file')               
                for tag in data[d].keys():
                    elem = tree.findall('.//category[@name="{}"]/data[@name="{}"]'.format(d, tag))
                    if len(elem) > 1:
                        print('Error: attempted format over range - check input')
                        break   
                    elem[0].text = str(data[d][tag])
        tree.write(path)
    return 0

def update_N(data, path):
    """ Update config file from input dict"""
    tree = ET.parse(path)
    root = tree.getroot()

    #find all cell data in config file
    cell_types = root.findall( './/cell' )

    #initiate dictionary of cell data
    cell_data = {}

    # loop through input data and update config file
    for d,c in zip(data,cell_types):
        if d != c[0].text:
            print('Error: data is not ordered as config file')
            break        
        for tag in data[d].keys():
            for p in data[d][tag].keys():
                elem = tree.findall('.//cell[@name="{}"]/particle[@name="{}"]/data[@name="{}"]'.format(d, tag,p))
                if len(elem) > 1:
                    print('Error: attempted format over range - check input')
                    break   
            elem[0].text = str(data[d][tag][p])
    tree.write(path)
    return 0

def calc_fitness(tissue, treated_tissue):
    """ calculate number of killed cells"""

    #currently this is simple measure of killed/not-killed. 
    #tissue input can be used to calculate what cell types were killed
    # (or structure can be changed to status) - similarly can include eg # of NPs internalised
    killed = []
    targets = []
    
    for c in treated_tissue:
        if c.type == 'dead':
            killed.append(c) 
    return len(killed)/(len(treated_tissue) -1)

def get_cell_data(path, *argv):
    """ Cell options from config and cast into Cell object"""

    tree = ET.parse(path)
    root = tree.getroot()

    #find all cell data in config file
    cell_types = root.findall( './/cell' )

    #initiate array of cell data
    cells = []

    for c in cell_types: 
        #initiate dictionary for cell
        cell_data = {}
        cell_data['type'] = c[0].text
        cell_data['N_cell'] = int(c[1].text)
        cell_data['N_prtcl'] = int(c[2].text)
        for child in c[3:]:
            if child.tag != 'particle':
                cell_data[child.attrib.get("name")] = child.text
            else:
                npi = {}
                for p in range(len(child)):                    
                    # if no range is given
                    if len(child[p]) is 0:
                        npi[child[p].attrib.get("name")] = float(child[p].text)
                    else:
                        # check if range is sampled
                        if int(child[p][0].text) == 1:
                            low = float(child[p][1].text)
                            high = float(child[p][2].text)
                            np_range = []
                            [np_range.append(np.random.uniform(low, high)) for i in range(int(child[p][3].text))]
                        # check if individual settings are included
                        else:
                            # catch accidental sample options
                            if child[p][-1].attrib.get("name") == 'N_cell':
                                print('error: incorrect sample option')
                                return
                            np_range = []
                            [np_range.append(float(c.text)) for c in child[p][1:]]
                        npi[child[p].attrib.get("name")] = np_range
                cell_data[child.attrib.get("name")] = npi
        cells.append(cell_data)
    return cells
        
def make_cells(cell_data):
    """ Creates list of cell objects from data """
    cells = []

    for cell_type in cell_data:
        for n in range(cell_type['N_cell']):
            new_cell = copy.deepcopy(cell_type) 
            for d in cell_type:
                if d[0] == 'P':
                    for  key, value in cell_type[d].items():
                        if isinstance(value, list):
                            new_cell[d][key] = value[n]
            cells.append(Cell(new_cell))    
    return cells

def make_tissue(tissue, cells, tissue_map):
    """  Makes tissue array using cell objects and tissue map (for mapping cell types to int) """
    tissue = [tissue_map[c] for c in tissue]
    # loop matching cell data to tissue posn (TODO: bettter way to do this)    
    for n, cell_type in enumerate(tissue):
        i = 0
        while len(cells) > 0 :
            if cell_type == cells[i].type:
                tissue[n] = cells[i]
                cells.remove(cells[i])
                i = 0
                break
            else:
                i+=1
    return tissue