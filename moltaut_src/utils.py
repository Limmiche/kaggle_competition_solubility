from openbabel.openbabel import OBMolAtomIter, OBMolBondIter

###################################################### edit
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
###################################################### edit

def filter_mol(pmol, element_list=[1,6,7,8,9,16,17]):
    obmol = pmol.OBMol
    if obmol is not None:
        elements = all([atom.GetAtomicNum() in element_list for atom in OBMolAtomIter(obmol)])
        if elements:
            return True
        else:
            return False

