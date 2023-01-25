import os
import sys
from openbabel import pybel

def pdbqt_to_sdf(pdbqt_file=None,output=None):

    results = [m for m in pybel.readfile(filename=pdbqt_file,format='pdbqt')]
    out=pybel.Outputfile(filename=output,format='sdf',overwrite=True)
    for pose in results:
        pose.data.update({'Pose':pose.data['MODEL']})
        del pose.data['MODEL'], pose.data['REMARK'], pose.data['TORSDO']

        out.write(pose)
    out.close()
    
folder = sys.argv[1]
for item in os.listdir(folder):
    full_name = folder + '/' + item
    if item.endswith('pdbqt'):
        pdbqt_to_sdf(full_name, full_name.replace('pdbqt', 'sdf')), 
    