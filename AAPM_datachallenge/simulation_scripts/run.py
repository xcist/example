import skimage.transform
import json
from scipy.io import loadmat, savemat
import numpy as np
import gecatsim as xc
from gecatsim.reconstruction.pyfiles import recon

def dumpjson(filename, jsondict):
    out_file = open(filename, "w") 
    json.dump(jsondict, out_file, indent = 4) 
    out_file.close()

# load metal masks, of size 256x256
metal_file = loadmat("./metal_masks.mat")
metal_file = metal_file['tumor_imgs']/255
metal_file[metal_file>=0.1] = 1
metal_file[metal_file<0.1] = 0

metal = np.float32(metal_file[0])
metaldiam = 2.*np.sqrt(metal.sum()/np.pi) # effective diameter
with open("metal.vf", 'wb') as fout:
    tmp1 = np.float32(metal)
    tmp1 = tmp1.copy(order="C")
    fout.write(tmp1)  # Write the phantom layer to the matching file name

# pix size for phantoms (excluding metal)
ph_pixsize = 400./512 #in mm/pixel

# following parameters are random/customizable in real simulations
tgt_mtdiam = 10 # effective diameter in mm
x_offset = 100 # offset (position) of metal, in pixels
y_offset = 100
metal_mat = 'Ti' # material of metal
# end of random parameters

# metal pixel size, in mm/pixel
mt_pixsize = tgt_mtdiam/metaldiam

phmetal_json = json.load(open("sim_base.json"))
phmetal_json['n_materials'] = 3
phmetal_json['mat_name'].append("water")
phmetal_json['volumefractionmap_datatype'].append('float')
phmetal_json["volumefractionmap_filename"].append("phantom.vf")
phmetal_json['cols'].append(int(512))
phmetal_json['rows'].append(int(512))
phmetal_json['slices'].append(int(1))
phmetal_json['x_size'].append(ph_pixsize)
phmetal_json['y_size'].append(ph_pixsize)
phmetal_json['z_size'].append(100.0)
phmetal_json['x_offset'].append(256.5)
phmetal_json['y_offset'].append(256.5)
phmetal_json['z_offset'].append(1.0)
phmetal_json['density_scale'].append(1.0)
phmetal_json['mat_name'].append("water")
phmetal_json['mat_name'].append(metal_mat)
phmetal_json['volumefractionmap_datatype'].append('float')
phmetal_json['volumefractionmap_datatype'].append('float')
phmetal_json["volumefractionmap_filename"].append("metal.vf")
phmetal_json["volumefractionmap_filename"].append("metal.vf")
phmetal_json['cols'].append(int(256))
phmetal_json['cols'].append(int(256))
phmetal_json['rows'].append(int(256))
phmetal_json['rows'].append(int(256))
phmetal_json['slices'].append(int(1))
phmetal_json['slices'].append(int(1))
phmetal_json['x_size'].append(mt_pixsize)
phmetal_json['x_size'].append(mt_pixsize)
phmetal_json['y_size'].append(mt_pixsize)
phmetal_json['y_size'].append(mt_pixsize)
phmetal_json['z_size'].append(100.0)
phmetal_json['z_size'].append(100.0)
phmetal_json['x_offset'].append(x_offset*ph_pixsize/mt_pixsize)
phmetal_json['x_offset'].append(x_offset*ph_pixsize/mt_pixsize)
phmetal_json['y_offset'].append(y_offset*ph_pixsize/mt_pixsize)
phmetal_json['y_offset'].append(y_offset*ph_pixsize/mt_pixsize)
phmetal_json['z_offset'].append(1.0)
phmetal_json['z_offset'].append(1.0)
phmetal_json['density_scale'].append(-1.0)
phmetal_json['density_scale'].append(1.0)

dumpjson("sim.json", phmetal_json) 

# simulations
ct = xc.CatSim("AAPM_MAR_phantom",
               "AAPM_MAR_protocol",
               "AAPM_MAR_physics",
               "AAPM_MAR_scanner",
               "AAPM_MAR_recon")

ct.resultsName = "out"

ct.run_all()  # run the scans defined by protocol.scanTypes

# reconstruction
cfg = ct.get_current_cfg();
cfg.do_Recon = 1
cfg.waitForKeypress = 0
recon.recon(cfg)
