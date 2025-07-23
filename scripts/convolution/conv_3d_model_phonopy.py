
import phonopy

class model_info:
    def __init__(self):
        self.phonon = phonopy.load('phonopy.yaml')
        self.mesh = [11, 11, 11]
        self.phonon.run_mesh(self.mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        self.scattering_lengths = {'Ge': 8.185}
        self.temperature = 300
        self.cutoff = 8e-2

def init_model():
    global data
    data = model_info()

def model_disp_phonopy(vq1, vq2, vq3, data):
    Qpoints = np.column_stack([vq1,vq2,vq3])
    Q_prim = np.dot(Qpoints, data.phonon.primitive_matrix)
    data.phonon.run_qpoints(Q_prim, with_eigenvectors=False)
    band_dict = data.phonon.get_qpoints_dict()
    return np.transpose(band_dict['frequencies'])

def model_inten_phonopy(vq1, vq2, vq3, data):
    Qpoints = np.column_stack([vq1,vq2,vq3])
    Q_prim = np.dot(Qpoints, data.phonon.primitive_matrix)
    data.phonon.run_dynamic_structure_factor(Q_prim, data.temperature,
    scattering_lengths=data.scattering_lengths, freq_min=data.cutoff)
    dsf = data.phonon.dynamic_structure_factor
    return np.transpose(dsf.dynamic_structure_factors)
