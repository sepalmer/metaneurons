import numpy as np
import analyses as als
import datasets as ds
import models as mds
import tools

# Load neural data
data_path = os.path.join(project_folder, 'data', 'nat_scenes_data')
D = ds.DataNatScenes(data_path, mov_ind=4)
D.trim(20)

# Compute receptive fields


# Save new dataset
D
