import os
from notebooks.file_reader import read_file
from config.path_config import PathConfig

fuel_df = read_file(os.path.join(PathConfig.BASE_DIR,
                                 PathConfig.DATA, PathConfig.FUEL_DATA_PATH))
print(os.path.join(PathConfig.BASE_DIR, PathConfig.DATA, PathConfig.FUEL_DATA_PATH))