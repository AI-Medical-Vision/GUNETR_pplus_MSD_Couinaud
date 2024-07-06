#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import unetr_pp
from unetr_pp.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from batchgenerators.utilities.file_and_folder_operations import *
from unetr_pp.experiment_planning.summarize_plans import summarize_plans
from unetr_pp.training.model_restore import recursive_find_python_class
import numpy as np
import pickle


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier

###### get_default_configuration(network, task, network_trainer, plans_identifier)
# network: 3d_fullres
# task: Task002_Synapse
# network_trained: unetr_pp_trainer_synapse
# plans_identifier: default_plans_identifier = "unetr_pp_Plansv2.1"
def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=(unetr_pp.__path__[0], "training", "network_training"),
                              base_module='unetr_pp.training.network_training'):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"
    
    #### preprocessing_output_dir
    # -> preprocessing_output_dir = os.environ['unetr_pp_preprocessed'] if "unetr_pp_preprocessed" in os.environ.keys() else None
    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")
        print(plans_file) # ../DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse\Task002_Synapse\unetr_pp_Plansv2.1_plans_3D.pkl

    plans = load_pickle(plans_file)
    # Maybe have two kinds of plans,choose the later one 
    # 9/7 Check: what is stage? and Check pkl file
    if len(plans['plans_per_stage']) == 2:
        Stage = 1
    else:
        Stage = 0
    print('Stage:', Stage) # 1
        
    if task == 'Task001_ACDC':
        plans['plans_per_stage'][Stage]['batch_size'] = 4
        plans['plans_per_stage'][Stage]['patch_size'] = np.array([16, 160, 160])
        pickle_file = open(plans_file, 'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()

    elif task == 'Task002_Synapse':
        plans['plans_per_stage'][Stage]['batch_size'] = 8 # 2
        plans['plans_per_stage'][Stage]['patch_size'] = np.array([64, 128, 128]) # 64x128X128
        #plans['plans_per_stage'][Stage]['patch_size'] = np.array([64, 64, 64])
        plans['plans_per_stage'][Stage]['pool_op_kernel_sizes'] = [[2, 2, 2], [2, 2, 2],
                                                                   [2, 2, 2]]  # for deep supervision
        
        plans['num_classes'] = 1 # our setting (1 or 8) // Couinaud segmentation 8 
        #print(plans)
        #print(plans['num_modalities']) # 1
        pickle_file = open(plans_file, 'wb')
        pickle.dump(plans, pickle_file)
        pickle_file.close()
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)
    
    # network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None
    network_training_output_dir = '../output_synapse'
    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)
    os.makedirs(output_folder_name, exist_ok=True)

    print("###############################################")
    print("I am running the following nnFormer: %s" % network)
    print("My trainer class is: ", trainer_class) # unetr_pp_trainer_synapse
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file) # unetr_plus_plus/unetr_pp/experiment_planning/summarize_plans.py
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")
    
    print(plans['data_identifier']) # unetr_pp_Data_plans_v2.1
    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
