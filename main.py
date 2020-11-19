import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import train
import test
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms
from loan_helper import LoanHelper
from image_helper import ImageHelper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")

vis = visdom.Visdom()
criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
def trigger_test_byindex(helper, index, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_index_" + str(index) + "_trigger")
def trigger_test_byname(helper, agent_name_key, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_" + str(agent_name_key) + "_trigger")
def vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):
        _name= names[i]
        _weight=weights[i]
        _is_poison=False
        if _name in adversarial_name_keys:
            _is_poison=True
        helper.target_model.weight_vis(vis=vis,epoch=epoch,weight=_weight, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)

def vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):
        _name= names[i]
        _alpha=alphas[i]
        _is_poison=False
        if _name in adversarial_name_keys:
            _is_poison=True
        helper.target_model.alpha_vis(vis=vis,epoch=epoch,alpha=_alpha, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)

if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    else:
        helper = None

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')

    similarity_other_path = helper.folder_path + "/model_similarity.txt"
    similarity_other_file = open(similarity_other_path, 'w')
    similarity_mean_path = helper.folder_path + "/model_other_similarity.txt"
    similarity_mean_file = open(similarity_mean_path, 'w')
    write_header = False

    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')

    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])

                nonattacker=[]
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist+nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary']==False:
                adversarial_name_keys=copy.deepcopy(helper.params['adversary_list'])
        logger.info('----------------------------------Start one training epoch------------------------------------------')
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}. Adversary List: {adversarial_name_keys}')
        epochs_submit_update_dict, num_samples_dict = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        
        logger.info(f'time spent on training: {time.time() - t}')
        logger.info('----------------------------------End one training epoch------------------------------------------')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        

        ''''''
        
        replace_eps = 1e-6
        begnign_name_keys = list(set(agent_name_keys) - set(adversarial_name_keys))
        adversarial_similarity_dict = {}
        begnign_similarity_dict = {}
        begnign_base_dict = {}
        adversarial_mean_dict = {}
        begnign_mean_dict = {}

        #for begnign_key in begnign_name_keys:
        for begnign_key in agent_name_keys:
            for parameter_name in updates[begnign_key][1]:
                update_array = updates[begnign_key][1][parameter_name].cpu().numpy().copy()
                if parameter_name not in begnign_base_dict:
                    begnign_base_dict[parameter_name] = update_array
                else:
                    begnign_base_dict[parameter_name] = begnign_base_dict[parameter_name] + update_array

                
        for parameter_name in begnign_base_dict:
            begnign_base_dict[parameter_name] = begnign_base_dict[parameter_name] / len(agent_name_keys)
            valid_index = np.where(begnign_base_dict[parameter_name] == 0)
            begnign_base_dict[parameter_name][valid_index] = replace_eps


        # For the adversarial
        for adversarial_key in adversarial_name_keys:
            adversarial_similarity_dict[adversarial_key] = {}
            adversarial_weight = updates[adversarial_key][1]

            for begnign_key in agent_name_keys:
                if adversarial_key != begnign_key:
                    begnign_weight = updates[begnign_key][1]

                    for parameter_name in adversarial_weight:
                        '''
                        valid_index = np.where(begnign_weight[parameter_name].numpy() != 0)
                        adversarial_weight_array = adversarial_weight[parameter_name].numpy()[valid_index]
                        begnign_weight_array = begnign_weight[parameter_name].numpy()[valid_index]
                        division = (adversarial_weight_array - begnign_weight_array) / begnign_weight_array
                        '''

                        adversarial_weight_array = adversarial_weight[parameter_name].cpu().numpy().copy()
                        begnign_weight_array = begnign_weight[parameter_name].cpu().numpy().copy()
                        replace_index = np.where(begnign_weight_array == 0)

                        begnign_weight_array[replace_index] = replace_eps
                        adversarial_weight_array[replace_index] = replace_eps
                        division = (adversarial_weight_array - begnign_weight_array) / begnign_weight_array

                        similarity_result = np.mean(np.abs(division))

                        if parameter_name not in adversarial_similarity_dict[adversarial_key]:
                            adversarial_similarity_dict[adversarial_key][parameter_name] = similarity_result
                        else:
                            adversarial_similarity_dict[adversarial_key][parameter_name] += similarity_result

        for adversarial_key in adversarial_name_keys:
            for parameter_name in adversarial_similarity_dict[adversarial_key]:
                adversarial_similarity_dict[adversarial_key][parameter_name] = round(adversarial_similarity_dict[adversarial_key][parameter_name] / (len(agent_name_keys) - 1), 2)

        
        for begnign_key_1 in begnign_name_keys:
            begnign_similarity_dict[begnign_key_1] = {}
            begnign_weight_1 = updates[begnign_key_1][1]

            for begnign_key_2 in agent_name_keys:
                
                if begnign_key_1 != begnign_key_2:
                    begnign_weight_2 = updates[begnign_key_2][1]

                    for parameter_name in begnign_weight_1:
                        '''
                        valid_index = np.where(begnign_weight_2[parameter_name].numpy != 0)
                        begnign_weight_array_1 = begnign_weight_1[parameter_name].numpy()[valid_index]
                        begnign_weight_array_2 = begnign_weight_2[parameter_name].numpy()[valid_index]
                        division = (begnign_weight_array_1 - begnign_weight_array_2) / begnign_weight_array_2
                        '''
                        begnign_weight_array1 = begnign_weight_1[parameter_name].cpu().numpy().copy()
                        begnign_weight_array2 = begnign_weight_2[parameter_name].cpu().numpy().copy()
                        replace_index = np.where(begnign_weight_array2 == 0)

                        begnign_weight_array1[replace_index] = replace_eps
                        begnign_weight_array2[replace_index] = replace_eps
                        division = (begnign_weight_array1 - begnign_weight_array2) / begnign_weight_array2

                        similarity_result = np.mean(np.abs(division))

                        if parameter_name not in begnign_similarity_dict[begnign_key_1]:
                            begnign_similarity_dict[begnign_key_1][parameter_name] = similarity_result
                        else:
                            begnign_similarity_dict[begnign_key_1][parameter_name] += similarity_result

        for begnign_key in begnign_name_keys:
            for parameter_name in begnign_similarity_dict[begnign_key]:
                begnign_similarity_dict[begnign_key][parameter_name] = round(begnign_similarity_dict[begnign_key][parameter_name] / (len(agent_name_keys)-1), 2)



        for adversarial_key in adversarial_name_keys:
            adversarial_mean_dict[adversarial_key] = {}
            adversarial_weight = updates[adversarial_key][1]
            for parameter_name in adversarial_weight:
                division = (adversarial_weight[parameter_name].cpu().numpy() - begnign_base_dict[parameter_name]) / begnign_base_dict[parameter_name]
                similarity_result = np.mean(np.abs(division))
                if parameter_name not in adversarial_mean_dict[adversarial_key]:
                    adversarial_mean_dict[adversarial_key][parameter_name] = similarity_result
                else:
                    adversarial_mean_dict[adversarial_key][parameter_name] += similarity_result

        for begnign_key in begnign_name_keys:
            begnign_mean_dict[begnign_key] = {}
            begnign_weight = updates[begnign_key][1]
            for parameter_name in begnign_weight:
                division = (begnign_weight[parameter_name].cpu().numpy() - begnign_base_dict[parameter_name]) / begnign_base_dict[parameter_name]
                similarity_result = np.round(np.mean(np.abs(division)), 2)
                if parameter_name not in begnign_mean_dict[begnign_key]:
                    begnign_mean_dict[begnign_key][parameter_name] = similarity_result
                else:
                    begnign_mean_dict[begnign_key][parameter_name] += similarity_result



        adversarial_layer_similarity = {}
        print("Attacker Record: ")
        for agent_key in adversarial_similarity_dict:
            print(agent_key, adversarial_similarity_dict[agent_key])
            
            for parameter_name in adversarial_similarity_dict[agent_key]:
                
                if parameter_name not in adversarial_layer_similarity:
                    adversarial_layer_similarity[parameter_name] = adversarial_similarity_dict[agent_key][parameter_name]
                else:
                    adversarial_layer_similarity[parameter_name] += adversarial_similarity_dict[agent_key][parameter_name]
                

        for parameter_name in adversarial_layer_similarity:
            adversarial_layer_similarity[parameter_name] = round(adversarial_layer_similarity[parameter_name] / (len(adversarial_name_keys)), 2)


        print("Average attacker record")
        print(adversarial_layer_similarity)


        begnign_layer_similarity = {}
        print("Bengin Record: ")
        for agent_key in begnign_similarity_dict:
            print(agent_key, begnign_similarity_dict[agent_key])
            for parameter_name in begnign_similarity_dict[agent_key]:
                if parameter_name in begnign_layer_similarity:
                    begnign_layer_similarity[parameter_name] += begnign_similarity_dict[agent_key][parameter_name]
                else:
                    begnign_layer_similarity[parameter_name] = begnign_similarity_dict[agent_key][parameter_name]

        for parameter_name in begnign_layer_similarity:
            begnign_layer_similarity[parameter_name] = round(begnign_layer_similarity[parameter_name] / (len(begnign_name_keys)), 2)


        if not write_header:
            similarity_other_file.write("Epoch  Agent  ")
            similarity_mean_file.write("Epoch  Agent  ")


        for parameter_name in begnign_layer_similarity:
            similarity_other_file.write(parameter_name + "  ")
            similarity_mean_file.write(parameter_name + "  ")
        similarity_other_file.write("\n")
        similarity_mean_file.write("\n")
        write_header = True


        similarity_other_file.write("Attacker: \n")
        for agent_key in adversarial_similarity_dict:
            similarity_other_file.write(str(epoch) + "  " + str(agent_key) + "  ")
            for parameter_name in adversarial_similarity_dict[agent_key]:
                similarity_other_file.write(str(adversarial_similarity_dict[agent_key][parameter_name]) + " ")
            similarity_other_file.write("\n")
        
        similarity_other_file.write("Attacker Average: \n      ")
        for parameter_name in adversarial_layer_similarity:
            similarity_other_file.write(str(adversarial_layer_similarity[parameter_name]) + "  ")
        similarity_other_file.write("\n")

        similarity_other_file.write("Begnign Worker: \n")
        for agent_key in begnign_similarity_dict:
            similarity_other_file.write(str(epoch) + "  " + str(agent_key) + "  ")
            for parameter_name in begnign_similarity_dict[agent_key]:
                similarity_other_file.write(str(begnign_similarity_dict[agent_key]) + "  ")
            similarity_other_file.write("\n")
        
        similarity_other_file.write("Begnign Average: \n      ")
        for parameter_name in begnign_layer_similarity:
            similarity_other_file.write(str(begnign_layer_similarity[parameter_name]) + "  ")
        similarity_other_file.write("\n")

        similarity_mean_file.write("Attacker: \n")
        for agent_key in adversarial_mean_dict:
            similarity_mean_file.write(str(epoch) + "  " + str(agent_key))
            for parameter_name in adversarial_mean_dict[agent_key]:
                similarity_mean_file.write(str(adversarial_mean_dict[agent_key][parameter_name]) + "  ")
            similarity_mean_file.write("\n")
        

        similarity_mean_file.write("Begnign Worker: \n")
        for agent_key in begnign_mean_dict:
            similarity_mean_file.write(str(epoch) + "  " + str(agent_key))
            for parameter_name in begnign_mean_dict[agent_key]:
                similarity_mean_file.write(str(begnign_mean_dict[agent_key][parameter_name]) + "  ")
            similarity_mean_file.write("\n")
        
                
        print("Average begnign record")
        print(begnign_layer_similarity)

        print("Attack mean record: ")
        for agent_key in adversarial_mean_dict:
            print(agent_key, adversarial_mean_dict[agent_key])
        print("Begnign mean record: ")
        for agent_key in begnign_mean_dict:
            print(agent_key, begnign_mean_dict[agent_key])

        ''''''
        
        
        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
            vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
            vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys)
            vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys )
            num_oracle_calls = 1

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params['is_poison']:

            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])


            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            if helper.params['vis_trigger_split_test']:
                helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,
                                                           eid=helper.params['environment_name'],
                                                           name="global_combine")
            if len(helper.params['adversary_list']) == 1:  # centralized attack
                if helper.params['centralized_test_trigger'] == True:  # centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, vis, epoch)
            else:  # distributed attack
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, vis, epoch)

        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)



    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")


    vis.save([helper.params['environment_name']])
    similarity_other_file.close()
    similarity_mean_file.close()