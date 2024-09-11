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


from collections import OrderedDict
from typing import Tuple
from tqdm import trange
import numpy as np
import shutil
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from _warnings import warn
from time import time, sleep
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params, get_default_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.loss_functions.bt_contrast_loss import Barlow_twins_loss
from nnunet.training.loss_functions.feature_distillation import feature_distillation
from nnunet.training.loss_functions.self_distillation import self_distillation
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *



class nnUNetTrainerV2_Stroke(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, ori_plans_file=None, output_folder=None, pretrained_directory=None, dataset_directory=None, \
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.pretrained_directory_base = pretrained_directory
        self.ori_plans_file = ori_plans_file
        self.max_num_epochs = 100
        self.initial_lr = 1e-4
        self.load_from_pretrained = True
        self.ct_loss = None
        self.diff_feat_loss = None
        self.self_dis_loss = None
        self.ds_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.teacher_network: Tuple[SegmentationNetwork, nn.DataParallel] = None


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans, self.ori_plans)#

            self.setup_DA_params()
            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.ds_loss = MultipleOutputLoss2(self.ds_loss, self.ds_loss_weights, len(self.net_num_pool_op_kernel_sizes))
            ################### END ###################
            self.ct_loss = Barlow_twins_loss(self.batch_size)
            self.self_dis_loss = self_distillation(self.batch_size)
            self.diff_feat_loss = feature_distillation(len(self.net_num_pool_op_kernel_sizes))
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset (convert npz-->npy)")
                    unpack_dataset(self.folder_with_preprocessed_data) #convert npz-->npy
                    #unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_default_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        !!!![NOTE change]: load pretrained teacher model 
                         self.network is initialized based on ct_plans_file
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            # norm_op = nn.InstanceNorm2d
            norm_op = nn.GroupNorm

        norm_op_kwargs = {'num_groups': 32, 'eps': 1e-5, 'affine': True}
        # norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        teacher_path = join(self.pretrained_directory_base, 'fold_'+str(self.fold))
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if self.teacher_network == None:
            self.teacher_network = Generic_UNet(self.teacher_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
            self.load_best_checkpoint(teacher_path)
        if torch.cuda.is_available():
            self.network.cuda()
            self.teacher_network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)
        print('[Loading ct plans file from %s]'%self.plans_file)
        self.ori_plans = load_pickle(self.ori_plans_file)
        print('[Loading multi_modalities plans file from %s]'%self.ori_plans_file)

    def process_plans(self, plans, ori_plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  
        self.intensity_properties = ori_plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = ori_plans['normalization_schemes']
        self.use_mask_for_norm = ori_plans['use_mask_for_norm']
        self.teacher_input_channels = ori_plans['num_modalities'] - 1
        
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  
        self.classes = plans['all_classes']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
            print('Set threeD to TRUE!!!(from trainerV2_Stroke-process_plans)')
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this has been added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def load_best_checkpoint(self, load_path=None, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if load_path is None:
            if isfile(join(self.output_folder, "model_best.model")):
                self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)
            else:
                self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                    "back to load_latest_checkpoint")
                self.load_latest_checkpoint(train)
        else:
            if isfile(join(load_path, "model_best.model")):
                self.print_to_log_file("loading pretrained dwi checkpoint from", join(load_path, "model_best.model"))
                saved_model = torch.load(join(load_path, "model_best.model"))
                self.teacher_network.load_state_dict(saved_model['state_dict'])
            else:
                self.print_to_log_file("WARNING! pretrained model_best.model does not exist! Cannot load pretrained best checkpoint.")

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            if self.load_from_pretrained:
                load_path = self.dataset_directory.replace('Task', 'pretrained_Task')
                splits_file = join(load_path, "splits_final.pkl")
                shutil.copy(splits_file, self.output_folder_base)
            else:
                splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        
        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)
        
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            self.teacher_network.eval()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)
        
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            ## Validation
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
                    
            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))

        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
        
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        ct = data[:,[0],:,:]
        mct = data[:,[-1],:,:]
        mm = data[:,:-1,:,:]

        ct = maybe_to_torch(ct)
        mct = maybe_to_torch(mct)
        mm = maybe_to_torch(mm)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            ct = to_cuda(ct)
            mct = to_cuda(mct)
            mm = to_cuda(mm)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        ct_outputs = self.network(ct)
        mct_outputs = self.network(mct)

        with torch.no_grad():
            mm_outputs = self.teacher_network(mm)
        del ct, mct, mm
        l = self.compute_loss(ct_outputs, mct_outputs, mm_outputs, target)

        if do_backprop:
            l.backward()
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(ct_outputs, target)

        del target

        return l.detach().cpu().numpy()
    
    def compute_loss(self, ct_outputs, mct_outputs, mm_outputs, target):
        total_loss = None
        # voting
        ct_sup_loss = self.ds_loss(ct_outputs, target)
        mct = []
        for i in range(len(mct_outputs)):
            mct.append(torch.flip(mct_outputs[i], dims=(3,)))
        mct_sup_loss = self.ds_loss(mct, target)
        if ct_sup_loss > mct_sup_loss:
            sup_loss = ct_sup_loss
        else:
            sup_loss = mct_sup_loss

        ### offline distillation
        region_bt_loss = self.ct_loss(ct_outputs[0], mct_outputs[0])
        diff_feat_loss = self.diff_feat_loss(ct_outputs, mct_outputs, mm_outputs)

        ### self-distillation
        self_dis_loss =  self.self_dis_loss(ct_outputs, mct_outputs)
        
        total_loss = 0.8 * sup_loss + 1.5 * diff_feat_loss + 1.5 * region_bt_loss + 0.5 * self_dis_loss

        return total_loss
        