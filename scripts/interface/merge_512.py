import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from lib.models.tools.model_manager import ModelManager
from lib.utils.helpers.runner_helper import RunnerHelper
from lib.utils.tools.configer import Configer



class MergeClsModel(nn.Module):
	def __init__(self,net1,net2):
		super(MergeClsModel, self).__init__()
		self.net1 = net1
		self.net2 = net2
	def forward(self,x):
		x1 = self.net1(x)
		x2 = self.net2(x)
		x = torch.cat((x1,x2),0)
		return x

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--net_type', default="", type=str,
					dest='deploy.net_type', help='The deploy model type.')
parser.add_argument('--pool_type', default='AVE', type=str,
					dest='deploy.pool_type', help='The pool type.')
parser.add_argument('--norm_type', default='L2', type=str,
					dest='deploy.norm_type', help='The pool type.')
parser.add_argument('--gpu_id', default=[0,1], type=int,
					dest='gpu_id', help='The gpu id.')
args = parser.parse_args()


#second
model_path1='../../checkpoints/cls/resnet152/google_landmark_2020_resnet152_v2cluster_448_GPU8_final.pth'
model_path2='../../checkpoints/cls/resnest200/google_landmark_2020_resnest200_v2cluster_448_GPU8_final.pth'

checkpoint_dict1 = torch.load(model_path1)
configer1 = Configer(config_dict=checkpoint_dict1['config_dict'], args_parser=args, valid_flag="deploy")
net1 = ModelManager(configer1).get_deploy_model()
RunnerHelper.load_state_dict(net1, checkpoint_dict1['state_dict'], False)



checkpoint_dict2 = torch.load(model_path2)
configer2 = Configer(config_dict=checkpoint_dict2['config_dict'], args_parser=args, valid_flag="deploy")
net2 = ModelManager(configer2).get_deploy_model()
RunnerHelper.load_state_dict(net2, checkpoint_dict2['state_dict'], False)



net = MergeClsModel(net1,net2)
device = torch.device('cpu')
net = net.to(device).eval()
dummy_input = torch.randn(1, 3, 512, 512).to(device)
out = net(dummy_input)
print(out.shape)
input_names = ["input"]
output_names = ["global_descriptor"]
save_path = 'merge_second.onnx'
torch.onnx.export(net, dummy_input, save_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=10)
