import pickle
import torch
from utils import inference_utils
from predictor import predictor_utils
from net import net_utils

def get_layer(model,point):
    """
    get model's partition layer
    """
    if point == 0:
        layer = None
    else:
        layer = model[point - 1]
    return layer


def get_input(HW):
    """
    根据HW生成相应的pytorch数据 -> torch(1,3,224,224)
    :param HW: HW表示输入的高度和宽度
    :return: torch数据
    """
    return torch.rand(size=(1, 3, HW, HW), requires_grad=False)


def neuron_surgeon_deployment(model,network_type,define_speed,show=False):
    """
    为DNN模型选取最优划分点
    :param model: DNN模型
    :param network_type: 3g or lte or wifi
    :param define_speed: bandwidth
    :param show: 是否展示
    :return: 选取的最优partition_point
    """
    ee_layer_index  = 0
    ec_layer_index  = 0
    
    return ee_layer_index, ec_layer_index



