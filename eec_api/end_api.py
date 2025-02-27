import torch
import sys,getopt
sys.path.append("..") 
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from multiprocessing    import Process
import multiprocessing

from net import eec_netutils as net_utils
from utils.inference_utils  import get_dnn_model
from utils.eec_deployment   import neuron_surgeon_deployment
from net.monitor_client import MonitorClient


"""
    边缘设备api，用于启动边缘设备，进行前半部分计算后，将中间数据传递给云端设备
    client 启动指令 python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net
    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "le_net" "mobile_net"
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"        是否开启客户端GPU计算 cpu or cuda
    "-n", "--network"       网络类型 wifi or 3g or lte
    "-s", "--speed"         网络速度MB/s for wifi/lte  kB/s for 3g  
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:i:p:d:n:s:", ["type=","ip=","port=","device=","network=","speed="])
        
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    model_type = "vgg_net"
    ip,port0 = "127.0.0.1",1145
    device = "cpu"
    network_type = "wifi"
    speed = 10
    
    for opt, arg in opts:
        if opt in ("-t", "--type"):
            model_type = arg
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port0 = int(arg)
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-n", "--network"):
            network_type = arg
        elif opt in ("-s", "--speed"):
            speed = int(arg)
                  

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")

    # step2 准备input数据
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)

    # 客户端进行传输
    model = get_dnn_model(model_type)

    # 部署阶段 - 选择优化分层点
    upload_bandwidth    = speed   
    ee_layer_index, ec_layer_index  = neuron_surgeon_deployment(model,network_type="wifi",define_speed=upload_bandwidth,show=True)

    # 使用云边协同的方式进行模拟
    net_utils.start_end_client(ip,port0,x,model_type,ee_layer_index, ec_layer_index, device)

