import sys,getopt
sys.path.append("..") 
import warnings
warnings.filterwarnings("ignore")

import torch

from net import eec_netutils as net_utils
from net.monitor_server import MonitorServer

"""
    云端设备api 用于接收中间数据，并在云端执行剩余的DNN部分，将结果保存在excel表格中
    server 启动指令 python cloud_api.py -i 127.0.0.1 -p 9999 -d cpu
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"     是否开启服务端GPU计算 cpu or cuda

"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:p:d:", ["ip=","port=","device"])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    ip,port1 = "127.0.0.1",1514
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port1 = int(arg)
        elif opt in ("-d", "--device"):
            device = arg


    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")
    
    
    socket_server = net_utils.get_socket_server(ip,port1)
    # 等待edge server连接
    conn1, client = net_utils.wait_client(socket_server)
    while True:
        net_utils.start_cloud_server(conn1,device)
