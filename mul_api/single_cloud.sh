
#    云端设备api 用于接收中间数据，并在云端执行剩余的DNN部分，将结果保存在excel表格中
#    server 启动指令 python cloud_api.py -i 127.0.0.1 -p 9999 -d cpu
#    "-i", "--ip"            服务端 ip地址
#    "-p", "--port"          服务端 开放端口
#    "-d", "--device"     是否开启服务端GPU计算 cpu or cuda

python cloud_api.py -i 127.0.0.1 -p 9999 -d cuda