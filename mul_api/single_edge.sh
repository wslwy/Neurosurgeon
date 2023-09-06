#    边缘设备api，用于启动边缘设备，进行前半部分计算后，将中间数据传递给云端设备
#    client 启动指令 python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net
#    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "le_net" "mobile_net"
#    "-i", "--ip"            服务端 ip地址
#    "-p", "--port"          服务端 开放端口
#    "-d", "--device"        是否开启客户端GPU计算 cpu or cuda
#    "-n", "--network"       网络类型 wifi or 3g or lte
#    "-s", "--speed"         网络速度 MB/s for wifi/lte  kB/s for 3g  


#python ../edge_api.py -i 127.0.0.1 -p 9999 -d cpu -s 5 -t vgg_net -n wifi

# "alex_net" "vgg_net" "le_net" "mobilenet"
python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t mobile_net -s 1 -n 3g