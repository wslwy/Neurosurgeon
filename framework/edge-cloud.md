-   cloud_api
    -   monitor_server
    -   start_server
        -   wait for client connection  //conn, client = wait_client(socket_server)
        -   get model type in short data and load model  //model_type = get_short_data(conn)
        -   get partition points in short data and load cloud parts to device
        -   get edge intermediate data and recode transfer latency
        -   do the cloud inference and recode inference latency
        -   finish DNN inference collaboration

-   edge_api
    -   monitor_client get connection bandwidth
    -   prepare data and load target model  //model = get_dnn_model(model_type)
    -   get best partition points according to the bandwidth and model feature  //partition_point = neuron_surgeon_deployment(model,network_type="wifi",define_speed=upload_bandwidth,show=True)
        -   test partition points in model devide model into edge and cloud parts
        -   use latency preditor to directly predict latency
            -   sum up each layers of model. use kernel_predictor_creator to finely get latency           
        -   predict transmission by getting intermediate data size and bandwidth
        -   sum up edge latency, cloud latency and transmission latency and get best partition points

    -   start DNN inference edge side
        -   send model type and partition points in short data to server
        -   load model edge parts to device and to the edge inference
        -   send edge intermediate data to server and recode inference latency
         
cloud-edge-end network
-   cloud_api
-   edge_api
-   end_api

single instance collaborative inference
    - edge server connect to cloud server
    - collaborative inference
    - end connection

multi instance collaborative inference
    - edge server connect to cloud server
    - multi instance collaborative inference
    - end connection

pipeline collaborative inference
    