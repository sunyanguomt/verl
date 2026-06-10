
ray start --head --port=65379 --dashboard-host=0.0.0.0 --dashboard-port=8872 # 当前节点为头节点 启动Ray

ray start --head --port=65379 --dashboard-host=0.0.0.0 --dashboard-port=8872 --num-cpus=64 --num-gpus=8 # 限制GPU, CPU 资源


# pkill -9 -f "ray::" && rm -rf /tmp/ray && ray stop --force # 停止Ray

ray start --address='10.202.40.141:65379' --dashboard-host=0.0.0.0 --dashboard-port=8872 --num-cpus=64 --num-gpus=8 # 新节点加入到已有头节点

ray start --address='10.202.40.21:65379' --dashboard-host=0.0.0.0 --dashboard-port=8872 --num-cpus=64 --num-gpus=8 
#--port：设置头节点的主要通信端口（默认6379）
#--node-manager-port：节点管理器端口
#--object-manager-port：对象管理器端口