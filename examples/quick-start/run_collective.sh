export FLAGS_sync_nccl_allreduce=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_batchnorm_spatial_persistent=1

export GLOG_v=1
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export NCCL_DEBUG=INFO
# Unset proxy
unset https_proxy http_proxy

set -xe

python -m paddle.distributed.launch \
    --cluster_node_ips=127.0.0.1 \
	--node_ip=127.0.0.1 \
	--selected_gpus="0,1,2,3,4,5,6,7" \
	--log_dir=mylog \
    collective_train.py
