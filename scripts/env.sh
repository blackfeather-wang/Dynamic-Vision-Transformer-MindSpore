source /root/archiconda3/bin/activate ci3.7

export SLOG_PRINT_TO_STDOUT=0
export GLOG_v=3  # 1 for info
# >>> me env >>>
LOCAL_HIAI=/usr/local/Ascend
export TBE_IMPL_PATH=${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/impl/
export LD_LIBRARY_PATH=${LOCAL_HIAI}/fwkacllib/lib64/:${LOCAL_HIAI}/add-ons/:${LOCAL_HIAI}/driver/lib64/common:${LD_LIBRARY_PATH}
export PATH=${LOCAL_HIAI}/fwkacllib/ccec_compiler/bin/:${PATH}
export PYTHONPATH=${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/:${PYTHONPATH}
# export PYTHONPATH=${LOCAL_HIAI}/opp/op_impl/built-in/ai_core/tbe/:/data/cgf/mindspore_t2t_vit/src/mindspore:${PYTHONPATH}
export MINDSPORE_CONFIG_PATH=${MINDSPORE_DIR}/config/

export MSLIBS_SERVER=10.29.74.101
export MSLIBS_CACHE_PATH=/data/zhaoting/.mslib/

export CONDA_PYTHON_EXE=/root/archiconda3/bin/python
