#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

source /root/archiconda3/bin/activate ci3.7
source scripts/env.sh

export SKT_ENABLE=1  # superkernel
export SERVER_ID=0

device_str=$1

case $device_str in
    0-7)
        device_list=(0 1 2 3 4 5 6 7)
        export RANK_TABLE_FILE=configs/rank_table_8p.json
        ;;
esac

experiment_name=$2
shift;
shift;
args="$*"

device_num=${#device_list[@]}
export DEVICE_NUM=$device_num
export RANK_SIZE=$device_num

cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
avg_core_per_rank=`expr $cores \/ 8`
core_gap=`expr $avg_core_per_rank \- 1`

cur_path=$PWD
datetime="$(date +'%Y_%m_%d-%H_%M_%S')"
for (( i=$DEVICE_NUM-1 ; i>=0 ; i-- )) ;
do
    start=`expr $i \* $avg_core_per_rank`
    end=`expr $start \+ $core_gap`
    cmdopt=$start"-"$end

    export DEVICE_ID=${device_list[${i}]}
    export RANK_ID=$i

    results_dir="../results/"${datetime}"__"${experiment_name}

    subdir=$results_dir"/rank_"$i
    rm -rf ${subdir}
    mkdir -p ${subdir}

    CACHE="../results/cache/kernel_meta"
    if [ -d "$CACHE" ]; then
        cp -r ${CACHE} ${subdir}
    fi

    cp ./*py ${subdir}
    cp -r utils ${subdir}
    cp -r nn ${subdir}
    cp -r configs ${subdir}
    cp -r networks ${subdir}

    cd ${subdir} || exit
    env > env.log

    if [ $i -eq 0 ]; then
        taskset -c $cmdopt python train.py ${args} 2>&1 | tee log_$RANK_ID.log
    else
        taskset -c $cmdopt python train.py ${args} &> log_$RANK_ID.log &
    fi

    cd ${cur_path}
done
echo "Finished"
