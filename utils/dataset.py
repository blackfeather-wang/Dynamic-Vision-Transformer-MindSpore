"""
Create train or eval dataset.
"""
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as P
from mindspore.dataset.vision.py_transforms import MixUp
from mindspore.communication.management import init
from PIL import Image
from io import BytesIO

from .autoaugment import ImageNetPolicy

class ToNumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, resize_size=256,
                   crop_size=224, target="Ascend", num_threads=12, autoaugment=False, mixup=0.0, num_classes=1001):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    device_num = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv('RANK_ID'))

    if target == "GPU":
        init("nccl")

    if do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_threads, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)
    else:
        '''
        padded_sample = {}
        white_io = BytesIO()
        Image.new('RGB',(224,224),(255,255,255)).save(white_io, 'JPEG')
        padded_sample['data'] = white_io.getvalue()
        padded_sample['label'] = -1
        batch_per_step = batch_size * device_num
        print("eval batch per step:", batch_per_step)
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        print("padded samples:", num_padded)
        ds = de.MindDataset(dataset_path+'/imagenet_eval.mindrecord0', columns_list, num_parallel_workers=8, shuffle=False,
                            num_shards=device_num, shard_id=rank_id, padded_sample=padded_sample, num_padded=num_padded)
        print("eval dataset size", ds.get_dataset_size())
        '''
        padded_sample = {}
        white_io = BytesIO()
        Image.new('RGB',(crop_size, crop_size),(255, 255, 255)).save(white_io, 'JPEG')
        batch_per_step = batch_size * device_num
        print("eval batch per step: {}".format(batch_per_step))
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        padded_sample['image'] = np.asarray(bytearray(white_io.getvalue()), dtype='uint8')
        padded_sample['label'] = np.array(-1, np.int32)
        if num_padded != 0:
            sample = [padded_sample for x in range(num_padded)]
            ds_pad = de.PaddedDataset(sample)
            ds_imagefolder = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_threads)
            ds = ds_pad + ds_imagefolder
            distributeSampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id,
                                                      shuffle=False, num_samples=None)
            ds.use_sampler(distributeSampler)
        else:
            ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_threads, shuffle=False,
                                       num_shards=device_num, shard_id=rank_id)
            print("eval dataset size: {}".format(ds.get_dataset_size()))

    # from google tensorflow code
    # mean = [123.68, 116.78, 103.94]
    # std = [1.0, 1.0, 1.0]

    # from nvidia mxnet code
    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    # define map operations
    if do_train:
        if autoaugment:
            trans = [
                C.RandomCropDecodeResize(crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5),
                P.ToPIL(),
                ImageNetPolicy(),
                ToNumpy(),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW(),
                #C2.TypeCast(mstype.float16)
            ]
        else:
            trans = [
                C.RandomCropDecodeResize(crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW(),
                #C2.TypeCast(mstype.float16)
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(resize_size),
            C.CenterCrop(crop_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    if do_train:
        ds = ds.map(input_columns="image", num_parallel_workers=num_threads, operations=trans)
    else:
        ds = ds.map(input_columns="image", num_parallel_workers=num_threads, operations=trans)

    ds = ds.map(input_columns="label", num_parallel_workers=num_threads, operations=type_cast_op)

    #mixup
    if do_train and mixup > 0:
        one_hot_encode = C2.OneHot(num_classes)
        ds = ds.map(operations=one_hot_encode, input_columns=["label"])

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    #mixup
    if do_train and mixup > 0:
        trans_mixup = C.MixUpBatch(alpha=mixup)
        ds = ds.map(input_columns=["image", "label"], num_parallel_workers=num_threads, operations=trans_mixup)

        print("get in mixup......")

    return ds

