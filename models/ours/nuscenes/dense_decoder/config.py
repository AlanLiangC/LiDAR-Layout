_base_ = ["./_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = False
save_path="./logs/dense_decoder_gaus_10cm"
weight="../models/ours/nuscenes/dense_decoder/model_last.pth"

# model settings
model = dict(
    type="DenseDecoderV0",
    num_classes=16,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    head=dict(
        type="GSDecoder",
        feat_dim=64,
    ),
    criteria=dict(
        type="GSLoss",
    ),
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "NuScenesCubeDecodeDataset"
data_root = "/home/alan/AlanLiang/Dataset/pointcept_nuscenes"
# data_root = "/mnt/scratch/e/e1493786/AlanLiang/Dataset/pointcept_nuscenes"
ignore_index = -1
names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

data = dict(
    load_semantic=False,
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="FiltPoint", 
                 point_cloud_range=(-51.2, -51.2, -51.2, 51.2, 51.2, 51.2),
                 range_filter = [1.0, 56.0]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="ToRange",
                size=[32, 1024],
                fov=[ 10,-30 ],
                depth_range=[ 1.0,56.0 ],
                depth_scale=5.84,  # np.log2(depth_max + 1)
                log_scale=True),
            dict(type="CoordConvert", voxel_size=0.1, mask=True, p=0.8),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "range_img", "ray_drop"),
                feat_keys=("coord"))
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="FiltPoint", 
                 point_cloud_range=(-51.2, -51.2, -51.2, 51.2, 51.2, 51.2),
                 range_filter = [1.0, 56.0]),
            dict(
                type="ToRange",
                size=[32, 1024],
                fov=[ 10,-30 ],
                depth_range=[ 1.0,56.0 ],
                depth_scale=5.84,  # np.log2(depth_max + 1)
                log_scale=True),
            dict(type="CoordConvert", voxel_size=0.1, mask=True, p=0.8),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "range_img", "ray_drop"),
                feat_keys=("coord"))
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="FiltPoint", 
                 point_cloud_range=(-51.2, -51.2, -51.2, 51.2, 51.2, 51.2),
                 range_filter = [1.0, 56.0]),
            dict(
                type="ToRange",
                size=[32, 1024],
                fov=[ 10,-30 ],
                depth_range=[ 1.0,56.0 ],
                depth_scale=5.84,  # np.log2(depth_max + 1)
                log_scale=True),
            dict(type="CoordConvert", voxel_size=0.1, mask=True, p=0.8),
            dict(
                type="GridSample",
                grid_size=0.1,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "range_img", "ray_drop"),
                feat_keys=("coord"))
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver_woEval", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

inference = dict(type="Inferencer")
