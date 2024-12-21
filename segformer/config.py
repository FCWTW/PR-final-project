_base_ = [
    '/home/wayne/Desktop/PR_final/segformer/__base__/segformer_mit_b0.py',
    '/home/wayne/Desktop/PR_final/segformer/__base__/cityscapes_1024.py',
    '/home/wayne/Desktop/PR_final/segformer/__base__/default_runtime.py',
    '/home/wayne/Desktop/PR_final/segformer/__base__/schedule_160k.py'
]

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5000),  # output log every 5000 iteration
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3),  # 每 50 epoch 儲存一次模型，最多保留 3 個檢查點
    sampler_seed=dict(type='DistSamplerSeedHook')
)