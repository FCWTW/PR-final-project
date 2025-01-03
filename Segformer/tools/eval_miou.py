from mmengine.config import Config
from mmseg.apis import inference_model, init_model
from mmseg.datasets import CityscapesDataset
from mmseg.evaluation import IoUMetric

# Use following command to calculate mIoU
# python test.py /home/wayne/Desktop/PR_final/segformer/config_b5.py /home/wayne/Desktop/PR_final/segformer/segformer_b5.pth --eval mIoU

# Load
config_file = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'
cfg = Config.fromfile(config_file)
model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')

dataset = CityscapesDataset(
    data_root='/media/wayne/614E3B357F566CB2/cityscapes/',
    img_dir='leftImg8bit/leftImg8bit/test',
    ann_dir='gtFine/gtFine/test',
    pipeline=cfg.test_pipeline
)

# 推理和评估
evaluator = IoUMetric(iou_metrics=['mIoU'])

for data in dataset:
    result = inference_model(model, data['img'])
    evaluator.process(data['gt_seg_map'], result.pred_sem_seg.data.cpu().numpy())

# 输出评估结果
metrics = evaluator.evaluate()
print(metrics)