from ptflops import get_model_complexity_info
import torch
from mmseg.models import build_segmentor
from mmseg.apis import init_model, inference_model
from mmengine import Config
from torch.profiler import profile, record_function, ProfilerActivity

# Load model
config_file = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'
cfg = Config.fromfile(config_file)
model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')
input_shape = (3, 512, 1024)
dummy_input = torch.randn(input_shape).cuda()

# Calculate FLOPs and Params
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
#     with record_function("model_inference"):
#         model(dummy_input)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True, verbose=True)

print(f"FLOPs: {macs}, Params: {params}")