import sys
sys.path.append("/big_vision/")
from scenic.common_lib import export_utils
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.clip import model as clip_model
from scenic.projects.owl_vit.configs import owl_v2_clip_b16

config = owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
module = models.TextZeroShotDetectionModule(
    body_configs=config.model.body,
    normalize=config.model.normalize,
    box_bias=config.model.box_bias)
variables = module.load_variables(config.init_from.checkpoint_path)