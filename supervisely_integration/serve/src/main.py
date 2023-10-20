import sys

sys.path.append("/big_vision/")
import supervisely as sly
import os
from dotenv import load_dotenv
from typing import Literal
import tensorflow as tf
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.notebooks import inference
from scenic.model_lib.base_models import box_utils

load_dotenv("supervisely_integration/serve/debug.env")
load_dotenv(os.path.expanduser("./supervisely.env"))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
models_data_path = "./supervisely_integration/models/models_data.json"


class OWLViTv2Model(sly.nn.inference.PromptBasedObjectDetection):
    def get_models(self):
        models_data = sly.json.load_json_file(models_data_path)
        return models_data

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Rectangle, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        if device.startswith("cuda"):
            # set GPU as visible device
            gpus = tf.config.list_physical_devices("GPU")
            tf.config.set_visible_devices(gpus[0], "GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            # hide GPUs from visible devices
            tf.config.set_visible_devices([], "GPU")
        # load model
        selected_model = self.gui.get_checkpoint_info()["Model"]
        if selected_model == "OWLv2 CLIP B/16 ST":
            config = configs.owl_v2_clip_b16.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWLv2 CLIP B/16 ST+FT":
            config = configs.owl_v2_clip_b16.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWLv2 CLIP B/16 ST/FT ens":
            config = configs.owl_v2_clip_b16.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWLv2 CLIP L/14 ST":
            config = configs.owl_v2_clip_l14.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWLv2 CLIP L/14 ST+FT":
            config = configs.owl_v2_clip_l14.get_config(init_mode="canonical_checkpoint")
        elif selected_model == "OWLv2 CLIP L/14 ST/FT ens":
            config = configs.owl_v2_clip_l14.get_config(init_mode="canonical_checkpoint")
        module = models.TextZeroShotDetectionModule(
            body_configs=config.model.body,
            objectness_head_configs=config.model.objectness_head,
            normalize=config.model.normalize,
            box_bias=config.model.box_bias,
        )
        variables = module.load_variables(config.init_from.checkpoint_path)
        self.model = inference.Model(config, module, variables)
        self.model.warm_up()
        # define class names
        self.class_names = ["object"]
        # list for storing box colors
        self.box_colors = []
