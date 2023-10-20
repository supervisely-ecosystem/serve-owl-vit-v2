import sys
sys.path.append("/big_vision/")
import supervisely as sly
import os
from dotenv import load_dotenv
from typing import Literal
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

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

        # define class names
        self.class_names = ["object"]
        # list for storing box colors
        self.box_colors = []
    