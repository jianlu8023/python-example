import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import visualkeras


def model_structure_png(model: keras.Model):
    visualkeras.layered_view(model,
                             legend=True, legend_text_spacing_offset=0, ).show()
