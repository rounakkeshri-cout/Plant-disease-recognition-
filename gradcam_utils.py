"""
gradcam_utils.py
-----------------
Drop-in Grad-CAM utility for the Plant Disease Recognition System.

Works with:
  - The plain 38-class CNN disease classifier (trained_model.h5)
  - The EfficientNetV2B0-based leaf-gate model (leaf_detector.h5), since it
    auto-detects the last conv layer even inside a nested/transfer-learning
    sub-model.

No retraining required -- this only reads gradients from the already-trained model.
Zero external plotting/image libraries required: only numpy, tensorflow, PIL.
"""

__version__ = "no-matplotlib-no-opencv-v1"

import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model):
    """
    Walk the model backwards and return the name of the last Conv2D-type
    layer (falling back to any 4D-output layer if no explicit Conv2D is
    found). Handles both plain Sequential/Functional CNNs and models that
    wrap a pretrained backbone (e.g. EfficientNetV2B0) as a single nested
    sub-model. Compatible with both legacy `layer.output_shape` and
    Keras 3's `layer.output.shape`.
    """
    def layer_output_ndims(layer):
        try:
            shape = layer.output_shape
        except AttributeError:
            try:
                shape = tuple(layer.output.shape)
            except Exception:
                shape = None
        return shape

    # Pass 1: look for an explicit Conv-type layer (preferred, matches the
    # standard Grad-CAM formulation which uses the last convolutional layer).
    for layer in reversed(model.layers):
        cls_name = layer.__class__.__name__
        if "Conv" in cls_name:
            shape = layer_output_ndims(layer)
            if isinstance(shape, (tuple, list)) and len(shape) == 4:
                return layer.name
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 0:
            inner_name = find_last_conv_layer(layer)
            if inner_name is not None:
                return inner_name

    # Pass 2: fall back to any layer with a 4D (spatial) output.
    for layer in reversed(model.layers):
        shape = layer_output_ndims(layer)
        if isinstance(shape, (tuple, list)) and len(shape) == 4:
            return layer.name
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 0:
            inner_name = find_last_conv_layer(layer)
            if inner_name is not None:
                return inner_name
    return None


def _build_grad_model(model, last_conv_layer_name):
    """
    Rebuild the forward pass layer-by-layer on a fresh Input, guaranteeing a
    single connected graph from input -> last_conv_layer -> final output.
    This is more robust than `Model(model.inputs, [layer.output, model.output])`
    across Keras versions/model types (Sequential, Functional, and models
    wrapping a nested transfer-learning backbone such as EfficientNetV2B0,
    where the target conv layer lives one level down inside a sub-model).
    """
    input_shape = model.inputs[0].shape[1:]
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_out = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if layer.name == last_conv_layer_name:
            x = layer(x)
            conv_out = x
            continue

        # Nested sub-model (e.g. a frozen/transfer-learning backbone) that
        # contains the target layer one level down.
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 0:
            inner_names = [l.name for l in layer.layers]
            if last_conv_layer_name in inner_names:
                # Extract BOTH the intermediate activation and the sub-model's
                # own final output from a single call, so both remain part of
                # the same differentiable trace (calling the sub-model twice
                # would create two disconnected graphs and break gradients).
                nested_extractor = tf.keras.Model(
                    layer.inputs,
                    [layer.get_layer(last_conv_layer_name).output, layer.output],
                )
                conv_out, nested_final = nested_extractor(x)
                x = nested_final
                continue

        x = layer(x)

    if conv_out is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' could not be located while rebuilding the graph.")
    return tf.keras.Model(inputs, [conv_out, x])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed input tensor, shape (1, H, W, 3), already normalised
               exactly the way it is fed to model.predict() elsewhere in the app.
    model: the loaded Keras model (e.g. trained_model.h5 or leaf_detector.h5).
    last_conv_layer_name: optional; auto-detected if not supplied.
    pred_index: optional class index to explain; defaults to the predicted class.

    Returns: (heatmap, pred_index, confidence)
        heatmap -- 2D numpy array in [0, 1], same spatial size as the last conv layer.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError(
                "Could not automatically find a convolutional layer in this model. "
                "Pass last_conv_layer_name explicitly (see model.summary())."
            )

    grad_model = _build_grad_model(model, last_conv_layer_name)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if predictions.shape[-1] == 1:
            # Binary sigmoid output (leaf-gate model) -- single score
            class_channel = predictions[:, 0]
            pred_index = 0
            confidence = float(predictions[0][0])
        else:
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]
            confidence = float(predictions[0][pred_index])

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradient computation returned None. This usually means the chosen "
            "layer is not connected to the output in a differentiable path."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # alpha_k^c, Eq. 2.1

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)  # ReLU + normalise, Eq. 2.2
    return heatmap.numpy(), pred_index, confidence


def _jet_colormap(x):
    """
    Pure-numpy approximation of the classic 'jet' colormap
    (blue -> cyan -> green -> yellow -> red as x goes 0 -> 1).
    x: array of floats in [0, 1]. Returns array of shape (..., 3) in [0, 1].
    No matplotlib/OpenCV dependency required.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    original_image: PIL Image or numpy array (H, W, 3), RGB, uint8, NOT normalised.
    heatmap: 2D array in [0, 1] as returned by make_gradcam_heatmap().
    Returns: numpy array (H, W, 3), RGB, uint8 -- ready for st.image().

    Uses only PIL + pure numpy for colour mapping (no OpenCV, no matplotlib).
    """
    if hasattr(original_image, "convert"):  # PIL Image
        original_image = original_image.convert("RGB")
        orig_arr = np.array(original_image)
    else:
        orig_arr = np.array(original_image).astype(np.uint8)

    h, w = orig_arr.shape[:2]

    # Resize the small heatmap up to the original image size using PIL
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_uint8).resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0  # back to [0, 1]

    # Colourise using a pure-numpy "jet" colormap (red = high influence)
    heatmap_colored = _jet_colormap(heatmap_resized)  # values in [0, 1]
    heatmap_colored = np.uint8(255 * heatmap_colored)

    # Alpha-blend the colour heatmap onto the original image
    overlayed = orig_arr.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    return overlayed
