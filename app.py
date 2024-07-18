import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm
import keras.backend as K

# Custom loss and metric functions
def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# Load the model
satellite_model = load_model('model/full_segmentation_model2.h5',
                             custom_objects={'dice_loss_plus_1focal_loss': total_loss, 'jaccard_coef': jaccard_coef})

# Color mapping for classes
color_mapping = {
    0: [60, 16, 152],    # Building
    1: [132, 41, 246],   # Land
    2: [110, 193, 228],  # Road
    3: [254, 221, 58],   # Vegetation
    4: [226, 169, 41],   # Water
    5: [155, 155, 155]   # Unlabeled
}

# Function to process input image
def process_input_image(image_source):
    image_source = Image.fromarray(image_source).resize((256, 256))
    image = np.array(image_source) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)
    prediction = satellite_model.predict(image)
    predicted_image = np.argmax(prediction, axis=3)[0]
    
    # Create a color image from the prediction
    color_mask = np.zeros((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_mapping.items():
        color_mask[predicted_image == class_idx] = color
    
    return 'Predicted Masked Image', color_mask

# Gradio application
with gr.Blocks() as my_app:
    gr.Markdown("Satellite Image Segmentation Application UI with Gradio")
    with gr.Tabs():
        with gr.TabItem("Select your image"):
            with gr.Row():
                with gr.Column():
                    img_source = gr.Image(label="Please select source Image", type="numpy")
                    source_image_loader = gr.Button("Load above Image")
                with gr.Column():
                    output_label = gr.Label(label="Image Info")
                    img_output = gr.Image(label="Image Output")
    source_image_loader.click(
        process_input_image,
        [img_source],
        [output_label, img_output]
    )

my_app.launch(debug=True)