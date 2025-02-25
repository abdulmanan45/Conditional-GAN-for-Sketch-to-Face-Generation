# Sketch2Face: Generating Realistic Faces from Sketches using Conditional GANs

👋 Hey there! Welcome to the **Sketch2Face** project!

This project is about turning sketches of faces into realistic-looking photos using **Conditional Generative Adversarial Networks (cGANs)**.


## Key Features

* **Sketch-to-Photo Magic:**  Generates realistic color face images from grayscale sketches.
* **Conditional GAN (cGAN) Power:** Leverages the power of cGANs to understand and translate sketches into photos.
* **U-Net Generator:** Uses a U-Net architecture for the generator network to capture details from the sketch.
* **PatchGAN Discriminator:** Employs a PatchGAN discriminator to ensure the generated faces are realistic in patches.
* **Person Face Sketches Dataset:** Trained and tested on the "Person Face Sketches" dataset.

## Model Architecture

We use two main AI models that work together:

* **Generator (The Artist):** It takes a sketch as input and tries to create a realistic face photo. It's built using a **U-Net architecture**, which is good at keeping important details from the sketch.

* **Discriminator (The Art Critic):** It looks at both real face photos and the photos created by the Generator. It tries to tell the difference between real and fake. We use a **PatchGAN discriminator** which looks at small patches of the image to make its decision.

These two models are trained together in an adverserial game. The Generator tries to fool the Discriminator, and the Discriminator tries to get better at spotting fakes. This back-and-forth is what makes the Generator get better and better at creating realistic faces.

Here's a simplified breakdown of the networks:

**Generator:**

* **Input:** Grayscale sketch (256x256 pixels).
* **U-Net Architecture:**  Uses a series of downsampling and upsampling layers with skip connections to preserve sketch details.
* **Output:** Color face image (256x256 pixels).

**Discriminator (PatchGAN):**

* **Input:**  A pair of images - a sketch and either a real face photo or a generated face photo.
* **Patch-Based Classification:**  Classifies 70x70 pixel patches of the image as real or fake.
* **Output:**  Determines if the given image is a plausible real photo corresponding to the sketch.

## Dataset

We trained and tested this model using the **Person Face Sketches Dataset**.  It's a collection of sketches and their corresponding real photos of faces. We split the dataset into three parts:

* **Training Dataset:** Used to teach the cGAN model how to generate faces from sketches.
* **Validation Dataset:**  Used during training to check how well the model is learning and to prevent it from overfitting.
* **Test Dataset:** Used to see how well the final trained model performs on new, unseen sketches.

## Loss Functions & Training

* **Generator Loss:**  We want the Generator to create images that are both realistic *and* look like the sketch. So, we use a combination of two losses:
    * **GAN Loss:**  This pushes the Generator to fool the Discriminator into thinking its generated images are real.
    * **L1 Loss:** This makes sure the generated image is pixel-by-pixel similar to the real photo that matches the sketch, ensuring it's not just realistic but also *accurate* to the sketch's content.

* **Discriminator Loss:** We want the Discriminator to correctly identify real photos and spot the fake ones from the Generator.  We use a **binary cross-entropy loss** for this.

The training process is as follows:

1. **Generator Creates a Fake Image:** The Generator takes a sketch and creates a fake face photo.
2. **Discriminator Judges:** The Discriminator looks at both real photos and the Generator's fakes and tries to classify them correctly.
3. **Models Learn and Improve:** Based on how well they performed, both the Generator and Discriminator adjust their internal settings to get better in the next round.  This is done using an **Adam optimizer**.

## Getting Started

Want to try it out or dive into the code? Here's how:

### Prerequisites


* **Python 3.10.14**
* **TensorFlow 2.16.1**
* **Keras 3.3.3**
* **Libraries:**  You'll need to install these Python libraries. You can do this using pip:
   ```bash
   pip install tensorflow matplotlib pathlib
   
 
### Dataset Setup

   * Download the Person Face Sketches Dataset: You can find it at https://www.kaggle.com/datasets/almightyj/person-face-sketches

 * Organize your data: Make sure your dataset is structured like this (the link has it set up like this):
<pre>
person-face-sketches/
├── train
│   ├── photos
│   └── sketches
├── val
│   ├── photos
│   └── sketches
└── test
    ├── photos
    └── sketches
</pre>

### Running the Code

* Clone the repository:
    ```bash
    cd [your-repository-directory]
    git clone https://github.com/abdulmanan45/Conditional-GAN-for-Sketch-to-Face-Generation


* Use the training notebook (c_gan_train.ipynb):
    <i> # Make sure to adjust dataset paths in the script .</i>


### Usage - Generating Faces from Your Sketches

* Once you have a trained model, you can generate face photos from the test sketches!

* Use the test generation notebook (c_gan_test_images.ipynb):
 <i> # Make sure to adjust dataset paths in the script.</i>
        

* The generated image will be saved and also shown in the notebook.

## Repository Files

Here's a quick overview of what you'll find in this repository:

* **c_gan_train.ipynb**: The main Jupyter Notebbok to train the Conditional GAN model.

*    **c_gan_test_images.ipynb**: Jupyter Notebbok to generate face images from new sketches using a trained model.

 *   **training_checkpoints/**: Directory to save model checkpoints during training. (Will be created during training).

 *   **generated_images/**: Directory where generated face images are saved. (Will be created when you run generation).

  *  **ConditionalGANforSketchtoFaceGeneration.pdf**: The research paper detailing the project.

   * **README.md**: This file!

## Acknowledgements

* **Person Face Sketches Dataset**: Thanks to the creators of the Person Face Sketches dataset for providing the data for this project.

* **Original cGAN and pix2pix Papers**: This project is inspired by and builds upon the foundational work in Conditional GANs and image-to-image translation, particularly the pix2pix paper.

## Generated Images
![comparison_plot_10](https://github.com/user-attachments/assets/14fc2d56-809f-4860-b051-3192df7e0122)
![comparison_plot_8](https://github.com/user-attachments/assets/54ecdd6e-1e90-4bef-9a58-d3d3c2407af8)
![comparison_plot_2](https://github.com/user-attachments/assets/7a648e8d-0ff9-44d7-b13e-b5e6a2bb99f0)


