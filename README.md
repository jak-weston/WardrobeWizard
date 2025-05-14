# WardrobeWizard 🧙‍♂️👔
Implementation of cycle GAN-based model capable transforming a person’s clothing style based on user-specified attribute
Paper: https://arxiv.org/pdf/1710.07346

## Project Overview
WardrobeWizard is an AI-powered fashion manipulation system that uses advanced deep learning techniques to modify clothing styles in images based on text descriptions. The project implements a two-stage GAN (Generative Adversarial Network) architecture for precise clothing manipulation while preserving the person's identity.

## Key Features
- **Text-Guided Image Manipulation**: Transform clothing styles based on natural language descriptions
- **Two-Stage Generation**:
  - Shape Generation: Creates segmentation masks for desired clothing
  - Image Generation: Generates realistic clothing while preserving identity
- **Identity Preservation**: Maintains facial features and hair during clothing transformation
- **Conditional Generation**: Supports multiple clothing types and styles

## Technical Implementation
1. Reimplement G_shape GAN
   1. Create Data loader for training loop
   2. Implement GAN structure (generator / discriminator)
   3. Train
   4. Test
2. Reimplement G_image

### Architecture
- **Shape Generator**: Creates semantic segmentation masks from text descriptions
- **Image Generator**: Transforms segmentation masks into realistic clothing
- **Discriminators**: Ensure realistic and accurate generation for both stages
- **Conditional Inputs**: 
  - Text embeddings for style description
  - Segmentation masks for clothing regions
  - Noise vectors for style variation

### Model Components
- `G_shape.py`: Shape generation pipeline
- `G_image.py`: Image generation pipeline
- `G_shape_model.py`: Shape generator architecture
- `G_image_model.py`: Image generator architecture
- `dataloader.py`: Data preprocessing and loading utilities
- `segmentation.py`: Image segmentation utilities

## Results and Visualization
- `plot_G_shape_results.py`: Visualize shape generation results
- `plot_G_image_results.py`: Visualize final image generation results
- `net_graph_sr1.py`: Network architecture visualization

## Training and Evaluation
- Custom loss functions for realistic generation
- Identity preservation metrics
- Style transfer accuracy evaluation
- Segmentation accuracy metrics

## Technologies Used
- PyTorch
- CUDA for GPU acceleration
- NumPy
- Matplotlib
- H5py for data management

## Future Improvements
- Enhanced text understanding capabilities
- More diverse clothing style options
- Improved resolution and detail generation
- Real-time processing capabilities
- Mobile deployment support

## Installation and Usage
1. Clone the repository
2. Install dependencies:
```bash
pip install torch numpy h5py matplotlib
```
3. Prepare your data following the format in `dataloader.py`
4. Run the main script:
```bash
python main.py
```

## Author
- jak-weston
- Sean-Fuhrman

## License
This project is open source and available under the MIT License.
