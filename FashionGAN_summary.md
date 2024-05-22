Paper Contritbutions:

Changing the outfit of someone while maintaining body part positions.

1. Generating shape and textures in different stages.

Data Requirements:

- One photo of the user with a sentence describing the outfit.

Methods:

Two stage GAN that generates shape and textures in different stages.

Pre-processing:

Design coding is generated based on human features and text input. Full details in paper, but it is 40 dimensions of text encoding and 10 dimensions of non-parameterized human feature encodings.

Segmentation of original image.

Stage 1:

- Generates a human segmentation map that specifics the regions for body parts and upper-body garment of new clothing map.

Segmentation Map Generation:

The segmentation map is conditioned on a random gaussian noise vector of dimension 100, the design coding, and a low-resolution representation of the segmentation called “spacial constraint”.

For segmentation they have 7 labels. “Background, hair, face, upper-clothes, pants/shorts, legs, and arms”, in the original segmentation map.

In the spatial constraint merged-down there are 4 labels, background, hair, face, and rest. Rest represents the rest of the entire body. Purpose is to still encode body information but lose semantic information about clothes, so the model is able to generate that.

The goal of this stage is to generate a segmentation map that has the new clothing segments in mind. Consistent with design coding.

They use softmax at the end of the generator, as to make it consistent with segmentation.

Stage 2:

- Takes the segmentation map and textual description as conditions, then renders the region specific textures.

The texture for each semantic segmentation is generated separately. They call this compositional mapping. Then it is added together.

The mapping for face and hair are generated directly from the original image. Aka no GAN, as it is not necessary.

TRAINING GOAL:

Given input image x, and text description t.

Stage 1: Segment image x, generate feature vector description. Uses text input

The tuple, {down-sampled map, design encoding, segmentation map} are used in this GAN. down-sampled map and design encoding are conditions, segmentation map is what it is trying to generate.

Stage 2: Using segmentation, feature vector and text. Regenerate the original image.

Down sampled map, design encoding, and original image are used in this case.