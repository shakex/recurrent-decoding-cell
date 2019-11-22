# BrainWeb

- images downloaded from brainWeb website: https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html
- three modalities: t1, t2, pd
- Normal brain dataset: Slice thickness: 1mm, Noise: 0, Intensity non-uniformity: 0
- for each modality: 399 images for training and testing (the first 30 and last 30 images are removed)
- Images have the size of 217x181, 181x181 and 181x217 in three orthogonal views
- 60% for training, 40% for testing
- Skull stripping is conducted as the pre-processing technique before network training