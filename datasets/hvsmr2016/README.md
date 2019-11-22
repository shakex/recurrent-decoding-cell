# HVSMR2016 Segmentation

- images downloaded from HVSMR 2016 website: http://segchd.csail.mit.edu/data.html
- one MR modality only
- for each modality: 399 images for training and testing (the first 30 and last 30 images are removed)
- 10 MRI volumes and their ground truth annotation for network training and evaluation, among which 1868 slices and 1473 slices are utilized for training and testing
- no pre-processing before network training

### Description (see http://segchd.csail.mit.edu/data.html)
Manual segmentation of the blood pool and ventricular myocardium was performed by a trained rater, and validated by two clinical experts. Segmentations were done in an approximate short-axis view and then transformed back to the original image space (axial view). Manual segmentation was done considering all three planes, but the quality of the segmentation in the short-axis view was the deciding factor.

The blood pool class includes the left and right atria, left and right ventricles, aorta, pulmonary veins, pulmonary arteries, and the superior and inferior vena cava. Vessels (except the aorta) are extended only a few centimeters past their origin: this is because vessels that are too long are disruptive when the 3D heart surface models are used for surgical planning. The myocardium class includes the thick muscle surrounding the two ventricles and the septum between them. Coronaries are not included in the blood pool class, and are labeled as myocardium if they travel within the ventricular myocardium.

