# Hyperspectral-Image-Classification


Used The Indiana Pines data. The data is in .mat format (indian_pines_corrected.mat)...This contain the image: no of rows x no of columns x no of bands



In the ground truth file, the label per pixel is given: there are 16 labels in total.

In order to work with the softmax classifier, you have to convert the labels in the one hot format.



Considering 50% pixels per class as training and remaining 50% pixels as testing.



Task will be to design a softmax classifier. Training is to be done using the maximum likelihood criteria (in this case, this is also known as cross-entropy loss).

