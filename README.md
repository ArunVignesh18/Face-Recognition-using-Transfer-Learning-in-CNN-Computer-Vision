# Face-Recognition-using-Transfer-Learning-in-CNN-Computer-Vision
This Project helps to detect person's face using the minimal dataset given by the user with the help of Transfer Learning In Convolutional Neural Network (AI)

Face Recognition using Tensor Flow and FaceNet.</br>
Goal: To generate a model which recognises the faces, with images given as input.


To get face feature embeddings, we used FaceNet model.
FaceNet is a one-shot model, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. We used embeddings from FaceNet to get features which are further used to predict the class representing face of a particular person.

## PREPARING DATA ##
### Training Data ### 
 Images of 11 persons including me (4 images each) were loaded using matplotlib and openCV.</br>
 Embedder from FaceNet is used to get the final feature vector.</br>
 Labels for the training data were given manually as an array with values as their names representing 11 different persons.
 
 ### Model ###
 On the extracted feature vector, a multiclass logistic regression was applied to learn a classification model.
 
 ### Prediction ###
 If the camera detects a person's face from the dataset, it gives that person's name as output. Or else a person with similar face features name will be given as output.

 ### Justification of Accuracy ###
 The FaceNet is trained on a huge dataset of face images, with lot of variation. When we applied it on our small dataset(44 images for 11 persons), all the face features were embedded perfectly. When we applied the multiclass classification on those features, it learnt the variation perfectly resulting in 100% accuracy.
