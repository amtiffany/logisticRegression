# rlogistic regression

Summary:
small program that trains a predictive model based on the mnist dataset for arabic numerals.

Explaination:
models predict the probability of any one image belonging to the target class, one class at a time.
training images are loaded in from the dataset and fed into the model using fit() to generate the weights 
and bias of the model. test images are then loaded and predicitons are generated using predict(). the number
predict returns for each image is a float from 0-1 representing the confidence in classifying the image as 
either class 0 or 1 (closeness to either number meaning higher confidence in that class). predicitons over
a certain threshold (0.5) are classified as the target.

In addition to simple binary classification, multi-class classification is achieved using multiple
binary classifiers and choosing the digit with the highest predicted probablity.

this program uses numpy for basic array operations and sklearn for the classification report.



Report:

Though simple, the binary logistic regression model produces suprizingly good results on arabic numerals.
When running simple is/is not binary classifiers, the accurate prediction can be as high as 95-99%, depending 
on the digit. The lowest accuracy occurs in digits that are similar to eachother, such as 3 and 8. Thus the 
high overall accuracy may be due to the fact that there are very few classes present in the dataset and 
therefore there is less similarity between classes causing mis-classification. The accuracy may be lower when 
trying to classify an alphabet, for example.

The classifier most often makes mistakes in trying to recognize the target. It almost never mis-identifies
other digits for the target, but more often fails to correctly classify the target digit. Even in digits with 
a low predicion accuracy, the rate of classifing another digit as the target is always very low (at least 97% 
accuracy), even when the rate of correctly recognizing the target digit is relatively low (as low as 81%, for 
the digit 8). This means that we can be reasonably confident in the set of digits that have been classified as 
the target, even if we cannot be confident that all target digits have been found.

The multi-digit classifier is also suprizingly accurate, though it is obviously less accurate than the single 
digit classifiers. the overall aaccuracy of the multidigit classifier is 90%, and the highest accuracy for a 
single digit is 95% for 1, the lowest 83% for 8. as with the single binary classifiers, accuracy is greatest 
for digits with no analogue like 0 and 1 and is lowest for similar numbers like 3 and 8.
