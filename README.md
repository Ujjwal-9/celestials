# Celestials

## Section-B

It can be more flexible to predict probabilities of an observation belonging to each class in a classification problem rather than predicting classes directly.

This flexibility comes from the way that probabilities may be interpreted using different thresholds that allow the operator of the model to trade-off concerns in the errors made by the model, such as the number of false positives compared to the number of false negatives. This is required when using models where the cost of one error outweighs the cost of other types of errors.

Two diagnostic tools that help in the interpretation of probabilistic forecast for binary (two-class) classification predictive modeling problems are ROC Curves and Precision-Recall curves.

ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for imbalanced datasets.

**A precision-recall curve of a perfect classifier**

A classifier with the perfect performance level shows a combination of two straight lines – from the top left corner (0.0, 1.0) to the top right corner (1.0, 1.0) and further down to the end point (1.0, P / (P + N)).

So from our precision recall graph for SVM with **Linear, Polynomial and Radial basis function (rbf) kernel**, we can evaluate that they are almost perfect classifier for the given data. While sigmoid kernel SVM and Neural Network (Multi-Layer Perceptron) Model showed a bit shakiness in classification of the data.

**Precision-recall curves for multiple models**

It is easy to compare several classifiers in the precision-recall plot. Curves close to the perfect precision-recall curve have a better performance level than the ones closes to the baseline. In other words, a curve above the other curve has a better performance level. So in pur case neural network model emerged better than sigmoid kernel SVM.


**Rigorous hyperparameter tuning is not done in above task**

## Section C

Disparity map refers to the apparent pixel difference or motion between a pair of stereo images. To experience this, try closing one of your eyes and then rapidly close it while opening the other. Objects that are close to you will appear to jump a significant distance while objects further away will move very little.

### Calculating SAD
Let's say this window is n x n in size. You would also have some window in your left image WL and some window in your right image WR. The idea is to find the pair that has the smallest SAD.

So, for each left window pixel pl at some location in the window (x,y) you would the absolute value of difference of the right window pixel pr also located at (x,y). you would also want some running value, which is the sum of these absolute differences.
After you calculate the SAD for this pair of windows, WL and WR you will want to "slide" WR to a new location and calculate another SAD. You want to find the pair of WL and WR with the smallest SAD - which you can think of as being the most similar windows. In other words, the WL and WR with the smallest SAD are "matched". When you have the minimum SAD for the current WL you will "slide" WL and repeat.

Disparity is calculated by the distance between the matched WL and WR. For visualization, you can scale this distance to be between 0-255 and output that to another image.
