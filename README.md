Updates to the Original Project
The original academic project report is written in Serbian. After completing the project, we updated the code to improve the model evaluation process.

In the updated version, the dataset is split into training and test sets before preprocessing. Numerical features are standardized, while categorical features are transformed using one-hot encoding. These steps are included in a scikit-learn pipeline, so preprocessing is fitted separately within each cross-validation fold.

Linear Discriminant Analysis (LDA) is also included as an optional pipeline step. For each classifier, GridSearchCV selects the model parameters and whether to use LDA based on cross-validation accuracy. The selected configuration is then evaluated on the test set.
Updated Results
The results below were obtained using one stratified train–test split, with 85% of the data for training and 15% for testing.

|        Model        | Selected configuration | CV accuracy | Test accuracy |
| Logistic Regression |      Without LDA       |    95.60%   |     94.32%    |
| K-Nearest Neighbors |        With LDA        |    93.26%   |     93.38%    |
|    Decision Tree    |      Without LDA       |    94.26%   |     94.64%    |
|    Random Forest    |      Without LDA       |    94.48%   |     95.58%    |


Random Forest achieved the highest test accuracy in this run. These results may differ from those in the original report because the preprocessing and model-selection procedures were changed.
Limitations
These results describe performance on a single test split and should not be interpreted as evidence of clinical effectiveness. Height and weight are included as input features and are closely related to obesity classification, which is important when interpreting model performance.
