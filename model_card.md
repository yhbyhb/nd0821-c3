# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
I (HanByul Yang) created the this model. it is Random Forest Classifier using the default parameters in scikit-learn 1.1.2

## Intended Use
This model should be used to predict the salary given input such as age, workclass, education and so on. 

## Training Data
The data was obtained from the UCI Machine Learning Repository (Census Income Data Set - https://archive.ics.uci.edu/ml/datasets/census+income). Original data has lots of space between comma, so, cleaned data (spaced-removed) are used for training and testing. 

## Evaluation Data
Source of data for evaluating is same as training data. 80% of data are used for training and 20% of data are used for testing.


## Metrics
The model evaluated with precision, recall and fbeta.
- precision : 0.76
- recall : 0.62
- fbeta : 0.68

## Ethical Considerations
The data contains lots of private infomation such as race, gender and education.

## Caveats and Recommendations
The data seems imbalanced on 'salary' label. The model may have bias to higher number of labels. To alleviate this, balancing samples with 'salary' label.
