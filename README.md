In this demo I try to guess if someone's income is below $50k or above $50k by training non-linear ML models on some data. I load a train and test TSV and train Logistic regression, random forest, gradiant boosted trees and a neural network (PyTorch) on the data. Data Visualizations are done in Tableau and data processing is done in PySpark.

#Analysis of Training data

##Income (Target Variable)

![Income](https://github.com/bpkucsb/pyspark_ml/assets/13769127/4c8d3d1b-343c-47d9-873b-bedbcc157578)

We see that about 25% of our records have an income below $50k. This is a well enough balanced dataset that we will not do any downsampling.

##Education

![Education vs Education-Num](https://github.com/bpkucsb/pyspark_ml/assets/13769127/7498cdb2-6c2e-4a41-9219-6254522f5b0b)
![Education](https://github.com/bpkucsb/pyspark_ml/assets/13769127/c8258031-e6cc-4524-b4ae-347708abcab3)


We see that education correlates exactly with the education-num column. Therefore we will only use one of the two, education-num in this case and keep it as an integer.

##Age

![Age](https://github.com/bpkucsb/pyspark_ml/assets/13769127/49fda125-93da-460a-ba0b-310cecf89db1)


We see that age has a long tail. In order for the ML models to not overfit on small pockets of data we will cap the age at 70 and bin it by 5 years intervals. Also no one between the age of 15-20 has an income over $50k so we could probably just force such records to be below $50k.

##Native-Country

![Native-Country](https://github.com/bpkucsb/pyspark_ml/assets/13769127/f4116de3-f528-40c3-befc-050570e08714)

Almost all the data has US records. Therefore in order to not overfit on small pockets of data we will build a binary variable US/non-US

##Workclass and Occupation

![Occupation](https://github.com/bpkucsb/pyspark_ml/assets/13769127/0f9f80b4-431e-4817-89d8-3f9c7d3d4a78)
![Workclass](https://github.com/bpkucsb/pyspark_ml/assets/13769127/355eb117-f96c-4ced-aa2b-3cc42a3063f7)

We will one hot encode both of these

![Occupation vs Workclass](https://github.com/bpkucsb/pyspark_ml/assets/13769127/30cffcfb-eaf9-4b78-ad7d-73f34092697e)
![Workclass vs Occupation](https://github.com/bpkucsb/pyspark_ml/assets/13769127/13a504ad-59f9-49fa-83c8-c89b573468c8)


There is not really a strong correlation between both of these fields so we will need to use both these fields as features.

##Relationship

![Relationship](https://github.com/bpkucsb/pyspark_ml/assets/13769127/44294f34-04a7-4332-b349-d451b199da6f)

Each category is well populated so we will use one hot encoding

##Final Weight

![Final Weight](https://github.com/bpkucsb/pyspark_ml/assets/13769127/71897ce4-e347-4713-a652-73f8331441c6)

In order to avoid overfitting we will bin by 200K intervals and cap this at 400K

##Hours-Per-Week

![Hours Per Week](https://github.com/bpkucsb/pyspark_ml/assets/13769127/0456fe24-4b3d-40a8-b974-ba47fc31122e)

The majority of records have 40 hour workweeks so we will build an ordinal variable -1, 0, 1 for <40 hours, 40 hours and >40 hours work week.

##Capital Gain/Loss

![Capital Gain](https://github.com/bpkucsb/pyspark_ml/assets/13769127/d9edd5bb-aa7b-46b2-aa99-d6cd0900ae6e)
![Capital Loss](https://github.com/bpkucsb/pyspark_ml/assets/13769127/5b29fa03-d856-463d-a4bd-09dd3fefa81d)

Because of the long tail we will combine these two into a capital loss/gain ordinal variable where if there is a loss the variable is -1, no loss or gain then 0 and a gain then +1.

##Gender and Race

![Gender](https://github.com/bpkucsb/pyspark_ml/assets/13769127/9d348ad1-b717-4c86-be12-18282584a002)
![Race](https://github.com/bpkucsb/pyspark_ml/assets/13769127/952f418c-2613-4d61-8975-ac61d003df71)

These variables can be sensitive from a regulatory standpoint. One strategy to make sure that our ML models are not discriminatory is to train models with and without these variables and compare the performance. If the performance is unchanged we can discard them for safer variables.

#Model training and evaluation

We use grid search in order to optimize the training parameters for the logistic regression, random forest and gradient boosted forest in PySpark. The ROCs of the best models are shown below

##Logistic Regression

<img width="626" alt="Screen Shot 2024-03-05 at 6 50 52 PM" src="https://github.com/bpkucsb/pyspark_ml/assets/13769127/9ac84522-f303-44a9-9569-675f70f6535c">

Looks good, model has similar performance on the testing and training datasets, which means we are not overfitting.

##Random Forest

<img width="620" alt="Screen Shot 2024-03-05 at 6 51 02 PM" src="https://github.com/bpkucsb/pyspark_ml/assets/13769127/dfca8198-95dd-490b-8b8a-c20ba145e112">

Weird kink at low TPR. Needs to be investigated in the data to see what records are causing this unexpected behavior.

##Gradient Boosted Trees

<img width="626" alt="Screen Shot 2024-03-05 at 6 51 16 PM" src="https://github.com/bpkucsb/pyspark_ml/assets/13769127/c61e361e-fc97-4621-b8b5-9b780748f27c">

Looks good, however some evidence of overfitting because the training curve is above the test curve. Note that this is not XGBoost or LightGBM, which should perform better. Will update with these ML techniques in the future

##Neural Networks

<img width="616" alt="Screen Shot 2024-03-05 at 6 51 37 PM" src="https://github.com/bpkucsb/pyspark_ml/assets/13769127/5f4bb986-3186-4ecd-bd24-230ba4f34d9a">

The neural networks give similar performance as the other models.

#Conclusion

The logisitic regression, gradient boosted trees and neural network all have similar ROC performance. Two limitations here are the size of the dataset and also how we binned and capped some of our input variables in order to avoid overfitting on small pockets of outliers. If we adjust some of this data preprocessing we could improve the ML models performance. We will also try training this data on XGBoost (H2O) and lightGBM in the future.

