#this project is from the kaggle competition :

Problem Description :
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we have  to complete the analysis of what sorts of people were likely to survive. In particular, we have to apply the tools of machine learning to predict which passengers survived the tragedy.

Data Description :
The data has been split into two groups:

training set (train.csv)
test set (test.csv)
The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. 

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. For each passenger in the test set, use the model trained to predict whether or not they survived the sinking of the Titanic.

Data Dictionary  :
Survival : 0:not surivved,1:survived
pclass: 1=1st class,2=2nd class,3=3rd class 
sex : male,female
Age : age in years
sibsp : no. of siblings / spouses aboard the Titanic
parch : no.of parents / children aboard the Titanic
ticket : ticket number
fare : fare of passengers
cabin : cabim number
embarked : Port of embarkation(C = Cherbourg, Q = Queenstown, S = Southampton)
