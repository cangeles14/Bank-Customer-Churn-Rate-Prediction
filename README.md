# Supervised Machine Learning Model @ Ironhack Paris

## Define The Problem

Churn Rate is the rate of customer attrition in a company, or the speed at which a customer leaves your company. It’s more expensive to acquire a new customer than it is to retain a current one, therefore a small increase in customer retention equates to a large decrease in company costs, greatly benefiting a company.

The objectives of this project is to:
- Reduce Account Closure
- Increase Customer / Service Ratio
- Reduce Bank Service Transfer


## Roadmap

- Gather data on customer behavior
- Use the data to predict and segment customers who are likely to churn
- Create models to demonstrate churn risk on company revenue
- Design and implement intervention strategy on segmented customers
- Repeat every fiscal year

## Understand the Data

The dataset used was provided by Kaggle.com

- [Churn Prediction of Bank Customers](https://www.kaggle.com/sonalidasgupta95/churn-prediction-of-bank-customers#Churn_Modelling.csv)

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/Dataset.png" width="75%" height="75%">

After cleaning and exploring the data, I focused on factors that may have the most impact on Churn rate

- Number of Products
- Age
- Credit Score
- Tenure
- Has a Credit Card

## Data Manipulation, Feature Engineering & Data Preprocessing

Before we can start the exploration and analysis of our data we need to make sure our data is showing what we want and when analyzed, we are extracting insightful information, and not misinformation.

To do this, I turned some numberical variables into categorical ones. This will allow me to visualize how this variable correlates to churn rate.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CategoricalData.png" width="30%" height="30%">

A good tip when working with both categorical and numerical data is to group your categorical and numerical variables into two distinct lists. This will help when you want to run analysis on just that data type and will greatly help down the pipeline when you need to process or manipulate your data.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CatNumCols.png" width="75%" height="75%">


## Exploratory Analysis

Lets have a look at some simple demographics of our dataset.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CustomerChurnPercentage.png" width="50%" height="50%">

It looks like our dataset has 80% customers that havent churned, and 20% that have. This is good to know if our dataset is biased and will allow us to know if we can extract some meaningful data insights from it. It will also let us know how we will proceed with a prediction model as we will want to train our model on all types of real life examples.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/AgeDistribution.png" width="50%" height="50%">

While looking at some of the data, we can see that the age distribution is not normally distributed, which is expected. This allows us to understand that we will need to normalize this data before we construct our prediction model.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/DataDemographic.png" width="50%" height="50%">

Here we see the number of customers that churned by region. France has a lower churn rate compared to Germany and Spain, with Germany having the most customers who churned.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CreditCard.png" width="50%" height="50%">

Here we can see the percent churn rate in correlation with the customers possession of a credit card service. We can see that having a credit card has no impact on weather or not a customer is more likely to churn.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/NumOfProducts.png" width="50%" height="50%">

Lastly, the most insightful visualization is the rate of churn on the number of services or products that are offered. We can see that customers that have 3+ products are churning, as we see very few customers that have 3 or 4 products are staying with the bank.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CorrelationMaxtrix.png" width="50%" height="50%">

Lastly, knowing the correlations between your data variables can give many insights from allowing you to dig deeper into one variable or to know if you have strong correlations to avoid. This will also help when building a prediction model as we do not want to overfit our model.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/VIFMultic.png" width="50%" height="50%">

Variance inflation factor (VIF) provides a measurement on how much variance increases due to collinearity. We want to keep our multicollinearity as low as possible for when we are building a prediction model. What multicollinearity does is that it will make the estimatation of our model highly unstable. What this means is that if there is a small or slight change in one variable, it can create a large change in another and in our predictions.

## Build Models

For this prediction model I used CatBoost or Categorical Boost. It is a machine learning algorithm that works by converting categorical values to numerical ones based on statistical combinations. You can read more about this algorithm in the link below.

To start, I created a train/text split on 1/3 of my dataset.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/CatBoostModel.png" width="50%" height="50%">

Catboost allows the use of categorical variables to not be dummied, as long as you declare which variables are, in fact, categorical.

One of catboost's more notable features is its ability to optimize one metric through running multiple iterations on the same model. Here I focused on optimizing recall score. I will go more into why I chose recall in the metric section, but recall tells us how accurate is our true positive prediction. It can be seen as answering the question "When it is actually the positive result, how often does it predict correctly?".

## Model Performance & Metrics

Testing, and optimizing a machine learning model is just as important as making one. If the model makes wrong predictions, is overfitted, underfitted, or simply has very low prediction accuracy, it can provide false information. Not good. 

One metric for testing if a prediction model is making accurate predictions is the confusion matrix.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/confusionmatrix_example.png" width="50%" height="50%">

Here we can see how well our prediction model works. By comparing our predicted results with the actual results in our dataset, we are able to test our model on how well its predicting the churn rate of a customer.


Let's take a closer look at our metrics and give it a real example. If we use our model to predict a new customer, and it predicts that this customer won't churn any time soon. This customer is happy. Now let's say that this prediction was incorrect, we can say that it was a False Negative prediction. Meaning, the customer was "negative" ( or not going to churn), but our prediction was false, meaning they DID in fact, churn. We can take this as the worse cast scenario. We could have prevented this customer from churning, but our model predicted incorrectly.

Since we know our "worst case scenario" metric, which is false negative (FN), we can optimize our model to reduce this error, even at the cost of lower accuracy or lower false positives. To do this we can set our CatBoost model to optimize the models Recall score, thus lowering our FN score as much as possible.

<img src="https://github.com/cangeles14/Bank-Customer-Churn-Rate-Prediction/blob/master/images/cfmatrix.png" width="50%" height="50%">

## Proceeding With a Strategy

Step 1: Explore the data to see what most effects customer churn rate and categorize what contributes to churn

Step 2: Use the prediction model to predict new or active customers likelihood of withdrawal or churn at the bank

Step 3: Target those most likely in efforts to reduce likelihood of withdrawal through proposed marketing strategies, including promotions or sales for select customers via email or phone calls

Step 4: Implement strategy, continue to collect data, and revisit model next fiscal year

## Built With

* [Python](https://docs.python.org/3/) - The programming language used
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language
* [MySQL](https://www.mysql.com/) -  MySQL is an open-source relational database management system for SQL
* [Tableau](https://www.tableau.com/) - Popular Data visualization tool
* [MatPlotLib](https://matplotlib.org/contents.html) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms
* [CatBoost](https://catboost.ai/) - CatBoost is a high-performance open source library for gradient boosting on decision trees

## Authors

* **Christopher Angeles** - [cangeles14](https://github.com/cangeles14)

## Acknowledgments

* [Ironhack](https://www.ironhack.com/en/data-analytics) -  Data Analytics Bootcamp @ Ironhack Paris
