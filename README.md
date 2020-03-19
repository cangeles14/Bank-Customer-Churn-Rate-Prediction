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

## Build Models

## Model Performance & Metrics

## Built With

* [Python](https://docs.python.org/3/) - The programming language used
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language
* [MySQL](https://www.mysql.com/) -  MySQL is an open-source relational database management system for SQL
* [Tableau](https://www.tableau.com/) - Popular Data visualization tool
* [MatPlotLib](https://matplotlib.org/contents.html) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms

## Authors

* **Christopher Angeles** - [cangeles14](https://github.com/cangeles14)

## Acknowledgments

* [Ironhack](https://www.ironhack.com/en/data-analytics) -  Data Analytics Bootcamp @ Ironhack Paris
