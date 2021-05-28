# Root Insurance Project -- Team Doodee
This is a Erdos data science Bootcamp project. We explore the dataset from insurance customers in a car insurance website. We'll try to evaluate different tyeps of customers and give different bid prices for different customers so as to show the insurance advertisement at a rank.

___
## Overview

- [Project description](#description)
- [Data Exploration](#data-exploration)
- [Machine learning approaches for probabilities](#machine-learning-method-for-probability)
- [Strategy for the bidding price](#strategy-for-the-bidding-price)
- [Authors](#authorship)

___
## Project Description

The goals of the project are:
- Understand the dataset: we have dataset with 10,000 "impressions" of ads, where the relation between customers and the insurance commercials hides.
- The marketing manager of the insurance company wants to understand how to bid differently for different customers so as to improve the ads performance.
- Bidding strategy: "optimize the cost per customer while having 4% customer rate over all ads shown". Bidding higher will make the ad to be shown higher in the ranking, but we do not know how the bidding change the ranking.
- Find some interesting relations for the website manager.

[Back to Overview](#overview)
___

## Data Exploration

### Customer Features

We'll first explore how the dataset looks like and how the customers are featured.
<p align="center">
<img src = "datatable.png" width="600" height="150"></img>
</p>

There are four main features for each customers: `Currently Insured`, `Number of Vehicles`, `Number of Drivers` and `Marital Status`, where each could serve as categorical feature with 36 possibilities in total. Note that there are miss data for `Currently Insured` as `unknown`. We could interpret this missing data as the middle of insured or not, but given our limited feature size, we treat it as an independent category.

Moreover, when removing duplicated features, there are only 35 unique combination of features.

Only two features are numerical (integers): `Number of Vehicles` and `Number of Drivers`, ranging from [1, 3] and [1,2], respectively. So there are not many numerical relations to explore.

Since the total number of samples (10,000) is much larger than the number of unique features of customers (35), we expect that:
- 1). For each type of costumer, there is a distribution of the ads ranks to be shown <img src="https://latex.codecogs.com/svg.image?P_i(r)" title="P_i(r)" />, where i is the costumer type and r=1,2,3,4,5 denote the ads rank; 
- 2). Given costumer features and ranks, there is also a probability for the costumer at a rank to click on the ads, <img src="https://latex.codecogs.com/svg.image?P_C" title="P_C" />;
- 3). Given the clicked costumer, there is also a conditional probability for the insurance to be sold, <img src="https://latex.codecogs.com/svg.image?P(S|C)" title="P(S|C)" />.

Thus, statistically, we're not dealing with a classification problem but a probability problem, where the accuracy of prediction is not very important. However, from the perspective of the insurance company, we do want to invest more on the types of customers who are prone to buy the policy after click (Only clicked ads need to be paid). By investing more (bidding more), we could improve the rank of the ads and thus increase the click probability.

[Back to Overview](#overview)
___
### Targets of the Problem

Notice that the probabilities are all conditioned, e.g. the probability of the 2nd-rank ads being clicked but not sold. Thus, in principle, we have a classification of 15 type (5 ranks, click and sold or not, no click).

<img src="https://latex.codecogs.com/svg.image?newtar&space;=&space;3(r-1)&space;&plus;&space;i,&space;\quad&space;i&space;=&space;0,&space;1,&space;2," title="newtar = 3(r-1) + i, \quad i = 0, 1, 2," />

where i=0,1,2 for sold, click but not sold and not click, respectively. However, the samples are limitted for some targets to stratify. So a more suitable one to assume that the sold rate is independent once the customer click, then we only need to have 10 type of new targets

<img src="https://latex.codecogs.com/svg.image?newtar&space;=&space;2(r-1)&space;&plus;&space;i,&space;\quad&space;i&space;=&space;0,&space;1," title="newtar = 2(r-1) + i, \quad i = 0, 1," />

##### However

How should we deal with the relation between bidding price and the ranking? We have no other info from the dataset, or we need to search for more supporting relations. But we could adopt a simple but very reasonable assumption:
`The overall buying probability of a particular type of clicked customers is independent of their ranking`
Afterall, the ranking is an evaluation of the market (other companies) to the customer (how much they want to earn this customer). Once the customer clicked, the probability of buying should be the internal feature of the customer. Thus, if we view a customer as a stock, ranking is more like the market price while the buying probability is the EPS (earning per share), measuring how profitable of the stock company.
<p align="center">
<img src = "rankDist.png" width="400" height="250"></img>
</p>
Different ranks have similar frequencies.

[Back to Overview](#overview)
___
## Machine Learning Approaches for Probabilities

As will be clear later, it is necessary for later steps in this project to know the probability that a customer will click on our insurance company advertisement and the probability that he/she will purchase the insurance company's policy, given the customer information and the ranking in which the advertisement is displayed. Although one can obtain these probabilities simply by counting the data, this is not an option for all types of customers in all ranks, as for some types of customers we do not have the data of those in all the ranks. For example, for uninsured, married customers with 3 vehicles and 2 drivers, we only have their data in ranks 3, 4 and 5. The need to fill in the blanks and predict the probabilities of click and purchase by customers of all types and ranks motivates us to construct a machine learning model to complete the task.

We explored various models with the aim of predicting the probabilities of click and purchase given customer features and ranks. The models include <a href="https://en.wikipedia.org/wiki/Random_forest">random forest,</a> <a href="https://en.wikipedia.org/wiki/Logistic_regression">logistic regression,</a> <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbor,</a> <a href="https://en.wikipedia.org/wiki/Support-vector_machine">support-vector machine</a> and <a href="https://en.wikipedia.org/wiki/Neural_network">neural network.</a> Each model is tuned using cross-validation on the training data set, while we set aside the test data set for final model selection. With the goal of predicting the probabilities rather than predicting the definite values of categorical targets, we utilize the metric of <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a> (KL divergence) to quantify the probability distribution difference between the training and testing set. 

The KL divergence for each model is shown in the table below for the predicted probabilities of click and purchase. 
<p align="center">
<img src = "KL_divergences.png" width="600" height="375"></img>
</p>




[Back to Overview](#overview)
___
## Strategy for the Bidding Price

Disclaimer: we used the probabilities obtained from <a href="https://github.com/dmlc/xgboost">XGBoost</a>, which is an efficient application of <a href="https://en.wikipedia.org/wiki/Gradient_boosting">Gradient boosting</a>, to perform the computation in this section.

Since the goal is to "optimize the cost per customer while having 4% customer rate over all ads shown". The simpliest intuition is to bid more on valuable customers. If we forget about the 4% constraint for a second, to decrease the cost per sold, we only need to consider the probability `P(sold|click)` for a customer as the company only need to pay when clicks happen

<img src="https://latex.codecogs.com/svg.image?P(sold|click)=\frac{P(\text{sold&space;and&space;click})}{P(click)}=\frac{P(sold)}{P(click)}" title="P(sold|click)=\frac{P(\text{sold and click})}{P(click)}=\frac{P(sold)}{P(click)}" />

Since current cost per customer is around 24.0 dollars per customer and the sold rate (sold/shown) is 7.83\% and average P(sold|click)=41.69\%, if we set the average as the baseline for 10 dollars and assume we invest linearly with the probability `P(sold|click)`, we'll have cost per customer even higher 24.19 dollars.


However, we give ads to all the samples. In reality, we should have some budget and stop showing more ads once the number of click with paid price reaches our budget. Current cost is $18,780 and we could set it as our budget and stop once reached though random sampling.


#### As we can see, from all these sampling trials, the simple linear strategy gives a bit higher cost per customer. We need to add more to current strategy. 

What if we take extreme cases? In the limit of infinite budget and customer samples, we should invest all the budget to the most valuable customer so as to obtain the best cost per customer. However, the limited budget and customer samples requires us invest on more customer with lower bound given by the 4% customer rate. Compared to previous strategy, the linear relation with the average `P(sold|click)` rate might be two slow. Thus, here we try a exponential function function 

<img src="https://latex.codecogs.com/svg.image?B=&space;1&plus;e^{C&space;(P-\bar{P})}" title="B= 1+e^{C (P-\bar{P})}" />

where the bidding price has minimum 1 dollar. The coefficient in the exponent <img src="https://latex.codecogs.com/svg.image?C\equiv&space;20" title="C\equiv 20" />. <img src="https://latex.codecogs.com/svg.image?\bar{P}" title="\bar{P}" /> is the average of <img src="https://latex.codecogs.com/svg.image?P(S|C)" title="P(S|C)" />.


[Back to Overview](#overview)
___
## Authorship

This project is a collaboration of physicists from Ohio State and Rutgers University: Zengle Huang, Xiaoyu Liu, Yossathorn (Josh) Tawabutr, Fangdi Wen, Angkun Wu, Yushan Yang.
