# Root Insurance Project -- Team Doodee
This is a Erdos data science Bootcamp project. We explore the dataset from insurance customers in a car insurance website. We'll try to evaluate different tyeps of customers and give different bid prices for different customers so as to show the insurance advertisement at a rank.

___
## Overview

- [Project Description](#description)
- [Customer features](#features)
- [Target of the problem](#targets)
- [Machine Learning approaches for probabilities](#machine-learning-method-for-probability)
- [Strategy for the bidding price](#strategy-for-the-bidding-price)
- [Authors](#authorship)

___
## Description

The goals of the project are:
- Understand the dataset: we have dataset with 10,000 "impressions" of ads, where the relation between customers and the insurance commercials hides.
- The marketing manager of the insurance company wants to understand how to bid differently for different customers so as to improve the ads performance.
- Bidding strategy: "optimize the cost per customer while having 4% customer rate over all ads shown". Bidding higher will make the ad to be shown higher in the ranking, but we do not know how the bidding change the ranking.
- Find some interesting relations for the website manager.

[Back to Overview](#overview)
___


## Features

We'll first explore how the dataset looks like and how the customers are featured.

<img src = "datatable.png" style="width:30%"></img>

There are four main features for each customers: `Currently Insured`, `Number of Vehicles`, `Number of Drivers` and `Marital Status`, where each could serve as categorical feature with 36 possibilities in total. Note that there are miss data for `Currently Insured` as `unknown`. We could interpret this missing data as the middle of insured or not, but given our limited feature size, we treat it as an independent category.

Moreover, when removing duplicated features, there are only 35 unique combination of features.

Only two features are numerical (integers): `Number of Vehicles` and `Number of Drivers`, ranging from [1, 3] and [1,2], respectively. So there are not many numerical relations to explore.

Since the total number of samples (10,000) is much larger than the number of unique features of customers (35), we expect that:
- 1). For each type of costumer, there is a distribution of the ads ranks to be shown P_i(r), where i is the costumer type and r=1,2,3,4,5 denote the ads rank; 
- 2). Given costumer features and ranks, there is also a probability for the costumer at a rank to click on the ads, P_C;
- 3). Given the clicked costumer, there is also a conditional probability for the insurance to be sold, P(S|C).

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P_i(r)" />

Thus, statistically, we're not dealing with a classification problem but a probability problem, where the accuracy of prediction is not very important. However, from the perspective of the insurance company, we do want to invest more on the types of customers who are prone to buy the policy after click (Only clicked ads need to be paid). By investing more (bidding more), we could improve the rank of the ads and thus increase the click probability.

[Back to Overview](#overview)
___
## Targets

Notice that we the probabilities are all conditioned, e.g. the probability of rank 2th and then click but not sold. Thus, in principle, we have a classification of 15 type (5 ranks, click and sold or not, no click).
$$ newtar = 3(r-1) + i, \quad i = 0, 1, 2,$$
where $i=0,1,2$ for sold, click but not sold and not click, respectively. However, the samples are limitted for some targets to stratify. So a more suitable one to assume that the sold rate is independent once the customer click, then we only need to have 10 type of new targets
$$ newtar = 2(r-1) + i, \quad i = 0, 1,$$

#### However

How should we deal with the relation between bidding price and the ranking? We have no other info from the dataset, or we need to search for more supporting relations. But we could adopt a simple but very reasonable assumption:
`The overall buying probability of a particular type of clicked customers is independent of their ranking`
Afterall, the ranking is an evaluation of the market (other companies) to the customer (how much they want to earn this customer). Once the customer clicked, the probability of buying should be the internal feature of the customer. Thus, if we view a customer as a stock, ranking is more like the market price while the buying probability is the EPS (earning per share), measuring how profitable of the stock company.

[Back to Overview](#overview)
___
## Machine Learning method for probability

Despite we're working on a probability problem, we could utilize powerful machine learning method with cross validation to obtain the probability conditioned on observed data. We'll explore a varity of classification methods and use the metric of <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a> to quantify the distribution difference between the training and testing set. Specifically, we'll show the probabilities obtained from <a href="https://github.com/dmlc/xgboost">XGBoost</a>, which is an efficient application of <a href="https://en.wikipedia.org/wiki/Gradient_boosting">Gradient boosting</a>.

[Back to Overview](#overview)
___
## Strategy for the bidding price

Since the goal is to "optimize the cost per customer while having 4% customer rate over all ads shown". The simpliest intuition is to bid more on valuable customers. If we forget about the $4%$ constraint for a second, to decrease the cost per sold, we only need to consider the probability `P(sold|click)` for a customer as the company only need to pay when clicks happen
$$P(sold|click)=\frac{P(\text{sold and click})}{P(click)}=\frac{P(sold)}{P(click)}$$

Since current cost per customer is around 24.0 dollars per customer and the sold rate (sold/shown) is 7.83\% and average P(sold|click)=41.69\%, if we set the average as the baseline for 10 dollars and assume we invest linearly with the probability `P(sold|click)`, we'll have cost per customer even higher 24.19 dollars.


However, we give ads to all the samples. In reality, we should have some budget and stop showing more ads once the number of click with paid price reaches our budget. Current cost is $18,780 and we could set it as our budget and stop once reached though random sampling.


#### As we can see, from all these sampling trials, the simple linear strategy gives a bit higher cost per customer. We need to add more to current strategy. 

What if we take extreme cases? In the limit of infinite budget and customer samples, we should invest all the budget to the most valuable customer so as to obtain the best cost per customer. However, the limited budget and customer samples requires us invest on more customer with lower bound given by the 4% customer rate. Compared to previous strategy, the linear relation with the average `P(sold|click)` rate might be two slow. Thus, here we try a exponential function function 
$$ B= 1+e^{-C (P-\bar{P})}$$
where the bidding price has minimum 1 dollar. The coefficient in the exponent $C\equiv 20$. $\bar{P}$ is the average of $P(S|C)$.


[Back to Overview](#overview)
___
## Authorship

This project is a collaboration of physicists from Ohio State and Rutgers University: Zengle Huang, Xiaoyu Liu, Yossathorn (Josh) Tawabutr, Fangdi Wen, Angkun Wu, Yushan Yang.
