# Root Insurance Project -- Team Doodee
This is a Erdos data science Bootcamp project. We explore the dataset from insurance customers in a car insurance website. We'll try to evaluate different tyeps of customers and give different bid prices for different customers so as to show the insurance advertisement at a rank.

___
## Overview

- [Project description](#description)
- [Data Exploration](#data-exploration)
- [Machine learning approaches for probabilities](#machine-learning-method-for-probability)
- [Basic optimization method](#strategy-for-the-bidding-price)
- [Binomial regression for rank distribution](#binomial-fit)
- [Full optimization method](#gradient-descent)
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
<img src = "rankDist.png" width="500"></img>
</p>
Different ranks have similar frequencies.

[Back to Overview](#overview)
___
## Machine Learning Approaches for Probabilities

As will be clear in a [later section](#gradient-descent), it is necessary for later steps in this project to know the probability that a customer will click on our insurance company advertisement and the probability that he/she will purchase the insurance company's policy, given the customer information and the ranking in which the advertisement is displayed. In short, we will construct models with features: `Currently Insured`, `Number of Vehicles`, `Number of Drivers`, `Marital Status` and `rank`. Each model will separately predict: `Probability of click` and `Probability of purchase`.

Although one can obtain these probabilities simply by counting the data, this is not an option for all types of customers in all ranks, as for some types of customers we do not have the data of those in all the ranks. For example, for uninsured, married customers with 3 vehicles and 2 drivers, we only have their data in ranks 3, 4 and 5. The need to fill in the blanks and predict the probabilities of click and purchase by customers of all types and ranks motivates us to construct a machine learning model to complete the task.

We explored various models with the aim of predicting the probabilities of click and purchase given customer features and ranks. Before we jump into model construction, we explored two ways to treat the `Currently Insured` feature, which contains 3 categories: `Y`, `N` and `unknown`. The first possibility is to treat them as 3 saparate categories and use one-hot encoding to write it in terms of two dummy variables. This method follows the interpretation that `unknown` customers in fact belong to a distinct group; for example, they could voluntarily opt to not provide the information whether or not they already had an insurance when they entered the vertical search website. On the other hand, the second method is to encode `Y` to 1, `N` to 0, and "unknown" to 0.5. This method interprets that any `unknown` customer is either a `Y` or `N` customer with the information missing, and hence they should be treated using an imputation process. When we perform model selection, not only do we train different types of machine learning models, but we also consider these two options to encode the `Currently Insured` feature.

Now, as we select the models to predict the probabilities, the models in consideration include <a href="https://en.wikipedia.org/wiki/Random_forest">random forest,</a> <a href="https://en.wikipedia.org/wiki/Logistic_regression">logistic regression,</a> <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbor,</a> <a href="https://en.wikipedia.org/wiki/Support-vector_machine">support-vector machine</a> and <a href="https://en.wikipedia.org/wiki/Neural_network">neural network.</a>  Each model is tuned using cross-validation on the training data set, while we set aside the test data set for final model selection. With the goal of predicting the probabilities rather than predicting the definite values of categorical targets, we utilize the metric of <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a> (KL divergence) to quantify the accuracy of each model's probability predictions. In particular, we compute the KL divergence for the predicted click and purchase probabilities compared to those deduced from the data, for each combination of features for which the actual customer data are available. Then, we sum the statistic over all such feature combinations.

The KL divergence for each model is shown in the table below for the predicted probabilities of click and purchase. 
<p align="center">
<img src = "KL_divergences.png" width="600" height="375"></img>
</p>
From the table, we see that the random forest and logistic regression for both treatments of the `Currently Insured` feature have the lowest KL divergence, which implies better matches with the probabilities deduced from the data. This is true for the predictions of both probabilities. Since the fifth-best model, neural network, has total KL-divergence of more than 4 times the fourth-best model for the click rate, while the top four models have similar KL divergences, we decide to proceed with the average probabilities predicted by the two random forest and two logistic regression models. As for the purchase rate, the top-five rankings are similar, but now neural network performs almost as good as the four best models. However, we still decide not to include neural network into our average model for purchase rate because it is a more flexible model, which should perform significantly better in order to justifies its inclusion by the principle of parsimony. Hence, we use the average of the four models written in blue to predict both probabilities. The last row of the table shows the KL divergence for the average model. 
<br />
<br />In order to illustrate how well each model fits the actual probabilities, we show their residual plots below. In each plot, the horizontal axis corresponds the various combinations of values of categorical predictors. The vertical axis is the probability deduced from counting the raw data minus that predicted by the model. Note that only the combinations of predictors with available data are plotted.
<p align="center">
<img src = "Pclick_residuals.png" width="800"></img>
</p>
<p align="center">
<img src = "Psold_residuals.png" width="800"></img>
</p>




[Back to Overview](#overview)
___
## Basic Optimization Method

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
## Binomial Regression for Rank Distribution

Besides the [predictions of click and purchase probabilities](#machine-learning-method-for-probability), another ingredient necessary for our [full optimization method](#gradient-descent) for bidding price is the probability that the insurance company's advertisement is displayed in each ranking from 1 to 5, given the customer's background information and the amount our company bid for this customer. If the advertisement is displayed in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />, then <img src="https://latex.codecogs.com/svg.image?r-1" title="r-1" /> out of the 4 other companies competing in the vertical search channel bid higher than our company. The main goal of this section is to determine the probability mass function (PMF) <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)" title="Pi" /> of <img src="https://latex.codecogs.com/svg.image?r" title="r" /> given the bidding amount <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" /> for a customer with combination of features (i.e. type) <img src="https://latex.codecogs.com/svg.image?i" title="i" />. To do so, we make the following assumptions for each <img src="https://latex.codecogs.com/svg.image?i" title="i" />.
- For a fixed bidding price <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" />, each of the 4 other companies has the same probability, <img src="https://latex.codecogs.com/svg.image?\pi_i(B_i)" title="pi_i" />, of bidding higher than our company.
- Each of the 4 companies bid independently.

With the two assumptions above, we deduce that <img src="https://latex.codecogs.com/svg.image?P_i(r;%20B_i)" title="Pi" /> follows a binomial distribution in <img src="https://latex.codecogs.com/svg.image?r-1" title="r-1" /> with <img src="https://latex.codecogs.com/svg.image?n=4" title="n=4" /> trials and probability of success <img src="https://latex.codecogs.com/svg.image?\pi_i(B_i)" title="pi_i" />, depending on the bidding price <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" />: 
<p align="center">
<img src="https://latex.codecogs.com/svg.image?P_i(r;%20B_i)={4\choose%20r-1}\left[\pi_i(B_i)\right]^{r-1}\left[1-\pi_i(B_i)\right]^{5-r}" title="binom_pmf" />
</p>
Hence, this is a <a href="https://en.wikipedia.org/wiki/Binomial_regression">binomial regression problem.</a> 
<br />
<br /><p>To relate for each <img src="https://latex.codecogs.com/svg.image?i" title="i" /> the estimated probability of success <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" /> with <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" />, we employ the <a href="https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function">canonical link function,</a> that is, the logit link function. Explicitly, we have</p>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i=\frac{1}{1+e^{-\left[a_i(B_i-10)+b_{i0}\right]}}" title="logistic" />
</p>
<p>Here, <img src="https://latex.codecogs.com/svg.image?b_{i0}" title="bi0" /> implies the amount of bidding <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" /> at which <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i=\frac{1}{2}" title="pi_i_hat=1/2" />, while <img src="https://latex.codecogs.com/svg.image?a_i<0" title="a_i<0" /> measures how spread-out the other companies' bidding prices are. In particular, larger <img src="https://latex.codecogs.com/svg.image?\left|a_i\right|" title="abs_a_i" /> means other companies’ bidding prices are closer to one another (e.g $10.1, 10.2, $9.8...), and smaller <img src="https://latex.codecogs.com/svg.image?\left|a_i\right|" title="abs_a_i" /> means other companies’ bidding prices are more spread out (e.g $7, $9, $15...). More motivations for the choice of logit link function include:</p>
<p>- The values of <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" />, which is a probability, gets mapped onto the whole real line, which is the range for linear functions of <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" />.</p>
<p>- A change in bidding price has more impact on <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" /> and hence <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)" title="Pi" /> if <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" /> is in the region where <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" /> is close to 0.5, e.g. an increase from $8 to $10. This should be contrasted to e.g. an increase from $1000 to $1002, whose impact on <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" /> is minimal. </p>

<p>In our approach, we fit the ranking distribution from the data, in which the bidding price is $10 for all customers, with the binomial PMF to estimate <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i(B_i=\$10)" title="pi_i(Bi=10)" /> for each customer type <img src="https://latex.codecogs.com/svg.image?i" title="i" />. Then, we use the results to compute <img src="https://latex.codecogs.com/svg.image?b_{i0}" title="bi0" />'s using</p>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?b_{i0}=\ln\left(\frac{\hat{\pi}_i}{1-\hat{\pi}_i}\right)\bigg|_{B_i=\$10}" title="bi0_eqn" />
</p>
This method results in the following fits for various customer types. <br />
<br />
<p align="center">
<img src = "Binomial_regression_plots.png" width="900"></img>
</p>
<p>The plots above display the probabilities of being in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" /> for each customer type. The orange markers represent the actual probabilities deduced from counting the data, while the blue markers represent those resulted from the fits.</p>

(PERHAPS PROVIDE A TABLE OF b_i0's HERE. ALSO, IS IT POSSIBLE TO GIVE SOME NUMBERS THAT TELL THE READERS HOW WELL THE BINOMIAL FITS ARE?)

As for <img src="https://latex.codecogs.com/svg.image?a_i" title="a_i" />, we require the data at different bidding price in order to make a well-inform estimate of the parameter. Instead, we make another assumption that it is -ln(7) (HOW MUCH??? PLEASE MODIFY TO THE NUMBER YOU USED.) 

for all customer types. This number implies that, if we decrease our bidding price by $1, the odd, <img src="https://latex.codecogs.com/svg.image?\frac{\hat{\pi}_i}{1-\hat{\pi}_i}" title="odd_hat" />, that another company bids higher than ours will increase by a multiple of 7. (MODIFY THE NUMBER PLEASE)






[Back to Overview](#overview)
___
## Full Optimization Method

Will do. -JT-






[Back to Overview](#overview)
___
## Authorship

This project is a collaboration of physicists from Ohio State and Rutgers University: Zengle Huang, Xiaoyu Liu, Yossathorn (Josh) Tawabutr, Fangdi Wen, Angkun Wu, Yushan Yang.
