# Root Insurance Project -- Team Doodee
This is a Erdos data science Bootcamp project. We explore the dataset from insurance customers in a car insurance website. We'll try to evaluate different tyeps of customers and give different bid prices for different customers so as to show the insurance advertisement at a rank.

___
## Overview

- [Project description](#project-description)
- [Data Exploration](#data-exploration)
- [Machine learning approaches for probabilities](#machine-learning-approaches-for-probabilities)
- [Basic optimization method](#basic-optimization-method)
- [Binomial regression for rank distribution](#binomial-regression-for-rank-distribution)
- [Full optimization method](#full-optimization-method)
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


We'll first explore how the dataset looks like and how the customers are featured.
<p align="center">
<img src = "datatable.png" width="600" height="150"></img>
</p>

There are four main features for each customers: `Currently Insured`, `Number of Vehicles`, `Number of Drivers` and `Marital Status`, each of which is a categorical predictor. In total, there are (3 `Currently Insured`) x (3 `Number of Vehicles`) x (2 `Number of Drivers`) x (2 `Marital Status`) = 36 possible combinations, i.e. types. However, we have data from only 35 types of customers. 

Furthermore, the `Currently Insured` feature contains 3 categories: `Y`, `N` and `unknown`. There are two possible interpretations of the `unknown` group that result in different treatments. First, we can interpret that `unknown` customers in fact belong to a distinct group; for example, they could voluntarily opt to not provide the information whether or not they already had an insurance when they entered the vertical search website. As a result, we should treat the `Currently Insured` feature as containing 3 saparate categories and use one-hot encoding to express the feature in terms of two dummy variables. On the other hand, we can interpret that any `unknown` customer is either a `Y` or `N` customer with the information missing, and hence they should be treated using an imputation process. This leads to the encoding of `Y` to 1, `N` to 0, and "unknown" to 0.5, so that the `unknown` is in a sense the average between `Y` and `N`.

For each customer in the data set, the bidding price is $10. The ultimate goal of this project is to determine the bidding price <img src="https://latex.codecogs.com/svg.image?\mathbf{B}=(B_1,\ldots,B_{36})" title="B_vec" /> for the 36 customer types, such that the advertisement spending per policy purchase is minimized subject to the constraint that the overall purchase rate is at least 4%. 

Finally, the `rank` at which our company's advertisement appears in the customer's vertical search channel is given for each customer. The `rank` is treated differently in each step of our project. In particular, our [models for click and purchase probability](#machine-learning-method-for-probability) treat `rank` as a categorical predictor, while our [model for rank distribution](#binomial-fit) treats `rank` as the target. More detail about the list of predictors and target(s) for each of our models is given in the corresponding sections for the models.

Since the total number of samples (10,000) is much larger than the number of unique features of customers (35), we expect that:
- 1). For each type of customer, there is a distribution of the ranks, <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)" title="P_i(r;B_i)" />, where i is the customer type and r=1,2,3,4,5 denote the rank; 
- 2). Given customer types and ranks, there is a probability, <img src="https://latex.codecogs.com/svg.image?C_{i,r}" title="C_{i,r}" />, for the customer to click on the advertisement;
- 3). Given customer types and ranks, there is a probability, <img src="https://latex.codecogs.com/svg.image?S_{i,r}" title="S_{i,r}" />, for the customer to purchase the insurance policy;

Items 2) and 3) imply that, given a clicked customer, there is also a conditional probability, <img src="https://latex.codecogs.com/svg.image?P(S|C)" title="P(S|C)" />, for the customer to purchase the insurance policy.




[Back to Overview](#overview)
___
## Machine Learning Approaches for Probabilities

As will be clear in a [later section](#gradient-descent), it is necessary for later steps in this project to know the probability that a customer will click on our insurance company advertisement and the probability that he/she will purchase the insurance company's policy, given the customer information and the ranking in which the advertisement is displayed. In short, we will construct models with features: `Currently Insured`, `Number of Vehicles`, `Number of Drivers`, `Marital Status` and `rank`. Each model will separately predict: `Probability of click` and `Probability of purchase`.

Although one can obtain these probabilities simply by counting the data, this is not an option for all types of customers in all ranks, as for some types of customers we do not have the data of those in all the ranks. For example, for uninsured, married customers with 3 vehicles and 2 drivers, we only have their data in ranks 3, 4 and 5. The need to fill in the blanks and predict the probabilities of click and purchase by customers of all types and ranks motivates us to construct a machine learning model to complete the task.

We explored various models to predict the probabilities. For each model, we considered the [two possible ways](#data-exploration) to encode the `Currently Insured` feature. The models in consideration include <a href="https://en.wikipedia.org/wiki/Random_forest">random forest,</a> <a href="https://en.wikipedia.org/wiki/Logistic_regression">logistic regression,</a> <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">k-nearest neighbor,</a> <a href="https://en.wikipedia.org/wiki/Support-vector_machine">support-vector machine</a> and <a href="https://en.wikipedia.org/wiki/Neural_network">neural network.</a>  Each model is tuned using cross-validation on the training data set, while we set aside the test data set for final model selection. With the goal of predicting the probabilities rather than predicting the definite values of categorical targets, we utilize the metric of <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Leibler divergence</a> (KL divergence) to quantify the accuracy of each model's probability predictions. In particular, we compute the KL divergence for the predicted click and purchase probabilities compared to those deduced from the data, for each combination of features for which the actual customer data are available. Then, we sum the statistic over all such feature combinations.

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

Statistically, we're not dealing with a classification problem but a probability problem, where the accuracy of prediction is not very important. However, from the perspective of the insurance company, we do want to invest more on the types of customers who are prone to buy the policy after click (Only clicked ads need to be paid). By investing more (bidding more), we could improve the rank of the ads and thus increase the click probability.

### Targets of the Problem

Notice that the probabilities are all conditioned, e.g. the probability of the 2nd-rank ads being clicked but not sold. Thus, in principle, we have a classification of 15 type (5 ranks, click and sold or not, no click). However, the samples are limitted for some targets to stratify. So a more suitable one to assume that the sold rate is independent once the customer click, then we only need to have 10 type of new targets.

##### However

How should we deal with the relation between bidding price and the ranking? We have no other info from the dataset, or we need to search for more supporting relations. For a naive model, we could adopt a **key assumption**:

`The overall buying probability of a particular type of clicked customers is independent of their ranking`

Afterall, the ranking is an evaluation of the market (other companies) to the customer (how much they want to earn this customer). Once the customer clicked, the probability of buying should be the internal feature of the customer. Thus, if we view a customer as a stock, ranking is more like the market price while the buying probability is the EPS (earning per share), measuring how profitable of the stock company.
<p align="center">
<img src = "rankDist.png" width="500"></img>
</p>
Different ranks have similar frequencies.

### Naive Bidding Price Model

In this section, we used the probabilities obtained from <a href="https://github.com/dmlc/xgboost">XGBoost</a>, which is an efficient application of <a href="https://en.wikipedia.org/wiki/Gradient_boosting">Gradient boosting</a>, to perform the computation in this section.

Since the goal is to "optimize the cost per customer while having 4% customer rate over all ads shown". The simpliest intuition is to bid more on valuable customers. If we forget about the 4% constraint for a second, to decrease the cost per sold, we only need to consider the probability `P(sold|click)` for a customer as the company only need to pay when clicks happen

<img src="https://latex.codecogs.com/svg.image?P(\text{sold}|\text{click})=\frac{P(\text{sold&space;and&space;click})}{P(\text{click})}=\frac{P(\text{sold})}{P(\text{click})}" title="P(\text{sold}|\text{click})=\frac{P(\text{sold and click})}{P(\text{click})}=\frac{P(\text{sold})}{P(\text{click})}" />

Since current cost per customer is around 24.0 dollars per customer and the sold rate (sold/shown) is 7.83\% and average P(sold|click)=41.69\%, if we set the average as the baseline for 10 dollars and assume we invest linearly with the probability `P(sold|click)`, we'll have cost per customer even higher 24.19 dollars.


However, we give ads to all the samples. In reality, we should have some budget and stop showing more ads once the number of click with paid price reaches our budget. Current cost is $18,780 and we could set it as our budget and stop once reached though random sampling.

What if we take extreme cases? In the limit of infinite budget and customer samples, we should invest all the budget to the most valuable customer so as to obtain the best cost per customer. However, the limited budget and customer samples requires us invest on more customer with lower bound given by the 4% customer rate. Compared to previous strategy, the linear relation with the average `P(sold|click)` rate might be two slow. Thus, here we try a exponential function function 

<img src="https://latex.codecogs.com/svg.image?B=&space;1&plus;e^{C&space;(P-\bar{P})}" title="B= 1+e^{C (P-\bar{P})}" />

where the bidding price has minimum 1 dollar. If we constrain the bidding price range to [1,20], the coefficient in the exponent <img src="https://latex.codecogs.com/svg.image?C=&space;7.4" title="C= 7.4" />. <img src="https://latex.codecogs.com/svg.image?\bar{P}=0.2" title="\bar{P}=0.2" /> is an offset average.



[Back to Overview](#overview)
___
## Binomial Regression for Rank Distribution

Besides the [predictions of click and purchase probabilities](#machine-learning-method-for-probability), another ingredient necessary for our [full optimization method](#gradient-descent) for bidding price is the probability that the insurance company's advertisement is displayed in each ranking from 1 to 5, given the customer's background information and the amount our company bid for this customer. In this section, we construct a model with features: `Currently Insured`, `Number of Vehicles`, `Number of Drivers` and `Marital Status`, with the goal to predict the probability for each value of `rank`. 

If the advertisement is displayed in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />, then <img src="https://latex.codecogs.com/svg.image?r-1" title="r-1" /> out of the 4 other companies competing in the vertical search channel bid higher than our company. The main goal of this section is to determine the probability mass function (PMF) <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)" title="Pi" /> of <img src="https://latex.codecogs.com/svg.image?r" title="r" /> given the bidding amount <img src="https://latex.codecogs.com/svg.image?B_i" title="Bi" /> for a customer with combination of features (i.e. type) <img src="https://latex.codecogs.com/svg.image?i" title="i" />. To do so, we make the following assumptions for each <img src="https://latex.codecogs.com/svg.image?i" title="i" />.
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
<p>Here, <img src="https://latex.codecogs.com/svg.image?b_{i0}" title="bi0" /> implies the value of <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_i" title="pi_i_hat" /> when <img src="https://latex.codecogs.com/svg.image?B_i=10" title="Bi=10" />, while <img src="https://latex.codecogs.com/svg.image?a_i<0" title="a_i<0" /> measures how spread-out the other companies' bidding prices are. In particular, larger <img src="https://latex.codecogs.com/svg.image?\left|a_i\right|" title="abs_a_i" /> means other companies’ bidding prices are closer to one another (e.g $10.1, 10.2, $9.8...), and smaller <img src="https://latex.codecogs.com/svg.image?\left|a_i\right|" title="abs_a_i" /> means other companies’ bidding prices are more spread out (e.g $7, $9, $15...). More motivations for the choice of logit link function include:</p>
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

As for <img src="https://latex.codecogs.com/svg.image?a_i" title="a_i" />, we require the data at different bidding price in order to make a well-inform estimate of the parameter. Instead, we make another assumption that it is -0.5 for all customer types. This number implies that, if we decrease our bidding price by $1, the odd, <img src="https://latex.codecogs.com/svg.image?\frac{\hat{\pi}_i}{1-\hat{\pi}_i}" title="odd_hat" />, that another company bids higher than ours will increase by a multiple of <img src="https://latex.codecogs.com/svg.image?e^{0.5}\simeq%201.65" title="e^0.5" />. 






[Back to Overview](#overview)
___
## Full Optimization Method

<p>This section describes our full optimization method to determine the bidding price that minimizes the cost per customer's purchase, while keeping the overall purchase rate at 4% or higher. The optimization algorithm presented in this section takes as inputs the following quantitites deduced either from the data or from the models we developed in prior sections. </p>
<p>- <img src="https://latex.codecogs.com/svg.image?M_i" title="M_i" /> = (Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers)/(Total number of customers). This number relates to the customer population, and for the sake of this problem we deduce it from counting the data.</p>
<p>- <img src="https://latex.codecogs.com/svg.image?C_{i,r}" title="C_ir" /> = (Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" /> who click)/(Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />). This is the probability of click for customers of type <img src="https://latex.codecogs.com/svg.image?i" title="i" /> and rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />.  </p>
<p>- <img src="https://latex.codecogs.com/svg.image?S_{i,r}" title="S_ir" /> = (Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" /> who purchase)/(Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />). This is the probability of purchase for customers of type <img src="https://latex.codecogs.com/svg.image?i" title="i" /> and rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />. </p>
<p>- <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)" title="Pi" /> = (Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers in rank <img src="https://latex.codecogs.com/svg.image?r" title="r" />)/(Number of type-<img src="https://latex.codecogs.com/svg.image?i" title="i" /> customers). This is the rank distribution for customers of type <img src="https://latex.codecogs.com/svg.image?i" title="i" /> given the bidding price <img src="https://latex.codecogs.com/svg.image?B_i" title="B_i" />.  </p>

The second and third inputs were [previously predicted by our probability models,](#machine-learning-method-for-probability) while the last input was [determined by our binomial regression model.](#binomial-fit) Given these definitions, we define the following functions of the bidding vector.

<p>- <img src="https://latex.codecogs.com/svg.image?f(\mathbf{B})=\sum_{i,r}M_iC_{i,r}B_iP_i(r;B_i)" title="f(B)_def" /></p>
<p>- <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B})=\sum_{i,r}M_iS_{i,r}P_i(r;B_i)" title="g(B)_def" /></p>
<p>Physically, <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" /> is the total advertisement budget spent divided by the total number of sales, and <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B})" title="g(B)" /> is the overall proportion of customers who purchase. This allows us to express our constrained optimization problem can be phrased as</p>
<p>- Minimize: <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" /></p>
<p>- Constraint: <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B})\geq%200.04" title="g_constraint" /></p>

<p><a href="https://angms.science/doc/CVX/CVX_PGD.pdf">Projected gradient descent (PGD)</a> is a method used to find a (local) minimum of some function <img src="https://latex.codecogs.com/svg.image?h(\mathbf{x})" title="h(x)" /> subject to a constraint <img src="https://latex.codecogs.com/svg.image?\mathbf{x}\in\mathcal{C}\subset\mathbb{R}^n" title="x\in~C" />. For each step, PGD uses the usual method of gradient descent (GD) to determine the location <img src="https://latex.codecogs.com/svg.image?\mathbf{y}_{i+1}" title="\mathbf{y}_{i+1}" /> for the new step, but then it projects that location onto <img src="https://latex.codecogs.com/svg.image?\mathcal{C}" title="\mathcal{C}" />, that is, it finds 
<img src="https://latex.codecogs.com/svg.image?\mathbf{x}_{i+1}\in\mathcal{C}" title="\mathbf{x}_{i+1}\in\mathcal{C}" /> that minimizes <img src="https://latex.codecogs.com/svg.image?\left|\mathbf{x}_{i+1}%20-%20\mathbf{y}_{i+1}\right|^2" title="|x-y|^2" />.</p>

<p>In our case, the set <img src="https://latex.codecogs.com/svg.image?\mathcal{C}" title="\mathcal{C}" /> is implicitly defined. So, we will employ GD to determine the magnitude and direction of our new step, but whenever the new step takes the path to some 
<img src="https://latex.codecogs.com/svg.image?\mathbf{y}_{i+1}\notin\mathcal{C}" title="\mathbf{y}_{i+1}\notin\mathcal{C}" /> we discard the direction suggested by GD. Instead, we will add/subtract each cardinal component of <img src="https://latex.codecogs.com/svg.image?\mathbf{x}_i" title="\mathbf{x}_i" />, one-by-one, by the same step size and choose the cardinal direction with the largest decrease in <img src="https://latex.codecogs.com/svg.image?h(\mathbf{x})" title="h(x)" /> while the constraint still holds.</p>

<p>In the context of our problem, the algorithm goes as follows.</p>
<p>- Randomly assign a starting budget <img src="https://latex.codecogs.com/svg.image?\mathbf{B}^{(0)}" title="\mathbf{B}^{(0)}" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- If <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B}^{(0)})<0.04" title="g(\mathbf{B}^{(0)})<0.04" />, redo the random assignment.</p>
<p>- Repeat until a minimum, <img src="https://latex.codecogs.com/svg.image?\mathbf{B}" title="\mathbf{B}" />, is found.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Let <img src="https://latex.codecogs.com/svg.image?\mathbf{B}%27%20=%20\mathbf{B}^{(i)}%20-%20\ell\;\nabla_{\mathbf{B}^{(i)}}\left[\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}\right]" title="GD_step" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- If <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B}%27)%20\geq%200.04" title="g(\mathbf{B}%27)%20\geq%200.04" /> and <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20\frac{f(\mathbf{B}%27)}{g(\mathbf{B}%27)}%20\geq%20L_{\min}" title="\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20\frac{f(\mathbf{B}%27)}{g(\mathbf{B}%27)}%20\geq%20L_{\min}" />, take <img src="https://latex.codecogs.com/svg.image?\mathbf{B}^{(i+1)}=\mathbf{B}%27" title="\mathbf{B}^{(i+1)}=\mathbf{B}%27" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Else if <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20\frac{f(\mathbf{B}%27)}{g(\mathbf{B}%27)}%20\geq%20L_{\min}" title="\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20\frac{f(\mathbf{B}%27)}{g(\mathbf{B}%27)}%20\geq%20L_{\min}" />:</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Let <img src="https://latex.codecogs.com/svg.image?L^{(i+1)}%20=%20\ell\left|\nabla_{\mathbf{B}^{(i)}}\left[\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}\right]\right|" title="L^{(i+1)}%20=%20\ell\left|\nabla_{\mathbf{B}^{(i)}}\left[\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}\right]\right|" />. Also, let <img src="https://latex.codecogs.com/svg.image?j^*=-1" title="j^*=-1" />, <img src="https://latex.codecogs.com/svg.image?m^*=-1" title="m^*=-1" /> and <img src="https://latex.codecogs.com/svg.image?H^*=\infty" title="H^*=\infty" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Repeat for each <img src="https://latex.codecogs.com/svg.image?j\in\{1,\ldots,36\}" title="j\in\{1,\ldots,36\}" /> and <img src="https://latex.codecogs.com/svg.image?m\in\{0,1\}" title="m\in\{0,1\}" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Let <img src="https://latex.codecogs.com/svg.image?\mathbf{B}%27_{j,m}%20=%20\left(B^{(i)}_1,\ldots,B^{(i)}_j%20+%20(-1)^mL^{(i+1)},\ldots,B^{(i)}_{36}\right)" title="\mathbf{B}%27_{j,m}%20=%20\left(B^{(i)}_1,\ldots,B^{(i)}_j%20+%20(-1)^mL^{(i+1)},\ldots,B^{(i)}_{36}\right)" />. </p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- If <img src="https://latex.codecogs.com/svg.image?g(\mathbf{B}%27_{j,m})%20\geq%200.04" title="g(\mathbf{B}%27_{j,m})%20\geq%200.04" /> and <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B}%27_{j,m})}{g(\mathbf{B}%27_{j,m})}%20%3C%20\min\left\{\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20L_{\min},%20H^*\right\}" title="\frac{f(\mathbf{B}%27_{j,m})}{g(\mathbf{B}%27_{j,m})}%20%3C%20\min\left\{\frac{f(\mathbf{B}^{(i)})}{g(\mathbf{B}^{(i)})}%20-%20L_{\min},%20H^*\right\}" />, take <img src="https://latex.codecogs.com/svg.image?j^*=j" title="j^*=j" />, <img src="https://latex.codecogs.com/svg.image?m^*=m" title="m^*=m" /> and <img src="https://latex.codecogs.com/svg.image?H^*=\frac{f(\mathbf{B}'_{j,m})}{g(\mathbf{B}'_{j,m})}" title="H^*=\frac{f(\mathbf{B}'_{j,m})}{g(\mathbf{B}'_{j,m})}" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- If <img src="https://latex.codecogs.com/svg.image?j^*,m^*\geq%200" title="j^*,m^*\geq%200" />, take <img src="https://latex.codecogs.com/svg.image?\mathbf{B}^{(i+1)}=\mathbf{B}'_{j^*,m^*}" title="\mathbf{B}^{(i+1)}=\mathbf{B}'_{j^*,m^*}" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Else, return <img src="https://latex.codecogs.com/svg.image?\mathbf{B}=\mathbf{B}^{(i)}" title="\mathbf{B}=\mathbf{B}^{(i)}" />.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Else, return <img src="https://latex.codecogs.com/svg.image?\mathbf{B}=\mathbf{B}^{(i)}" title="\mathbf{B}=\mathbf{B}^{(i)}" />.</p>

<p>The first step to perform this algorithm is to compute the gradient of <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" />. Using the chain rules on the binomial distribution PMF, <img src="https://latex.codecogs.com/svg.image?P_i(r;B_i)={4%20\choose%20r-1}\pi_i^{r-1}(1-\pi_i)^{5-r}" title="P_i(B_i,r)={4%20\choose%20r-1}\pi_i^{r-1}(1-\pi_i)^{5-r}" />, and the scaled inverse link function, <img src="https://latex.codecogs.com/svg.image?\pi_i(B_i)=\frac{1}{1+\exp\left\{-\left[a_i(B_i-10)+b_{i0}\right]\right\}}" title="\pi_i(B_i)=\frac{1}{1+\exp\left\{-\left[a_i(B_i-10)+b_{i0}\right]\right\}}" />, we obtain the following component form of the gradient.</p>
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\frac{\partial}{\partial%20B_j}\frac{f(\mathbf{B})}{g(\mathbf{B})}=\frac{M_j}{g(\mathbf{B})}\sum_r\left\{\left[1+a_jB_j(r-1-4\pi_j)\right]C_{j,r}-\frac{f(\mathbf{B})}{g(\mathbf{B})}a_j(r-1-4\pi_j)S_{j,r}\right\}P_j(r;B_j)" title="gradient_f/g" />
</p>

<p>Since the function <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" /> we are minimizing in this problem has a rather complicated form, it likely contains many local minima. Hence, we should run the algorithm multiple  (HOW MANY?)</p>
  
<p>times and take <img src="https://latex.codecogs.com/svg.image?\mathbf{B}" title="\mathbf{B}" /> from the trial with smallest final <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" />, in order to obtain the bidding price that yields <img src="https://latex.codecogs.com/svg.image?\frac{f(\mathbf{B})}{g(\mathbf{B})}" title="f/g" /> as close as possible to its global minimum within the constrained region. We set the learning rate, <img src="https://latex.codecogs.com/svg.image?\ell" title="\ell" />, to (HOW MUCH?) </p>
  
<p>and the convergence limit, <img src="https://latex.codecogs.com/svg.image?L_{\min}" title="L_{\min}" />, to (HOW MUCH?)</p>

These choices of tuning parameters are carefully chosen with the tradeoff between accuracy and computational cost in mind.

(RESULTS AND INTERPRETATIONS GO HERE)





[Back to Overview](#overview)
___
## Authorship

This project is a collaboration of physicists from Ohio State and Rutgers University: Zengle Huang, Xiaoyu Liu, Yossathorn (Josh) Tawabutr, Fangdi Wen, Angkun Wu, Yushan Yang.
