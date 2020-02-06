# Interview-challenges

I share here some tests and challenges I accomplished during my internship and job interviews.

## Axa challenge :

The main purpose of the exercise is similar to a "Kaggle" competition: the goal is to build a predictive model from a data set. Here, it is a matter of estimating the annual net profit expected per customer on a motor insurance contract according to the customer's characteristics.

I have to use the data from labeled_dataset_axaggpdsc to build a predictive model and apply it to the scoring_dataset_axaggpdsc data in order to generate my predictions. The metric used to compare my results with the answers (true label) is the RMSE (Root Mean Squared Error) and the Gini index.

## Teads challenge : 

### Context : 

Every time a user browses an eligible web page, we have to decide whether to display an ad video or not.

If the video is displayed, then we will incur a fixed cost. However, we will benefit from a revenue only if the video is watched a minimum number of seconds by the user.

Thus, there are three possible cases:

- We do not display the video: Cost=0, Revenue=0, Profit=0
- We display a video that is not watched long enough: Cost>0, Revenue=0, Profit=-Cost
- We display a video that is watched long enough: Cost>0, Revenue>0, Profit=Revenue-Cost

In this challenge, I had to answer to the following questions: 

### Preliminary questions

- The margin being defined as (revenue - cost) / revenue, what is our global margin based on this log?
- Can you think of other interesting metrics to optimize?
- What are the most profitable Operating Systems?

### Machine learning questions

- How would you use this historical data to predict the event 'we benefit from a revenue' (ie revenue > 0) in a dataset where the revenue is not known yet?
- Compute the prediction accuracy of a well chosen algorithm and comment the results. Do not hesitate to describe your methodology.


## TinyClues challenge : 

The goal of this exercise is to fill missing data on a socio-demo user table ( a fake dataset of users provided by TinyClues ). In the interview, they expected a small note describing the methodology and the code in python used to fill the data

