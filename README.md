____

# Decision Trees - Classification.
____

![png]('download.png')

Decision tree algorithm is a supervised learning model used in predicting a dependent variable with a series of training variables.Decision trees algorithms can be used for classification and regression purposes.

In this particular project I am going to ilustrate its in classification of a discrete random variable.

## Some questions decision tree can answer.

1. Should a loan applicant be accepted or not?This is based on his/her history with and other measures.

1. Which drug is best for a particular patient.

1. Is a cancerous cell malignant ont benigh.

1. Is an email message spam or not?

and many more scenarios in real life.


## Understanding decision tree algorithm.

Decision trees are built using recursive partitioning to classify the data into two or more groups.

### Real life example.

Lets say we have data of patients who have gone through cancer screening over time.Based on the tests from screening exercises,the cells screened are classified as beningh and malignant.Based on this data a decision tree model can be built to predict these cases with the highest accuracy for future patients better than the doctors.
Decision tree alg splits the data variable by variable starting with the varible with the highest predictive power,less impurity and lower entropy.

The main aim of this method is to minimize impurity and each node.Impurity of nodes is calculated by **entropy of data** in the node.

**Entropy.**

Entropy is the amount of information dissorder or simply said is the amount of randomnes in the data or uncertainity.

The entyropy of a dataset depends on how nuch randomness is in the node.It should be noted that the lower the entropy the less uniform the distribution and the purer the node.If a sample is completely homogenous then the entropy is completely zero and if a sample is equaly devided it has an entropy of 1.

In refrence to the above data ,lets say a node has 7 malignant and 1 beningh while another node has 3 malignant and 5 benignh,the former is said to have a low entropy as compared to the latter.

This is how entropy is calculated mathematicaly:

$$Entropy = -p(Malignant)log(p(Malignant))-p(Benigh)log(p(Benigh))$$

The choice of the best tree depends on the node with the highest **Information Gain** after splitting.

**Information Gain**

This is the information that can increase the level of certainity after splitting.This is calculated as follows.

$$IG = Entropy\ of\ the\ tree\ before\ the\ split\ - weighted\ entropy\ after\ the\ split.$$

This process continues to build a basic decision tree.Below is a is a step by step process in python.
## Implimentation with Python.

For this project I will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. I will try to create a model that will help predict this.


**Data Features:**

Target Variable.
_'not.fully.paid'_ 1 if customer fully paid and 0 if otherwise

Explanatory Variables.

* _credit.policy:_ 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* _purpose:_ The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* _int.rate:_ The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* _installment:_ The monthly installments owed by the borrower if the loan is funded.
* _log.annual.inc:_ The natural log of the self-reported annual income of the borrower.
* _dti:_ The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* _fico:_ The FICO credit score of the borrower.
* _days.with.cr.line:_ The number of days the borrower has had a credit line.
* _revol.bal:_ The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* _revol.util:_ The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* _inq.last.6mths:_ The borrower's number of inquiries by creditors in the last 6 months.
* _delinq.2yrs:_ The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* _pub.rec:_ The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

[Click here](https://github.com/GeorgeOduor/decisiontrees/blob/master/DECISION%20TREES%20WITH%20PYTHON%20(1).ipynb) to see the python notebook associated with this project.
