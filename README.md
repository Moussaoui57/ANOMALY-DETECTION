# Anomaly-Detection

Airlines and travel agencies cooperate together to make their functions easier for the cus-
tomer to choose the best service. Thanks to smart investments in technology, airlines have
tremendously upgraded the digital customer experience. However, there still one big problem
that faces those travel companies which is fraud. 72 % of airlines programs has suffered from
fraud. Fraudsters benefit from the vulnerability of the travel agencies systems to make fake
transactions.
The International Air Transport Association (IATA) estimates that payment fraud costs
the airlines industry about $858 million per year [18]. Customers are also affected as they lose
time and money sorting out the effects of fraud. So that’s why Engineers worked to enhance
the security of transactions systems. But it is not enough, systems will remain vulnerable
against fraudsters.
Our project will present a solution to minimize those frauds, and prevent the fake tran-
sactions from happening. Using machine learning technology, we will be able to track down
those false online bookings. We consider those false transactions as anomalies.We use the
data set provided by the company, we will teach the Machine how to detect Anomaly.
In this context, the idea of our project was made :” Anomaly detection for Travel agencies
transactions using machine learning technology”.

# Conclusion 

Finaly, Ater evaluating models by comparing the performance of signification model results for
each classifiers by AUC and accuracy metrics. It can be seen that the most model is for
CART with the best score (97 %,98 %). Less then is for XGBOOST with (89 %,96 %) then
The third place with (87 %,95 %) is for SVM. The last three classifiers will remain relatively
decreases between (80 % AUC to 72 %) and (91.8 Accuracy to 91.2).
so we choose to build the CART classifier because it has the good performance then they
others.


# Requirments: 
- Create a virtualEnv Python Anaconda 3.7
- Install all libraries that exists in the import section
- Deploy model Alertifier before calling function in the test
- Using Test.py you can test transaction and train model if it's a new observation

