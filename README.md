# Machine_Learning_Concepts_without_libraries
An interesting repo where we see basic machine learning concepts without using any pre-built libraries like scikit-learn etc. Soon I will be publishing a **Medium** article on this.

There are many machine learning concepts that we use daily in our neural networks without knowing their actual implementation. These concepts are being used in different domains of AI e.g. in NLP, Computer Vision, Time Series, Spam detection, etc. Some of the below concepts/techniques that we will be looking into detail **without using any in-built libraries**:
* Bag of Words (CountVectorizer from scikit-learn)
* Tf-idf
* Word embeddings
* Positional Encoding
* Ordinal Encoding
* PCA
* KNN
  - An interesting use case where we want to do image classification using KNN but no library
* k-means clustering
* Nearest centroid classifier
* Logistic Regression
* Linear Regression
* Hard margin / Soft margin SVMs
  - Getting the support vectors without using scikit learn or LIBSVM
  - SVM with weighted samples
* SVM with Gaussian kernel, using simple NumPy
* Decision Trees
* Dropout
* Linear discriminant analysis
* Image geometric transformations
* L1/L2 regularization
* Feature Selection (No RFE from scikit-learn or PCA)
* A predictive modeling without the use of the gradient descent algorithm for backpropagation
* Isolation Forest without and with using PyTorch & PyOD
* Naive Bayes
* Perceptron
* Adaline
* Hierarchical Clustering
* Hidden Markov Model (HMM)
* Restricted Boltzmann Machine (RBM)
* SOM

Building machine learning algorithms from scratch is a valuable exercise that deepens understanding and hones programming skills. If you're looking to implement various ML concepts in pure Python, here are some questions or challenges you might consider:
1. **Foundational Algorithms**:
   - How can you implement linear regression using the least squares method?
   - How would you code logistic regression with gradient descent?
   - Can you build a decision tree classifier using the ID3 or CART algorithm?
   - How would you implement the k-means clustering algorithm?

2. **Nearest Neighbors**:
   - How can you compute the distance between two data points in an n-dimensional space?
   - How would you find the 'k' nearest data points to a given point?
   - How would you handle ties when classifying a new data point based on its neighbors?

3. **Support Vector Machines (SVM)**:
   - How can you find the maximum-margin hyperplane in a 2-dimensional space?
   - How would you adapt this for higher dimensions?
   - Can you implement the kernel trick for non-linearly separable data?

4. **Principal Component Analysis (PCA)**:
   - How can you compute the covariance matrix of a dataset?
   - How would you find the eigenvectors and eigenvalues of this matrix?
   - How would you project the original dataset into a reduced-dimensional space using the top 'k' eigenvectors?

5. **Regularization**:
   - How can you add L1 regularization (Lasso) to a linear regression model?
   - How would you implement L2 regularization (Ridge)?

6. **Neural Networks**:
   - Can you code a single-layer perceptron for binary classification?
   - How would you implement a multi-layer perceptron with a backpropagation algorithm?
   - How would you introduce dropout as a regularization method in neural networks?

7. **Naive Bayes**:
   - How can you compute probabilities for a Gaussian Naive Bayes classifier?
   - How would you handle underflow issues caused by multiplying many small probabilities?

8. **Ensemble Methods**:
   - How can you combine multiple decision trees to create a random forest?
   - Can you implement boosting methods like AdaBoost?

9. **Optimization and Training**:
   - How would you implement batch gradient descent versus stochastic gradient descent?
   - Can you code an adaptive learning rate method like AdaGrad or RMSprop?

10. **Evaluation Metrics**:
   - How can you compute the F1-score of a binary classifier without using any libraries?
   - Can you calculate ROC and AUC for model evaluation?

11. **Utility Functions 1**:
   - How would you normalize or standardize a dataset without relying on external libraries?
   - Can you code a function to split a dataset into training and testing sets?

12. **Advanced Challenges**:
   - How would you handle categorical variables in a dataset?
   - Can you implement anomaly detection using the one-class SVM approach?
   - How would you code collaborative filtering for recommendation systems?

13. **Optimization**:
   - Implement simulated annealing to solve optimization problems.
   - How would you code a genetic algorithm to evolve solutions to a given problem?

14. **Recommendation Systems**:
   - How would you implement user-based and item-based collaborative filtering from scratch?
   - Can you design a matrix factorization approach, like Singular Value Decomposition (SVD), for recommendations?

15. **Natural Language Processing**:
   - Implement a basic bag-of-words model to convert text into vectors.
   - How would you code the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme for text data?
   - Can you write a simple N-gram language model?

16. **Time Series**:
   - How would you implement the moving average or exponential smoothing for time series forecasting?
   - Can you code an autoregressive (AR) or moving average (MA) model?

17. **Reinforcement Learning**:
   - Implement a Q-learning algorithm to solve a simple maze problem.
   - How would you approach the multi-armed bandit problem using the epsilon-greedy strategy?

18. **Anomaly Detection**:
   - How would you use a Gaussian distribution to detect anomalies in a univariate dataset?
   - Can you implement the Isolation Forest algorithm to find anomalies in a dataset?

19. **Dimensionality Reduction**:
   - Implement the t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm.
   - How would you approach building an autoencoder for dimensionality reduction?

20. **Graph Algorithms**:
   - How would you code a graph-based clustering method like the Markov Clustering Algorithm (MCL)?
   - Implement a graph traversal algorithm, like Depth First Search (DFS) or Breadth First Search (BFS).

21. **Bayesian Methods**:
   - Can you implement a basic Bayesian network?
   - How would you approach the Monty Hall problem using Bayesian inference?

22. **Instance-based Algorithms**:
   - Implement the Learning Vector Quantization (LVQ) algorithm.
   - How would you code the Self-Organizing Map (SOM) algorithm?

23. **Feature Engineering**:
   - How would you handle missing data without using any libraries?
   - Can you code polynomial features for a given dataset?

24. **Advanced Neural Networks**:
   - How would you implement a recurrent neural network (RNN) from scratch?
   - Can you code a basic Long Short-Term Memory (LSTM) cell?

25. **Utility Functions 2**:
   - Implement k-fold cross-validation to evaluate model performance.
   - How would you code the bootstrap sampling method?

26. **Others**:
   - Implement a DBSCAN clustering algorithm.
   - How would you handle imbalanced datasets? Can you code the Synthetic Minority Over-sampling Technique (SMOTE)?

27. **Preprocessing**:
   - How would you manually standardize (z-score normalize) a feature column?
   - Can you implement Min-Max scaling from scratch?
   - How would you encode categorical features using one-hot encoding without using libraries?

28. **Feature Selection**:
   - Implement mutual information or chi-square test to rank features based on their importance.
   - How would you manually implement Recursive Feature Elimination (RFE)?

29. **Linear Models**:
   - Can you implement stochastic gradient descent (SGD) for training linear models?
   - How would you manually code Ridge and Lasso regression?

30. **Tree-based Models**:
   - Implement a function to compute the Gini impurity or entropy of a dataset.
   - Can you write a random forest algorithm without using the decision tree as a black-box?

31. **Neural Network Models**:
   - Can you manually code the forward and backward pass of a simple neural network?
   - How would you implement various activation functions like ReLU, Sigmoid, and Tanh?

32. **SVM**:
   - How would you solve the quadratic programming problem inherent to SVMs?
   - Implement the soft-margin SVM conceptually.

33. **Clustering**:
   - Can you implement the Mean Shift clustering algorithm?
   - How would you compute the silhouette score of a cluster?

34. **Manifold Learning**:
   - How would you implement the Locally Linear Embedding (LLE) algorithm?
   - Can you code the Isomap algorithm for dimensionality reduction?

35. **Model Selection and Evaluation**:
   - How would you calculate various metrics like precision, recall, and F1 score without using libraries?
   - Implement stratified k-fold cross-validation.

36. **Bayesian Methods**:
   - How would you code Gaussian Processes for regression problems?
   - Implement a naive Bayes classifier for multinomial datasets.

37. **Ensemble Methods**:
   - Can you write code for the gradient boosting algorithm without relying on decision trees as a black-box?
   - How would you implement stacking of multiple classifiers?
     
38. **Outlier Detection**:
   - Implement the concept behind the OneClassSVM for anomaly detection.
   - How would you manually compute the isolation score used in the Isolation Forest method?

39. **Pipeline and Utilities**:
   - How would you design a machine learning pipeline that sequences preprocessing, feature selection, and modeling?
   - Implement a grid search method for hyperparameter tuning without using Scikit-learn's `GridSearchCV`.

40. **Nearest Neighbors**:
   - Implement a KD-Tree or Ball Tree to optimize nearest neighbor searches.
   - How would you code the Nearest Centroid classifier?

41. **Density Estimation**:
   - Can you implement kernel density estimation?
   - How would you manually code the Parzen-window approach?
     
Remember, when implementing these from scratch, you won't have the optimizations provided by libraries like Scikit-learn or TensorFlow, so they might not be as efficient. However, the knowledge gained from this exercise is immensely valuable. Always compare the results of your custom implementations with established libraries to ensure correctness!
