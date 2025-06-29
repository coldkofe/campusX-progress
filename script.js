// Complete CampusX DSMP 2.0 Curriculum Data (Based on the Full PDF)
const curriculumData = [
    // Python Basics (Weeks 1-4)
    {
        id: 1,
        title: "Week 1: Basics of Python Programming",
        category: "python",
        sessions: [
            { id: "1-1", title: "Session 1: Python Basics", description: "Python fundamentals, data types, variables, comments, keywords" },
            { id: "1-2", title: "Session 2: Python Operators + if-else + Loops", description: "Operators, control structures, while/for loops" },
            { id: "1-3", title: "Session 3: Python Strings", description: "String operations, indexing, slicing, functions" },
            { id: "1-4", title: "Session on Time complexity", description: "Algorithm efficiency and time complexity analysis" },
            { id: "1-5", title: "Week 1 Interview Questions", description: "Practice questions and solutions" }
        ]
    },
    {
        id: 2,
        title: "Week 2: Python Data Types",
        category: "python",
        sessions: [
            { id: "2-1", title: "Session 4: Python Lists", description: "List operations, comprehensions, functions, memory storage" },
            { id: "2-2", title: "Session 5: Tuples + Set + Dictionary", description: "Advanced data structures and operations" },
            { id: "2-3", title: "Session 6: Python Functions", description: "Functions, args/kwargs, scope, lambda, higher-order functions" },
            { id: "2-4", title: "Array Interview Questions", description: "Practice problems and solutions" },
            { id: "2-5", title: "Week 2 Interview Questions", description: "Comprehensive review questions" }
        ]
    },
    {
        id: 3,
        title: "Week 3: Object Oriented Programming (OOP)",
        category: "python",
        sessions: [
            { id: "3-1", title: "Session 7: OOP Part1", description: "Classes, objects, methods, constructors, magic methods" },
            { id: "3-2", title: "Session 8: OOP Part2", description: "Encapsulation, static methods, reference variables, mutability" },
            { id: "3-3", title: "Session 9: OOP Part3", description: "Inheritance, polymorphism, method overriding, types of inheritance" },
            { id: "3-4", title: "Session on Abstraction", description: "Abstract classes and methods, bank example hierarchy" },
            { id: "3-5", title: "Session on OOP Project", description: "Hands-on OOP implementation project" }
        ]
    },
    {
        id: 4,
        title: "Week 4: Advanced Python",
        category: "python",
        sessions: [
            { id: "4-1", title: "Session 10: File Handling + Serialization", description: "File I/O, JSON, pickling, context managers" },
            { id: "4-2", title: "Session 11: Exception Handling", description: "Try-except-else-finally, custom exceptions" },
            { id: "4-3", title: "Session 12: Decorators and Namespaces", description: "LEGB rule, decorators with examples" },
            { id: "4-4", title: "Session on Iterators", description: "Iterator protocol, custom iterators, for loop internals" },
            { id: "4-5", title: "Session on Generator", description: "Generator functions, yield vs return, benefits" },
            { id: "4-6", title: "Session on Resume Building", description: "Professional resume creation for data science" },
            { id: "4-7", title: "Session on GUI Development using Python", description: "GUI development using tkinter" }
        ]
    },

    // Data Science Fundamentals (Weeks 5-12)
    {
        id: 5,
        title: "Week 5: Numpy",
        category: "data",
        sessions: [
            { id: "5-1", title: "Session 13: Numpy Fundamentals", description: "Arrays, operations, attributes, mathematical functions" },
            { id: "5-2", title: "Session 14: Advanced Numpy", description: "Broadcasting, fancy indexing, mathematical operations, sigmoid, MSE" },
            { id: "5-3", title: "Session 15: Numpy Tricks", description: "Advanced functions: sort, append, concatenate, percentile, set functions" },
            { id: "5-4", title: "Session on Web Development using Flask", description: "Flask basics, login system, name entity recognition API" }
        ]
    },
    {
        id: 6,
        title: "Week 6: Pandas",
        category: "data",
        sessions: [
            { id: "6-1", title: "Session 16: Pandas Series", description: "Series creation, methods, math operations, boolean indexing" },
            { id: "6-2", title: "Session 17: Pandas DataFrame", description: "DataFrame creation, read_csv, selecting rows/columns, filtering" },
            { id: "6-3", title: "Session 18: Important DataFrame Methods", description: "sort, index, reset_index, isnull, dropna, fillna, apply" },
            { id: "6-4", title: "Session on API Development using Flask", description: "Building APIs with Flask, hands-on project" },
            { id: "6-5", title: "Session on Numpy Interview Questions", description: "Common numpy interview questions and solutions" }
        ]
    },
    {
        id: 7,
        title: "Week 7: Advanced Pandas",
        category: "data",
        sessions: [
            { id: "7-1", title: "Session 19: GroupBy Object", description: "GroupBy operations, aggregation functions, IPL dataset hands-on" },
            { id: "7-2", title: "Session 20: Merging, Joining, Concatenating", description: "concat, merge, join methods with practical implementations" },
            { id: "7-3", title: "Session on Streamlit", description: "Streamlit introduction, features, benefits, Flask vs Streamlit" },
            { id: "7-4", title: "Session on Pandas Case Study", description: "Indian Startup Funding Dataset analysis with Streamlit dashboard" },
            { id: "7-5", title: "Session on Git", description: "Git basics, VCS, creating repos, add, commit, branches" },
            { id: "7-6", title: "Session on Git and GitHub", description: "Branching, merging, remote repos, collaboration" }
        ]
    },
    {
        id: 8,
        title: "Week 8: Advanced Pandas Continued",
        category: "data",
        sessions: [
            { id: "8-1", title: "Session 21: MultiIndex Series and DataFrames", description: "MultiIndex objects, stacking/unstacking, pivot tables" },
            { id: "8-2", title: "Session 22: Vectorized String Operations | Datetime", description: "String operations, datetime in pandas, agg functions" },
            { id: "8-3", title: "Session on Pandas Case Study – Time Series", description: "Time series analysis case study" },
            { id: "8-4", title: "Session on Pandas Case Study – Textual Data", description: "Working with textual data analysis" }
        ]
    },
    {
        id: 9,
        title: "Week 9: Data Visualization",
        category: "data",
        sessions: [
            { id: "9-1", title: "Session 23: Plotting Using Matplotlib", description: "Basic plots, labels, legends, scatter, bar, histogram, pie charts" },
            { id: "9-2", title: "Session 24: Advanced Matplotlib", description: "Colored scatter, subplots, 3D plots, contour plots, heatmaps" },
            { id: "9-3", title: "Session on Plotly Express", description: "Plotly introduction, advantages, hands-on examples" },
            { id: "9-4", title: "Session on Plotly Graph Objects", description: "Advanced Plotly with Graph Objects" },
            { id: "9-5", title: "Session on Plotly Dash", description: "Interactive dashboards with Dash" },
            { id: "9-6", title: "COVID-19 Dashboard Project", description: "Building COVID-19 dashboard using Plotly and Dash" },
            { id: "9-7", title: "Deploying Dash App on Heroku", description: "Cloud deployment of Dash applications" }
        ]
    },
    {
        id: 10,
        title: "Week 10: Data Visualization Continued",
        category: "data",
        sessions: [
            { id: "10-1", title: "Session 25: Seaborn Part 1", description: "Relational plots, distribution plots, KDE, matrix plots" },
            { id: "10-2", title: "Session 26: Seaborn Part 2", description: "Categorical plots, regression plots, FacetGrid, PairGrid" },
            { id: "10-3", title: "Session on Open-Source Software Part 1", description: "Introduction to open-source contribution" },
            { id: "10-4", title: "Session on Open-Source Software Part 2", description: "Advanced open-source practices" }
        ]
    },
    {
        id: 11,
        title: "Week 11: Data Analysis Process - Part1",
        category: "data",
        sessions: [
            { id: "11-1", title: "Session 27: Data Gathering", description: "Import from CSV, Excel, JSON, SQL, APIs, web scraping" },
            { id: "11-2", title: "Session 28: Data Assessing and Cleaning", description: "Data quality assessment, types of unclean data, cleaning techniques" },
            { id: "11-3", title: "Session on ETL using AWS RDS", description: "Extract, Transform, Load pipeline with AWS" },
            { id: "11-4", title: "Session on Advanced Web Scraping", description: "Selenium automation, Smartprix website scraping" }
        ]
    },
    {
        id: 12,
        title: "Week 12: Data Analysis Process – Part 2",
        category: "data",
        sessions: [
            { id: "12-1", title: "Session on Data Cleaning Case Study", description: "Smartphone dataset quality and tidiness issues" },
            { id: "12-2", title: "Session 29: Exploratory Data Analysis (EDA)", description: "EDA steps, univariate/bivariate analysis, feature engineering" },
            { id: "12-3", title: "Session on Data Cleaning Part 2", description: "Continued smartphone dataset cleaning" },
            { id: "12-4", title: "Session on EDA Case Study", description: "Complete EDA on smartphone dataset" }
        ]
    },

    // SQL (Weeks 13-16)
    {
        id: 13,
        title: "Week 13: SQL Basics",
        category: "data",
        sessions: [
            { id: "13-1", title: "Session 30: Database Fundamentals", description: "CRUD operations, database properties, DBMS, keys, cardinality" },
            { id: "13-2", title: "Session 31: SQL DDL Commands", description: "Data Definition Language commands, XAMPP setup" },
            { id: "13-3", title: "Session on Tableau Part 1", description: "Olympics dataset dashboard creation" }
        ]
    },
    {
        id: 14,
        title: "Week 14: SQL Continued – Part 1",
        category: "data",
        sessions: [
            { id: "14-1", title: "Session 32: SQL DML Commands", description: "INSERT, SELECT, UPDATE, DELETE, MySQL workbench" },
            { id: "14-2", title: "Session 33: SQL Grouping and Sorting", description: "ORDER BY, GROUP BY, HAVING clause, IPL dataset practice" },
            { id: "14-3", title: "Session on Tableau Part 2", description: "Advanced Tableau: measures, dimensions, calculated fields, geographical data" }
        ]
    },
    {
        id: 15,
        title: "Week 15: SQL Continued - Part 2",
        category: "data",
        sessions: [
            { id: "15-1", title: "Session 34: SQL Joins", description: "All types of joins, SET operations, SELF join, query execution order" },
            { id: "15-2", title: "Session on SQL Case Study 1", description: "Zomato dataset SQL questions and solutions" },
            { id: "15-3", title: "Session 35: Subqueries in SQL", description: "Independent and correlated subqueries" },
            { id: "15-4", title: "Session on Flight Dashboard", description: "Python-SQL integration with Streamlit dashboard" },
            { id: "15-5", title: "Session on SQL Interview Questions Part 1", description: "Database engines, collation, COUNT variations, NULL handling" }
        ]
    },
    {
        id: 16,
        title: "Week 16: Advanced SQL",
        category: "data",
        sessions: [
            { id: "16-1", title: "Session 36: Window Functions Part 1", description: "OVER(), RANK(), DENSE_RANK(), ROW_NUMBER(), frames concept" },
            { id: "16-2", title: "Session 37: Window Functions Part 2", description: "Cumulative operations, running averages, percent calculations" },
            { id: "16-3", title: "Session 37: Window Functions Part 3", description: "Quantiles, segmentation, cumulative distribution" },
            { id: "16-4", title: "Session on Data Cleaning using SQL", description: "Laptop dataset cleaning with SQL, string functions" },
            { id: "16-5", title: "Session on EDA using SQL", description: "Complete EDA on laptop dataset using SQL" }
        ]
    },

    // Statistics (Weeks 17-21)
    {
        id: 17,
        title: "Week 17: Descriptive Statistics",
        category: "ml",
        sessions: [
            { id: "17-1", title: "Session 38: Descriptive Statistics Part 1", description: "Population vs sample, data types, central tendency, dispersion" },
            { id: "17-2", title: "Session on Datetime in SQL", description: "Temporal data types, datetime functions, flights case study" }
        ]
    },
    {
        id: 18,
        title: "Week 18: Descriptive Statistics Continued",
        category: "ml",
        sessions: [
            { id: "18-1", title: "Session 39: Descriptive Statistics Part 2", description: "Quantiles, percentiles, boxplots, correlation vs causation" },
            { id: "18-2", title: "Session 40: Probability Distribution Functions", description: "PMF, CDF, PDF, density estimation, KDE" },
            { id: "18-3", title: "Session on SQL Datetime Case Study", description: "Flights dataset temporal analysis" },
            { id: "18-4", title: "Session on Database Design", description: "SQL data types, database normalization, ER diagrams" }
        ]
    },
    {
        id: 19,
        title: "Week 19: Probability Distributions",
        category: "ml",
        sessions: [
            { id: "19-1", title: "Session 41: Normal Distribution", description: "Normal distribution properties, z-table, empirical rule, skewness" },
            { id: "19-2", title: "Session 42: Non-Gaussian Distributions", description: "Kurtosis, QQ plots, uniform, log-normal, Pareto, transformations" },
            { id: "19-3", title: "Session on Views and UDFs", description: "SQL views and user-defined functions" },
            { id: "19-4", title: "Session on Transactions", description: "Stored procedures, ACID properties, transactions" }
        ]
    },
    {
        id: 20,
        title: "Week 20: Inferential Statistics",
        category: "ml",
        sessions: [
            { id: "20-1", title: "Session 43: Central Limit Theorem", description: "Bernoulli, binomial distributions, sampling distribution, CLT proof" },
            { id: "20-2", title: "Session 44: Confidence Intervals", description: "Point estimates, z-procedure, t-procedure, t-distribution" }
        ]
    },
    {
        id: 21,
        title: "Week 21: Hypothesis Testing",
        category: "ml",
        sessions: [
            { id: "21-1", title: "Session 45: Hypothesis Testing Part 1", description: "Null/alternate hypothesis, z-test, Type I/II errors" },
            { id: "21-2", title: "Session 46: Hypothesis Testing Part 2", description: "p-value interpretation, t-tests (single, independent, paired)" },
            { id: "21-3", title: "Session on Chi-square Test", description: "Chi-square distribution, goodness of fit, independence test" },
            { id: "21-4", title: "Session on ANOVA", description: "F-distribution, one-way ANOVA, post-hoc tests" }
        ]
    },

    // Linear Algebra & Machine Learning (Weeks 22-36)
    {
        id: 22,
        title: "Week 22: Linear Algebra",
        category: "ml",
        sessions: [
            { id: "22-1", title: "Session on Tensors", description: "0D to nD tensors, rank, axes, shape, ML examples" },
            { id: "22-2", title: "Session on Vectors", description: "Vector operations, dot product, angles, hyperplane equations" },
            { id: "22-3", title: "Linear Algebra Part 2", description: "Matrix operations, determinant, inverse, linear equations" },
            { id: "22-4", title: "Linear Algebra Part 3", description: "Linear transformations, basis vectors, matrix composition" }
        ]
    },
    {
        id: 23,
        title: "Week 23: Linear Regression",
        category: "ml",
        sessions: [
            { id: "23-1", title: "Session 48: Introduction to Machine Learning", description: "ML types, challenges, development lifecycle, job roles" },
            { id: "23-2", title: "Session 49: Simple Linear Regression", description: "Regression intuition, code from scratch, regression metrics" },
            { id: "23-3", title: "Session 50: Multiple Linear Regression", description: "MLR mathematical formulation, error minimization" },
            { id: "23-4", title: "Session on Optimization Big Picture", description: "ML as mathematical functions, loss functions, gradient descent" },
            { id: "23-5", title: "Session on Differential Calculus", description: "Derivatives, partial differentiation, matrix differentiation" }
        ]
    },
    {
        id: 24,
        title: "Week 24: Gradient Descent",
        category: "ml",
        sessions: [
            { id: "24-1", title: "Session 51: Gradient Descent from Scratch", description: "GD intuition, mathematical formulation, learning rate effects" },
            { id: "24-2", title: "Session 52: Batch Gradient Descent", description: "Batch GD mathematical formulation and implementation" },
            { id: "24-3", title: "Session 52: Stochastic Gradient Descent", description: "SGD problems and solutions, learning schedules" },
            { id: "24-4", title: "Session 52: Mini-batch Gradient Descent", description: "Mini-batch GD implementation and visualization" },
            { id: "24-5", title: "Doubt Clearance on Linear Regression", description: "Linear regression concepts clarification" }
        ]
    },
    {
        id: 25,
        title: "Week 25: Regression Analysis",
        category: "ml",
        sessions: [
            { id: "25-1", title: "Regression Analysis Part 1", description: "Statistical perspective, TSS/RSS/ESS, F-statistic" },
            { id: "25-2", title: "Regression Analysis Part 2", description: "R-squared, adjusted R-squared, t-statistic, confidence intervals" },
            { id: "25-3", title: "Session on Polynomial Regression", description: "Non-linear relationships, polynomial features" },
            { id: "25-4", title: "Session on Assumptions of Linear Regression", description: "Linearity, normality, homoscedasticity, multicollinearity" },
            { id: "25-5", title: "Session 53: Multicollinearity", description: "VIF, condition number, correlation, multicollinearity removal" }
        ]
    },
    {
        id: 26,
        title: "Week 26: Feature Selection",
        category: "ml",
        sessions: [
            { id: "26-1", title: "Session 54: Feature Selection Part 1", description: "Filter methods: variance threshold, correlation, ANOVA, chi-square" },
            { id: "26-2", title: "Session 55: Feature Selection Part 2", description: "Wrapper methods: exhaustive, forward/backward selection" },
            { id: "26-3", title: "Session on Feature Selection Part 3", description: "Embedded methods: regularized models, RFE" }
        ]
    },
    {
        id: 27,
        title: "Week 27: Regularization",
        category: "ml",
        sessions: [
            { id: "27-1", title: "Session on Bias-Variance Tradeoff", description: "Bias-variance decomposition, mathematical formulation" },
            { id: "27-2", title: "Ridge Regression Part 1", description: "Geometric intuition, sklearn implementation" },
            { id: "27-3", title: "Ridge Regression Part 2", description: "2D and nD data, implementation from scratch" },
            { id: "27-4", title: "Ridge Regression Part 3", description: "Ridge regression using gradient descent" },
            { id: "27-5", title: "Ridge Regression Part 4", description: "Key understandings: coefficients, bias-variance, contour plots" },
            { id: "27-6", title: "Lasso Regression", description: "Lasso intuition, sparsity creation, implementation" },
            { id: "27-7", title: "ElasticNet Regression", description: "Combined L1 and L2 regularization" }
        ]
    },
    {
        id: 28,
        title: "Week 28: K Nearest Neighbors",
        category: "ml",
        sessions: [
            { id: "28-1", title: "KNN Part 1", description: "KNN intuition, K selection, decision surface, overfitting" },
            { id: "28-2", title: "KNN from Scratch", description: "Complete KNN implementation" },
            { id: "28-3", title: "Decision Boundary Visualization", description: "Drawing decision boundaries for classification" },
            { id: "28-4", title: "Advanced KNN Part 2", description: "KNN regressor, weighted KNN, distance metrics, KD-tree" },
            { id: "28-5", title: "Classification Metrics Part 1", description: "Accuracy, confusion matrix, Type I/II errors" },
            { id: "28-6", title: "Classification Metrics Part 2", description: "Precision, recall, F1-score, multi-class metrics" }
        ]
    },
    {
        id: 29,
        title: "Week 29: PCA",
        category: "ml",
        sessions: [
            { id: "29-1", title: "Curse of Dimensionality", description: "High-dimensional data problems" },
            { id: "29-2", title: "PCA Part 1", description: "Geometric intuition, variance importance" },
            { id: "29-3", title: "PCA Part 2", description: "Mathematical formulation, covariance matrix, eigenvalues" },
            { id: "29-4", title: "PCA Part 3", description: "MNIST example, explained variance, optimal components" },
            { id: "29-5", title: "Eigenvalues and Eigenvectors", description: "Mathematical concepts and PCA applications" },
            { id: "29-6", title: "Eigen Decomposition and PCA Variants", description: "Spectral decomposition, Kernel PCA" },
            { id: "29-7", title: "Singular Value Decomposition", description: "SVD intuition, applications, relationship with PCA" }
        ]
    },
    {
        id: 30,
        title: "Week 30: Model Evaluation & Selection",
        category: "ml",
        sessions: [
            { id: "30-1", title: "ROC Curve", description: "ROC-AUC curve, TPR, FPR, different cases" },
            { id: "30-2", title: "Cross Validation", description: "Hold-out, LOOCV, K-fold, stratified cross-validation" },
            { id: "30-3", title: "Data Leakage", description: "Types of data leakage, detection, prevention" },
            { id: "30-4", title: "Hyperparameter Tuning", description: "Grid search, randomized search, parameter vs hyperparameter" }
        ]
    },
    {
        id: 31,
        title: "Week 31: Naive Bayes",
        category: "ml",
        sessions: [
            { id: "31-1", title: "Probability Crash Course Part 1", description: "Random experiments, sample space, events, probability types" },
            { id: "31-2", title: "Probability Crash Course Part 2", description: "Joint, marginal, conditional probability, Bayes theorem" },
            { id: "31-3", title: "Naive Bayes Session 1", description: "Intuition, mathematical formulation, numerical and textual data" },
            { id: "31-4", title: "Naive Bayes Session 2", description: "Log probabilities, Laplace smoothing, bias-variance tradeoff" },
            { id: "31-5", title: "Naive Bayes Session 3", description: "Gaussian, categorical, multinomial, Bernoulli variants" },
            { id: "31-6", title: "Email Spam Classifier Project", description: "End-to-end Naive Bayes project" }
        ]
    },
    {
        id: 32,
        title: "Week 32: Logistic Regression",
        category: "ml",
        sessions: [
            { id: "32-1", title: "Logistic Regression Session 1", description: "Classification problems, sigmoid function, log loss, gradient descent" },
            { id: "32-2", title: "Multiclass Classification", description: "One vs Rest, SoftMax regression approaches" },
            { id: "32-3", title: "Maximum Likelihood Estimation", description: "MLE concepts, probability vs likelihood, MLE in logistic regression" },
            { id: "32-4", title: "Logistic Regression Session 3", description: "Assumptions, odds ratio, polynomial features, regularization" },
            { id: "32-5", title: "Logistic Regression Hyperparameters", description: "Important hyperparameters and tuning" }
        ]
    },
    {
        id: 33,
        title: "Week 33: Support Vector Machines",
        category: "ml",
        sessions: [
            { id: "33-1", title: "SVM Part 1 - Hard Margin", description: "Maximum margin classifier, support vectors, mathematical formulation" },
            { id: "33-2", title: "SVM Part 2 - Soft Margin", description: "Slack variables, C parameter, bias-variance tradeoff" },
            { id: "33-3", title: "Constrained Optimization", description: "Kernel intuition, types of kernels, kernel trick" },
            { id: "33-4", title: "SVM Dual Problem", description: "KKT conditions, duality concept, dual formulation" },
            { id: "33-5", title: "Math Behind SVM Kernels", description: "Polynomial kernel, RBF kernel, custom kernels" }
        ]
    },

    // Feature Engineering (Extra Sessions)
    {
        id: 34,
        title: "Feature Engineering - Missing Values",
        category: "advanced",
        sessions: [
            { id: "34-1", title: "Handling Missing Values Part 1", description: "Types of missing values, complete case analysis, univariate imputation" },
            { id: "34-2", title: "Handling Missing Values Part 2", description: "Mean, median, mode imputation, missing indicator" },
            { id: "34-3", title: "Handling Missing Values Part 3", description: "KNN imputer, iterative imputer, MICE algorithm" }
        ]
    },

    // Decision Trees & Ensemble Methods (Weeks 34-36)
    {
        id: 35,
        title: "Week 34: Decision Trees",
        category: "ml",
        sessions: [
            { id: "35-1", title: "Decision Tree Session 1", description: "CART algorithm, Gini impurity, splitting criteria" },
            { id: "35-2", title: "Decision Tree Session 2", description: "Regression trees, geometric intuition, advantages/disadvantages" },
            { id: "35-3", title: "Decision Tree Session 3", description: "Feature importance, overfitting, pruning techniques" },
            { id: "35-4", title: "Decision Tree Visualization", description: "dtreeviz demo and coding" }
        ]
    },
    {
        id: 36,
        title: "Week 35: Ensemble Methods",
        category: "ml",
        sessions: [
            { id: "36-1", title: "Introduction to Ensemble Learning", description: "Types of ensemble, why it works, when to use" },
            { id: "36-2", title: "Bagging Part 1", description: "Core idea, when to use bagging" },
            { id: "36-3", title: "Bagging Part 2 - Classifier", description: "Bagging classifier intuition and implementation" },
            { id: "36-4", title: "Bagging Part 3 - Regressor", description: "Bagging regressor implementation" },
            { id: "36-5", title: "Random Forest Session 1", description: "Random forest intuition, feature importance, OOB score" },
            { id: "36-6", title: "Random Forest Session 2", description: "Hyperparameters, extremely randomized trees" }
        ]
    },
    {
        id: 37,
        title: "Week 36: Gradient Boosting",
        category: "ml",
        sessions: [
            { id: "37-1", title: "Gradient Boosting Session 1", description: "Boosting concept, how/what/why of gradient boosting" },
            { id: "37-2", title: "Gradient Boosting Session 2", description: "Function space vs parameter space, loss minimization" },
            { id: "37-3", title: "Gradient Boosting Session 3", description: "Classification implementation part 1" },
            { id: "37-4", title: "Gradient Boosting Classification 2", description: "Geometric intuition for classification" },
            { id: "37-5", title: "Gradient Boosting Classification 3", description: "Mathematical formulation, pseudo residuals" }
        ]
    },

    // Capstone Project
    {
        id: 38,
        title: "Capstone Project - Real Estate Price Prediction",
        category: "advanced",
        sessions: [
            { id: "38-1", title: "Session 1: Data Gathering", description: "Project overview, data collection and details" },
            { id: "38-2", title: "Session 2: Data Cleaning", description: "Merging data, basic level cleaning" },
            { id: "38-3", title: "Session 3: Feature Engineering", description: "Advanced feature engineering on multiple columns" },
            { id: "38-4", title: "Session 4: EDA", description: "Univariate, multivariate analysis, pandas profiling" },
            { id: "38-5", title: "Session 5: Outlier Detection", description: "Outlier detection and removal techniques" },
            { id: "38-6", title: "Session 6: Missing Value Imputation", description: "Advanced imputation strategies" },
            { id: "38-7", title: "Session 7: Feature Selection", description: "Multiple feature selection techniques, SHAP" },
            { id: "38-8", title: "Session 8: Model Selection", description: "Encoding selection, model comparison, web interface" },
            { id: "38-9", title: "Session 9: Analytics Module", description: "Geo maps, word clouds, interactive visualizations" },
            { id: "38-10", title: "Session 10: Recommender System", description: "Recommendation system using multiple approaches" },
            { id: "38-11", title: "Session 11: Recommender System Part 2", description: "Evaluation and web interface for recommendations" },
            { id: "38-12", title: "Session 12: Insights Module", description: "Building comprehensive insights dashboard" },
            { id: "38-13", title: "Session 13: AWS Deployment", description: "Deploying application on AWS cloud" }
        ]
    },

    // XGBoost Deep Dive
    {
        id: 39,
        title: "XGBoost (Extreme Gradient Boosting)",
        category: "advanced",
        sessions: [
            { id: "39-1", title: "Introduction to XGBoost", description: "Features: performance, speed, flexibility" },
            { id: "39-2", title: "XGBoost for Regression", description: "Step-by-step mathematical calculation" },
            { id: "39-3", title: "XGBoost for Classification", description: "Classification problem mathematical formulation" },
            { id: "39-4", title: "Complete Math of XGBoost", description: "Taylor series, objective function, similarity score derivation" }
        ]
    },

    // MLOps Curriculum (Weeks 1-9)
    {
        id: 40,
        title: "MLOps Week 1: Introduction to MLOps",
        category: "mlops",
        sessions: [
            { id: "40-1", title: "Session 1: Introduction to MLOps", description: "ML lifecycle, DevOps vs MLOps, version control basics" },
            { id: "40-2", title: "Session 2: Version Control", description: "GitHub setup, repositories, branching, pull requests" },
            { id: "40-3", title: "Doubt Clearance Session 1", description: "GitHub fundamentals, CLI operations, error resolution" }
        ]
    },
    {
        id: 41,
        title: "MLOps Week 2: ML Reproducibility & Versioning",
        category: "mlops",
        sessions: [
            { id: "41-1", title: "Session 3: Reproducibility", description: "Cookiecutter templates, project structure" },
            { id: "41-2", title: "Session 4: Data Versioning Control", description: "DVC setup, pipeline versioning, experiment reproduction" },
            { id: "41-3", title: "Doubt Clearance Session 2", description: "DVC with Google Drive, setup errors, version management" }
        ]
    },
    {
        id: 42,
        title: "MLOps Week 3: End-to-end ML Lifecycle",
        category: "mlops",
        sessions: [
            { id: "42-1", title: "Session 5: ML Pipelines", description: "MLFlow introduction, experimentation tracking" },
            { id: "42-2", title: "Session 6: MLOps Pipeline", description: "Credit card example, dvc.yml files, reproducibility" },
            { id: "42-3", title: "Doubt Clearance Session 3", description: "Pipeline assignments, file errors, DVC commands" }
        ]
    },
    {
        id: 43,
        title: "MLOps Week 4: Containerization & Deployment",
        category: "mlops",
        sessions: [
            { id: "43-1", title: "Session 7: Continuous Integration", description: "CI/CD philosophy, GitHub Actions, CML integration" },
            { id: "43-2", title: "Session 8: Containerization", description: "Docker fundamentals and containerization" }
        ]
    },
    {
        id: 44,
        title: "MLOps Week 5: DAGs in MLOps",
        category: "mlops",
        sessions: [
            { id: "44-1", title: "Session 9: Continuous Deployment", description: "FastAPI, pydantic, multi-container deployment" },
            { id: "44-2", title: "Doubt Clearance Session 4", description: "Assignment solutions and troubleshooting" }
        ]
    },
    {
        id: 45,
        title: "MLOps Week 6: Monitoring & Alerting",
        category: "mlops",
        sessions: [
            { id: "45-1", title: "Session 10: Introduction to AWS", description: "AWS ML services, SageMaker, S3, Lambda, ECR, ECS" },
            { id: "45-2", title: "Session 11: Deployment on AWS", description: "Credit card project deployment, EC2, self-runners" }
        ]
    },
    {
        id: 46,
        title: "MLOps Week 7: Scaling & Efficiency",
        category: "mlops",
        sessions: [
            { id: "46-1", title: "Session 12: Distributed Infrastructure", description: "Distributed computing, microservices architecture" },
            { id: "46-2", title: "Session 13: Kubernetes Internals", description: "Container orchestration, pods, nodes, clusters" },
            { id: "46-3", title: "Doubt Clearance Session 6", description: "Kubernetes and deployment troubleshooting" }
        ]
    },
    {
        id: 47,
        title: "MLOps Week 8: Final Project",
        category: "mlops",
        sessions: [
            { id: "47-1", title: "Session 14: Deployment on Kubernetes", description: "Kubectl deployment, strategies, load balancing" },
            { id: "47-2", title: "Session 15: Seldon Deployments", description: "Seldon Core, Kubeflow Pipelines, Apache Airflow" },
            { id: "47-3", title: "Doubt Clearance Session 7", description: "Advanced deployment troubleshooting" }
        ]
    },
    {
        id: 48,
        title: "MLOps Week 9: ML Technical Debt",
        category: "mlops",
        sessions: [
            { id: "48-1", title: "Session 16: Monitoring & Alerting", description: "Production monitoring strategies" },
            { id: "48-2", title: "Session 17: Rollout & Rollback", description: "Deployment strategies and rollback procedures" },
            { id: "48-3", title: "Session on MLOps Interview Questions", description: "Common MLOps interview preparation" },
            { id: "48-4", title: "Session 18: ML Technical Debt", description: "Managing technical debt in ML projects" },
            { id: "48-5", title: "Doubt Clearance Session 8", description: "Final MLOps concepts clarification" }
        ]
    },

    // Unsupervised Learning
    {
        id: 49,
        title: "Unsupervised Learning - Clustering",
        category: "advanced",
        sessions: [
            { id: "49-1", title: "KMeans Clustering Session 1", description: "Clustering applications, geometric intuition, elbow method, assumptions" },
            { id: "49-2", title: "KMeans Clustering Session 2", description: "Silhouette score, hyperparameters, K-means++" },
            { id: "49-3", title: "KMeans Clustering Session 3", description: "Lloyd's algorithm, time complexity, mini-batch K-means" },
            { id: "49-4", title: "K-Means from Scratch", description: "Complete algorithm implementation" }
        ]
    },
    {
        id: 50,
        title: "Unsupervised Learning - Other Algorithms",
        category: "advanced",
        sessions: [
            { id: "50-1", title: "DBSCAN", description: "Density-based clustering, core/border/noise points" },
            { id: "50-2", title: "Hierarchical Clustering", description: "Agglomerative clustering, linkage criteria, dendrograms" },
            { id: "50-3", title: "Gaussian Mixture Models Session 1", description: "GMM intuition, multivariate normal distribution, EM algorithm" },
            { id: "50-4", title: "Gaussian Mixture Models Session 2", description: "Covariance types, AIC/BIC, applications" },
            { id: "50-5", title: "T-SNE Session 1", description: "Dimensionality reduction for visualization" },
            { id: "50-6", title: "T-SNE Session 2", description: "Mathematical formulation, hyperparameters, best practices" }
        ]
    },

    // Advanced Feature Engineering
    {
        id: 51,
        title: "Advanced Feature Engineering - Encoding",
        category: "advanced",
        sessions: [
            { id: "51-1", title: "Encoding Categorical Features 1", description: "Ordinal, label, one-hot encoding, handling rare categories" },
            { id: "51-2", title: "Sklearn ColumnTransformer & Pipeline", description: "Pipeline creation, multiple transformations" },
            { id: "51-3", title: "Sklearn Deep Dive", description: "Custom estimators, transformers, mixins" },
            { id: "51-4", title: "Encoding Categorical Features 2", description: "Count, frequency, binary, target encoding" }
        ]
    },
    {
        id: 52,
        title: "Advanced Feature Engineering - Scaling & Transformation",
        category: "advanced",
        sessions: [
            { id: "52-1", title: "Discretization Session 1", description: "Why discretization, advantages and disadvantages" },
            { id: "52-2", title: "Discretization Session 2", description: "Uniform, quantile, K-means, decision tree binning" },
            { id: "52-3", title: "Feature Scaling Session 1", description: "Standardization, when to use feature scaling" },
            { id: "52-4", title: "Feature Scaling Session 2", description: "MinMax, robust, max absolute scaling, normalization" },
            { id: "52-5", title: "Outlier Detection Session 1", description: "Types of outliers, Z-score, IQR, isolation forest" },
            { id: "52-6", title: "Outlier Detection Session 2", description: "KNN-based detection, local vs global outliers, LOF" },
            { id: "52-7", title: "Outlier Detection Session 3", description: "DBSCAN for outliers, accuracy assessment" },
            { id: "52-8", title: "Feature Transformation", description: "Log, square root, reciprocal, Box-Cox, Yeo-Johnson" }
        ]
    },

    // Advanced XGBoost
    {
        id: 53,
        title: "Advanced XGBoost Deep Dive",
        category: "advanced",
        sessions: [
            { id: "53-1", title: "Revisiting XGBoost", description: "Supervised ML, stagewise additive modeling, objective function" },
            { id: "53-2", title: "XGBoost Regularization", description: "Overfitting reduction, gamma, max depth, early stopping" },
            { id: "53-3", title: "XGBoost Regularization Continued", description: "Min child weight, lambda, alpha, subsampling" },
            { id: "53-4", title: "XGBoost Optimizations", description: "Exact greedy, approximate methods, parallel processing" },
            { id: "53-5", title: "XGBoost Missing Values", description: "Handling missing values in XGBoost" }
        ]
    },

    // Competitive Data Science
    {
        id: 54,
        title: "Competitive Data Science",
        category: "advanced",
        sessions: [
            { id: "54-1", title: "Adaboost", description: "Weak learners, weights, learning rate, applications" },
            { id: "54-2", title: "Stacking", description: "Model ensembling, base models, meta-model, variations" },
            { id: "54-3", title: "LightGBM Session 1", description: "Histogram-based splitting, leaf-wise growth, GOSS, EFB" },
            { id: "54-4", title: "LightGBM Session 2", description: "GOSS and EFB deep dive" },
            { id: "54-5", title: "CatBoost", description: "Categorical feature handling, ordered boosting" },
            { id: "54-6", title: "Advanced Hyperparameter Tuning", description: "Bayesian optimization, Optuna, Hyperopt" },
            { id: "54-7", title: "Kaggle Competition", description: "Real competition participation, strategies, collaboration" }
        ]
    },

    // Miscellaneous Advanced Topics
    {
        id: 55,
        title: "Advanced Topics & Tools",
        category: "advanced",
        sessions: [
            { id: "55-1", title: "NoSQL", description: "Document, key-value, column-family, graph databases" },
            { id: "55-2", title: "Model Explainability", description: "LIME, SHAP, feature importance, interpretable models" },
            { id: "55-3", title: "FastAPI", description: "Modern API framework, type checking, automatic validation" },
            { id: "55-4", title: "AWS Sagemaker", description: "Fully managed ML service, model building to deployment" }
        ]
    },

    // Handling Imbalanced Data
    {
        id: 56,
        title: "Handling Imbalanced Data",
        category: "advanced",
        sessions: [
            { id: "56-1", title: "Imbalanced Data Session 1", description: "Problems with imbalanced data, oversampling, undersampling techniques" },
            { id: "56-2", title: "Imbalanced Data Session 2", description: "SMOTE, Borderline SMOTE, ADASYN, SMOTENC" },
            { id: "56-3", title: "Imbalanced Data Session 3", description: "Tomek links, edited nearest neighbors, cluster centroids" }
        ]
    },

    // Regular Expressions
    {
        id: 57,
        title: "Regular Expressions - Text Processing",
        category: "advanced",
        sessions: [
            { id: "57-1", title: "Regular Expressions Session 1", description: "Meta characters, character sets, quantifiers, greedy vs lazy" },
            { id: "57-2", title: "Regular Expressions Session 2", description: "Grouping, back references, assertions, flags, substitution" }
        ]
    },

    // Interview Questions
    {
        id: 58,
        title: "Statistics Interview Questions",
        category: "interview",
        sessions: [
            { id: "58-1", title: "Statistics Interview Questions Session 1", description: "Fundamental statistics concepts for interviews" },
            { id: "58-2", title: "Statistics Interview Questions Session 2", description: "Probability and distributions interview questions" },
            { id: "58-3", title: "Statistics Interview Questions Session 3", description: "Hypothesis testing and inference questions" },
            { id: "58-4", title: "Statistics Interview Questions Session 4", description: "Advanced statistics interview preparation" }
        ]
    },
    {
        id: 59,
        title: "Machine Learning Interview Questions",
        category: "interview",
        sessions: [
            { id: "59-1", title: "ML Interview Questions Session 1", description: "Core ML algorithms and concepts" },
            { id: "59-2", title: "ML Interview Questions Session 2", description: "Model evaluation and selection questions" },
            { id: "59-3", title: "ML Interview Questions Session 3", description: "Advanced ML topics for interviews" },
            { id: "59-4", title: "ML Interview Questions Session 4", description: "Practical ML implementation questions" }
        ]
    },
    {
        id: 60,
        title: "SQL & Python Interview Questions",
        category: "interview",
        sessions: [
            { id: "60-1", title: "SQL Interview Questions Session 1", description: "Database concepts and SQL query optimization" },
            { id: "60-2", title: "SQL Interview Questions Session 2", description: "Advanced SQL and database design questions" },
            { id: "60-3", title: "Python Interview Questions", description: "Python programming concepts for data science interviews" },
            { id: "60-4", title: "Project Based Interview Questions", description: "Real-world project scenarios and solutions" }
        ]
    },

    // MLOps Revisited (By Nitish Sir)
    {
        id: 61,
        title: "MLOps Revisited - Advanced Concepts",
        category: "mlops",
        sessions: [
            { id: "61-1", title: "MLOps Revisited Session 1", description: "MLOps introduction, problems in traditional ML projects" },
            { id: "61-2", title: "MLOps Revisited Session 2", description: "MLOps tools stack, benefits, challenges" },
            { id: "61-3", title: "MLOps Revisited Session 3", description: "Data management tools, pipeline integration" },
            { id: "61-4", title: "MLOps Revisited Session 4", description: "Code management, model building, AutoML" },
            { id: "61-5", title: "MLOps Revisited Session 5", description: "DevOps fundamentals, virtual machines, containerization" },
            { id: "61-6", title: "MLOps Revisited Session 6", description: "Microservices, Kubernetes, container orchestration" },
            { id: "61-7", title: "MLOps Revisited Session 7", description: "CI/CD pipelines, model serving" },
            { id: "61-8", title: "MLOps Revisited Session 8", description: "Cloud infrastructure, load balancers, auto scaling" },
            { id: "61-9", title: "MLOps Revisited Session 9", description: "ML pipelines using DVC, tweet emotion project" },
            { id: "61-10", title: "MLOps Revisited Session 10", description: "Improving ML pipelines, logging, exception handling" },
            { id: "61-11", title: "MLOps Revisited Session 11", description: "Version control, data versioning" },
            { id: "61-12", title: "MLOps Revisited Session 12", description: "Data versioning using DVC" },
            { id: "61-13", title: "MLOps Revisited Session 13", description: "Pipeline x Data Versioning x AWS S3" },
            { id: "61-14", title: "MLOps Revisited Session 14", description: "Experiment tracking using DVC" },
            { id: "61-15", title: "MLOps Revisited Session 15", description: "Introduction to MLflow" },
            { id: "61-16", title: "MLOps Revisited Session 16", description: "MLOps Remote Tracking using Dagshub and AWS" },
            { id: "61-17", title: "MLOps Revisited Session 17", description: "Autologging and Hyperparameter Tuning in MLflow" },
            { id: "61-18", title: "MLOps Revisited Session 18", description: "Model Registry" },
            { id: "61-19", title: "MLOps Revisited Session 19", description: "Mini Project" },
            { id: "61-20", title: "MLOps Revisited Session 20", description: "Model Serving" },
            { id: "61-21", title: "MLOps Revisited Session 21", description: "Fundamentals of CI" }
        ]
    }
];

class ProgressTracker {
    constructor() {
        this.completedSessions = new Set(JSON.parse(localStorage.getItem('campusx-completed-sessions') || '[]'));
        this.init();
    }

    init() {
        this.renderWeeks();
        this.updateOverallProgress();
        this.bindEvents();
    }

    renderWeeks() {
        const container = document.getElementById('weeksContainer');
        container.innerHTML = '';

        curriculumData.forEach(week => {
            const weekElement = this.createWeekElement(week);
            container.appendChild(weekElement);
        });
    }

    createWeekElement(week) {
        const weekDiv = document.createElement('div');
        weekDiv.className = `week-card ${week.category}`;
        weekDiv.dataset.category = week.category;

        const completedCount = week.sessions.filter(session => 
            this.completedSessions.has(session.id)
        ).length;
        const progressPercentage = (completedCount / week.sessions.length) * 100;

        weekDiv.innerHTML = `
            <div class="week-header">
                <h2 class="week-title">${week.title}</h2>
                <div class="week-progress">
                    <div class="week-progress-bar">
                        <div class="week-progress-fill" style="width: ${progressPercentage}%"></div>
                    </div>
                    <span>${Math.round(progressPercentage)}%</span>
                </div>
            </div>
            <div class="sessions-grid">
                ${week.sessions.map(session => this.createSessionElement(session)).join('')}
            </div>
        `;

        return weekDiv;
    }

    createSessionElement(session) {
        const isCompleted = this.completedSessions.has(session.id);
        
        return `
            <div class="session-item ${isCompleted ? 'completed' : ''}" data-session-id="${session.id}">
                <div class="session-header">
                    <h3 class="session-title">${session.title}</h3>
                    <div class="session-checkbox ${isCompleted ? 'checked' : ''}" data-session-id="${session.id}">
                        ${isCompleted ? '<i class="fas fa-check"></i>' : ''}
                    </div>
                </div>
                <p class="session-description">${session.description}</p>
            </div>
        `;
    }

    bindEvents() {
        // Session completion toggle
        document.addEventListener('click', (e) => {
            if (e.target.closest('.session-checkbox') || e.target.closest('.session-item')) {
                const sessionId = e.target.closest('.session-item').dataset.sessionId;
                this.toggleSession(sessionId);
            }
        });

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.filterWeeks(e.target.dataset.filter);
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // Floating action buttons
        document.getElementById('resetProgress').addEventListener('click', () => {
            if (confirm('Are you sure you want to reset all progress? This action cannot be undone.')) {
                this.resetProgress();
            }
        });

        document.getElementById('exportProgress').addEventListener('click', () => {
            this.exportProgress();
        });
    }

    toggleSession(sessionId) {
        if (this.completedSessions.has(sessionId)) {
            this.completedSessions.delete(sessionId);
        } else {
            this.completedSessions.add(sessionId);
        }
        
        this.saveProgress();
        this.renderWeeks();
        this.updateOverallProgress();
        
        // Add visual feedback
        const progressOverview = document.querySelector('.progress-overview');
        progressOverview.classList.add('progress-update');
        setTimeout(() => progressOverview.classList.remove('progress-update'), 500);
    }

    filterWeeks(category) {
        const weeks = document.querySelectorAll('.week-card');
        weeks.forEach(week => {
            if (category === 'all' || week.dataset.category === category) {
                week.classList.remove('hidden');
            } else {
                week.classList.add('hidden');
            }
        });
    }

    updateOverallProgress() {
        const totalSessions = curriculumData.reduce((total, week) => total + week.sessions.length, 0);
        const completedCount = this.completedSessions.size;
        const percentage = (completedCount / totalSessions) * 100;

        document.getElementById('overallProgress').style.width = `${percentage}%`;
        document.getElementById('overallPercentage').textContent = `${Math.round(percentage)}%`;
        document.getElementById('completedSessions').textContent = completedCount;
        document.getElementById('totalSessions').textContent = totalSessions;

        // Update current week
        const currentWeekIndex = curriculumData.findIndex(week => 
            week.sessions.some(session => !this.completedSessions.has(session.id))
        );
        document.getElementById('currentWeek').textContent = 
            currentWeekIndex >= 0 ? curriculumData[currentWeekIndex].title.split(':')[0] : 'Completed!';
    }

    saveProgress() {
        localStorage.setItem('campusx-completed-sessions', JSON.stringify([...this.completedSessions]));
    }

    resetProgress() {
        this.completedSessions.clear();
        this.saveProgress();
        this.renderWeeks();
        this.updateOverallProgress();
    }

    exportProgress() {
        const progressData = {
            completedSessions: [...this.completedSessions],
            totalSessions: curriculumData.reduce((total, week) => total + week.sessions.length, 0),
            completedCount: this.completedSessions.size,
            percentage: Math.round((this.completedSessions.size / curriculumData.reduce((total, week) => total + week.sessions.length, 0)) * 100),
            exportDate: new Date().toISOString(),
            curriculum: 'CampusX DSMP 2.0'
        };

        const blob = new Blob([JSON.stringify(progressData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `campusx-dsmp-progress-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize the progress tracker when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ProgressTracker();
});
