
#### **project title:**

## **Predicting Stock Market Trends Using Historical Price a**Dat

#### Project Overview

This project explores whether machine learning models can identify short-term patters in historical stock
price and trading volume data to predict **next-day market direction (Up or Down).** Using S&P 500 index
data, we evaluate whether simple classification models can outperform random guessing and provide
meaningful predicitive insight in highly financial data.

This repository contains a series of Jupyter notebooks corresponding to weekly project milestones, 
which include EDA, pre-processing and feature engineering, Initial modeling, final modeling,
tuning, and evaluation and Interpretation.a**

#### Brief background

The stock market is heavily influenced by many factors like a company’s performance, investor
behavior and other economic conditions. Even though the market is unpredictable, traders rely
on trends to make decisions on investments. We want to determine whether machine learning can
identify short-term patterns to help predict if stock prices will increase or decrease the next 
day based on recent treResearch nds.

#### Question

Can we predict next-day up and down movement of S&amp;P 500 stocks using historical price and trade volume data?

#### Hypothesis & Predictions

## **Hypothesis**:

If recent stock trends and trading volumes show consistent patterns, then simple classification
models like logistic regression or random forest can predict next-day stock movement more
accurately than random guessing.


## **Prediction**:

We expect that our model will correctly predict stock direction 55-60% of the time. This is
slightly better than chance but will prove that short-term patterns contain usable information. Our
hope is that m the model’s overall accuracy.ore historic data will improve

## Data Source
- S&P 500 Historical data retrieved from Yahoo Finance via the yfinance API
- Date Range for current Interpretation 2010 - 2025
- Data Range for future is continuous
- Daily observations Including:
    - OHLC prices
    - Trading Volume
    - Date Index

## Project Structure

- **Week 1 -Preliminary Project Proposal**
    - Proposal of a project idea and why it is important
- **Week 2 - Finalized project proposal**
    - Defining the research question, motivation, and explaining why its worth while.
- **Week 3**
    - EDA Report: Exploratory data analysis, time series visualization, correlation assessments,
      distribution analysis.
- **Week 4**
    - Pre-processing and Feature Engineering: Data cleaning, feature engineering, volatility measures,
      constructing the target variable
- **Week 5**
    - Pre-Processing and Initial models: Train/test splits, baseline modeling, and initial performance evaluation.
- **Week 6**
    - Model Tuning, Validation, Exploration: Advanced Evaluation metrics, Accuracy analysis, model comparisons,
      and Interpretation
- **Week 7**
    - Finalized Model: Final model tuning, ROC/AUC analysis, model comparisons, best model selection
      
- **Week 8 - Presnetation to Stakeholders and Final Write-Up**

## Custom Functions

This project includes several helper functions to improve code clarity and reproducibility:

- Reusable plotting and comparison logic
- Model Evaluation utilities for consistent metric reporting

Usages examples are directly in each jupyter notebook.

## Design Decision

For the Model Evaluation & Interpretation and Model Evaluation& Interpretation(V2) we intentionally
kept most evaluation and plotting inside the notebooks rather than offloading large portions into 
separate .py files. This is done to:

- Maximize Reproducibility
- Preserve analysis - specifically making it easier to follow

## Reproducibility & Testing

- All notebooks were run end-to-end locally without 

## Requirement Files

- **requirements.txt**

    - A dependency list containing only core libraries required to run analysis and models
      -> This is what most users will need.

- **requirements_full.txt**

    - A complete enviornment snapshot generated via 'pip freeze.' Includes Jupyter and
      development related dependencieserrors by the primary authors.

## Authors & Contributions

- Auriana Anderson - Data preparation, modeling, evaluation, documentation, Review, Testing, Feedback

- Ross Schanck - Data preparation, modeling, evaluation, documentation, Review, Testing, Feedback

- Chase Golden - Data preparation, modeling, evaluation, documentat the model’s overall accuracy.
