# NutriRate AI

### Advanced Machine Learning System for Nutritional Quality Classification

NutriRate AI is a machine learning-based nutritional classification system that predicts the Nutri-Score grade of a food product from its nutritional information.

The project treats Nutri-Score prediction as a five-class supervised classification problem with grades:

**A, B, C, D, and E**

The system uses nutritional attributes such as energy, fat, saturated fat, carbohydrates, sugars, protein, fiber, and salt, together with engineered nutritional ratio features.

A large-scale dataset containing **849,969 food products** was used for model development and evaluation.

The final Random Forest model achieved:

- **88.85% Test Accuracy**
- **0.8751 Test Macro F1**
- **0.8890 Test Weighted F1**

A Streamlit application is included to provide an interactive interface for predicting the Nutri-Score grade from nutritional information.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Experimental Results](#experimental-results)
- [Error Analysis](#error-analysis)
- [Feature Importance](#feature-importance)
- [Application](#application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Running the ML Pipeline](#running-the-ml-pipeline)
- [Results and Visualizations](#results-and-visualizations)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Technology Stack](#technology-stack)
- [Academic Context](#academic-context)
- [License](#license)

---

# Overview

Food products contain several nutritional attributes that can be difficult to interpret simultaneously.

NutriRate AI explores whether machine learning can learn the relationship between nutritional composition and Nutri-Score categories.

The project follows a complete machine learning workflow:

```text
Raw Nutritional Data
        ↓
Data Cleaning & Validation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Stratified Train/Validation/Test Split
        ↓
Model Training
        ↓
Model Comparison
        ↓
Error Analysis
        ↓
Feature Importance
        ↓
Final Model
        ↓
Streamlit Application