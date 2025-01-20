# Electricity Consumption Anomaly Detection with PCA and HMMs


## Team Members

This project was developed collaboratively by the following team members:

- **Kiarash Zamani**  
- **Sahba Hajihoseini**  
- **Quang Minh Dinh**  
- **Kayin Lam Chen**  


## Overview

This project uses multivariate **Hidden Markov Models (HMMs)** and **Principal Component Analysis (PCA)** to detect anomalies in electricity consumption data. By applying PCA for feature selection and HMMs for pattern modeling, this project identifies irregularities in time-series data obtained from supervisory control systems. The project also explores the potential benefits of reinforcement learning for future anomaly detection tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Challenges and Lessons Learned](#challenges-and-lessons-learned)
6. [How to Run](#how-to-run)
7. [References](#references)

---

## Introduction

With the increasing reliance on automated control systems, detecting anomalies in electricity consumption is crucial for intrusion detection and situational awareness. This project focuses on:
- Dimensionality reduction using **PCA**.
- Modeling and anomaly detection using **HMMs**.
- Comparing classical and reinforcement learning-based approaches for anomaly detection.

## Features

- **Dimensionality Reduction**: Applied PCA to reduce complexity and identify key response variables.
- **Anomaly Detection**: Trained HMMs with varying numbers of states to find the optimal model.
- **Dataset Analysis**: Evaluated anomalies across multiple datasets.
- **Reinforcement Learning Exploration**: Discussed potential benefits over classical methods.

## Methodology

### 1. Data Preprocessing
- Used a dataset containing electricity consumption measurements over ~3 years.
- Filled missing values using interpolation to maintain time-series integrity.
- Extracted time windows for training and testing using custom functions.

### 2. Principal Component Analysis (PCA)
- Identified key features (e.g., `Global_intensity` and `Voltage`) contributing to the highest variance.
- Reduced dimensionality while retaining 75% of the dataset's information.

### 3. Hidden Markov Models (HMMs)
- Trained HMMs with different numbers of hidden states.
- Selected the best model based on **Bayesian Information Criterion (BIC)** and **log-likelihood**.

### 4. Reinforcement Learning
- Discussed the potential advantages of reinforcement learning, such as better handling of incomplete data and delayed rewards.

---

## Results

- PCA reduced the dataset complexity, retaining 75% of the information with two principal components.
- The best HMM model had **10 states**, achieving a good balance between accuracy and complexity.
- Log-likelihood results for datasets with anomalies:
  - **Dataset 1**: -210.5856
  - **Dataset 2**: -558.8563 (most anomalies)
  - **Dataset 3**: -210.5856
- Visualization of PCA and HMM results highlights the effectiveness of feature selection and state optimization.

  ![image](https://github.com/user-attachments/assets/26208d1d-5648-4b24-a0ed-73f783a9efb2)


---

## Challenges and Lessons Learned

### Challenges
- **Overfitting and Underfitting**: Initially overfit the model with too many states, then underfit by reducing states excessively.
- **Feature Selection**: Experimented with various feature combinations before selecting the optimal set.

### Lessons Learned
- PCA simplifies datasets effectively for time-series analysis.
- Selecting the right number of HMM states is critical for model accuracy.
- Reinforcement learning offers exciting opportunities for future work in anomaly detection.

---

## How to Run

### Prerequisites
- **R Programming Language**: Ensure R is installed.
- Required libraries: `depmixS4`, `ggplot2`, `dplyr`.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
