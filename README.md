# Titanic Survival Prediction

## Overview
A machine learning pipeline predicting Titanic passenger survival, built in two iterations to apply progressively better ML practices.

## Version History

**v1** — Basic pipeline: single-model RandomForest, simple median/mode 
imputation, manual label encoding, single train/validation split.
Result: ~10,000th position on Kaggle leaderboard.

**v2** — Improved pipeline: engineered Title, HasCabin, and IsAlone 
features; group-based imputation (median by Title/Pclass instead of 
one global median); compared 4 models via 5-fold cross-validation; 
combined top models into a soft-voting ensemble.
Result: ~1,000th position on Kaggle leaderboard.

Interestingly, v1's single train/validation split reported a higher accuracy (0.88) than v2's cross-validated score (0.83). This illustrates exactly why cross-validation matters: a single split can give a misleadingly optimistic number due to chance, while cross-validation provides a more honest estimate. Despite the lower-looking validation score, v2 performed dramatically better on Kaggle's actual hidden test set (~1,000th vs ~10,000th), confirming that the cross-validated model generalized better.

## What Changed and Why

**Age imputation** — v1 used a single overall median to fill missing ages. 
v2 fills by median within each Title group (Mr/Mrs/Miss/Master), since a 
"Master" (young boy) and a "Mr" (adult man) have very different typical ages.

**Title extraction** — v1 dropped the Name column entirely. v2 extracts the 
title (Mr, Mrs, Miss, Master, etc.) as a new feature, capturing age group, 
marital status, and social class signals that raw text couldn't provide.

**Fare and Cabin imputation** — v1 used a single overall median for Fare. 
v2 fills both Fare and Cabin by median within each Pclass group, since fare 
and cabin deck are both strongly tied to passenger class.

**HasCabin feature** — v1 didn't use Cabin at all. v2 adds a binary flag 
for whether a cabin was originally recorded, since ~77% of Cabin values 
are missing and the fact that one *was* recorded may itself be meaningful.

**IsAlone feature** — v1 only had FamilySize. v2 adds a binary flag for 
traveling completely alone, since solo travelers may have meaningfully 
different survival patterns than those with even one family member aboard.

**Model comparison** — v1 only tried RandomForest with a single train/
validation split. v2 compares LogisticRegression, RandomForest, XGBoost, 
and GradientBoosting using 5-fold cross-validation, giving a more reliable 
performance estimate than a single split.

**Ensemble** — v2 adds a soft-voting ensemble combining LogisticRegression, 
RandomForest, and XGBoost, since different models make different mistakes 
and averaging their predictions often produces a more robust result.

## Tech stack
Python, Pandas, scikit-learn, XGBoost

## What I learned
[your own honest 2-3 sentences here]
