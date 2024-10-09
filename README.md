## Udemy Paid Course Analysis

In this repository, we attempt to compare the performance of paid and unpaid courses on the Udemy platform. 
Using Python packages such as pyplot, sklearn, and pandas, we evaluate the correlation structures of our data related to the is_paid variable.
We additionally attempt to predict whether a course is a paid course or not based on the other metrics available in our data set.

For this project, we used open-sourced data uploaded by Hossain on Kaggle, which can be found at the following link.
https://www.kaggle.com/datasets/hossaingh/udemy-courses?select=Course_info.csv

This repository contains the following files.
* Course_info.csv - The Data Set used for our analysis.
* paid_course_analysis.py - The Python script used to create our reports and figures.
* paid_course_out.txt - A copy of the print outputs from our Python script.
* paid_course_p1.png - A plot containing the data general correlation matrix of our data and boxplots showing the distribution of the is_paid variable.
* paid_course_p2.png - A plot visualizing the bivariate distributions of our data compared to is_paid and the confusion latent to our logistic regression models.
* paid_course_report.pdf - The original written report explaining the results of our analysis.

From our work, we find that Udemy's marketing strategy is unsurprisingly focused on its paid content. 
However, the improved performance of the paid courses is not distinct enough to be able to distinguish them from unpaid courses using a Logistic Regression model.
Future study should rely on more standard separation methods such as SVMs, Classification Tree methods, and Discriminant Analysis.
We question whether or not those paid observations which perform similarly to unpaid content should be changed to unpaid or have their prices lowered to increase use.
