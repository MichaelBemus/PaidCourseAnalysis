# Final Project.
# Coded this all in Jupyter, then transferred over here. This time, outputting 2 sets of plots is intentional.

# Importing Everything
import pandas as pd  # For Data Handling
from matplotlib import pyplot as plt  # For Graphs
import seaborn as sn  # More Graphs
import numpy as np  # For sqrt.
from scipy import stats  # For t-dist.
# Everything else is for linear model.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Function to plot dist of is_paid compared to another variable.
def plot_vs_paid(ind):  # Input is the variable's index in columns list
    # Create a violin plot using our data.
    vp = plt.violinplot([df[cols[ind]][df["is_paid"] == 0],
                         df[cols[ind]][df["is_paid"] == 1]])

    # Change the violin plot aesthetics.
    for i, j in zip(vp['bodies'], ["#FF00FF", "#00FFFF"]):
        i.set_facecolor(j)   # Make the Unpaid obs Red and the Paid obs Green.
        i.set_edgecolor("#8080FF")   # Draw an outer edge to our plots.

    sub.set_title(cols[ind] + " v. is_paid")  # Title as specified by variable name.
    sub.set_yticks([0, 1])  # only need binary 0 or 1.


# Function to calculate t-statistic.
def get_t(col):
    # Split column into paid (a) vs unpaid (b)
    a = df.loc[df['is_paid'] == 1][col]
    b = df.loc[df['is_paid'] == 0][col]

    # Calculate variances.
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    # Combine variances to get standard deviation.
    s = np.sqrt(var_a / len(a) + var_b / len(b))

    # Calculate the combined t-statistic
    t = (a.mean() - b.mean()) / s
    return t


# Function to print t-statistic and p-value.
def t_and_p(col, md=0):  # Function defaults to calulating for paid. md = 1 calculates for unpaid.

    # Running above function for t-statistic.
    if md == 1:  # Unpaid.
        t = -get_t(col)
    else:  # Paid.
        t = get_t(col)

    # Calculating p using CDF.
    p = 1 - stats.t.cdf(t, df=len(df.index)-2)

    # Printing T-Statistic.
    print("T-Statistic of", col + ":\t", t)

    # Printing P-Value. If Statement used for tabbing.
    if col == "num_reviews" or col == "avg_rating":
        print("P-Value of", col + ":\t\t", p)
    else:
        print("P-Value of", col + ":\t", p)
    print("")


# Function to output fit for logistic regression.
def fit_results(trts):
    # Determine which data to use.
    if trts == "train":
        x = X_tr
        y = y_tr
    elif trts == "test":
        x = X_ts
        y = y_ts
    else:
        return "Invalid Mode"

    # Create predictions based on selected X.
    pred = lr.predict(x)

    # Outputting results.
    print("Logistic Regression " + trts + " Results")
    print('Model Score on ' + trts + ':', lr.score(x, y))
    print(classification_report(y, pred))

    # Returns confusion_matrix to be used later.
    return confusion_matrix(y, pred)


# Function to plot confusion matrices.
def plot_confusion(confus):  # Input confusion matrix.
    sn.heatmap(confus, annot=True, fmt=".0f")
    plt.title("Confusion")
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")


# Importing full Data Frame.
df_path = input('Please enter the file path to "Course_info.csv":\n')
full = pd.read_csv(df_path)

# Selecting specific columns to work with.
df = full[['is_paid',
           'num_subscribers',
           'avg_rating',
           'num_reviews',
           'num_comments',
           'num_lectures',
           'content_length_min']]

# Checking for NA values in selected columns.
na_vals = df[df.isnull().any(axis=1)]

# Printing any NA's.
print(na_vals.head())
# We don't find any. Yay!

# Getting correlation matrix for graphing.
course_corr = df.corr()

# Listing column names for convenient iteration.
cols = ['num_subscribers', 'num_reviews', 'num_comments', 'avg_rating']

# Creating figure to graph on.
fig = plt.figure(figsize=(8, 8))

# Adjusting margins.
fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)

# List of values to iterate through to make graph.
X = [(3, 3, (1, 5), 'hello', 'there'),  # 'hello' and 'there' are arbitrary.
     (3, 3, 3, 0, 'g'),  # Dim1, Dim2, Plot Number, col index, color.
     (3, 3, 6, 1, 'r'),
     (3, 3, 7, 2, 'b'),
     (3, 3, 8, 3, 'k'),
     (3, 3, 9, 'oh dear', 'me')]  # 'oh dear' and 'me' are arbitrary.

# Iterating...
for nrows, ncols, plot_number, ind, col in X:
    # Selects subplot to graph on.
    sub = plt.subplot(nrows, ncols, plot_number)

    # Identifying correlation matrix.
    if ind == 'hello':
        # Graph correlation.
        sn.heatmap(course_corr, annot=True)
        sub.set_title('Correlation Matrix')
        sub.set_xticks([])  # x-labels too messy for subplot. Same as y-axis anyways.

    # Identifying comment box.
    elif ind == 'oh dear':
        # Just an empty graph.
        sub.set_xticks([])
        sub.set_yticks([])
        # Annotate in comment. Got coordinates right first try.
        sub.annotate('Low Correlations w/ is_paid.', (0.075, 0.5))

    # All other plots are defined as above.
    else:
        plot_vs_paid(ind)

# Display graph.
plt.savefig(df_path[:-4] + "_p1.png")
plt.show()

# Outputting test results.
# Comparing Subscribers.
print('1a.\tNull Hypothesis: Paid Courses have more subscribers.' +
      '\n\tAlternative Hypothesis: Paid Courses have fewer subscribers.')
t_and_p('num_subscribers')

# Comparing Reviews.
print('1b.\tNull Hypothesis: Paid Courses have more reviews.' +
      '\n\tAlternative Hypothesis: Paid Courses have fewer reviews.')
t_and_p('num_reviews')

# Comparing Comments.
print('1c.\tNull Hypothesis: Paid Courses have more comments.' +
      '\n\tAlternative Hypothesis: Paid Courses have fewer comments.')
t_and_p('num_comments')

# Comparing Ratings.
print('2.\tNull Hypothesis: Unpaid Courses have higher average ratings.' +
      '\n\tAlternative Hypothesis: Unpaid Courses have lower average ratings.')
t_and_p('avg_rating', 1)

# Conclusions:
print("As shown, for Hypothesis Set 1, under a 0.05 Alpha-Level, we fail to reject the null hypothesis. " +
      "In all metrics, paid courses see more user traffic.\n")
print("For Hypothesis 2, under the same 0.05 Alpha-Level, we reject the null hypothesis. " +
      "Unpaid courses see significantly lower average ratings than paid courses.\n")

# Logistic Regression!

# Dividing variables into X and y.
X = df[['num_subscribers', 'num_reviews', 'num_comments', 'avg_rating']]
y = df['is_paid']

# Splitting into training and testing data.
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, random_state=3)

# Creating model. 3 is arbitrary.
lr = LogisticRegression(random_state=3)

# Fitting model to data.
lr.fit(X_tr, y_tr)

# Outputting parameters (logit)
print('Logistic Regression Parameters:\n\tIntercept:', lr.intercept_[0])
for i in range(0, len(lr.coef_[0])):
    print('\tBeta ' + str(i + 1) + ':', lr.coef_[0][i])
print("")

# Running fit tests. Storing Confusion Matrices
confus1 = fit_results("train")
confus2 = fit_results("test")

# Create figure.
plt.figure(figsize=(9, 11))  # Works better in Jupyter. Axis titles keep overlapping here.

# Plotting Confusion Matrices
plt.subplot(223)  # Bottom left.
plot_confusion(confus1)

plt.subplot(224)  # Bottom right.
plot_confusion(confus2)

# More plots to display distribution of paid vs. unpaid.
plt.subplot(221)  # Top Left.
# Gray = Paid. Green = Unpaid.
plt.scatter(df['num_reviews'], df['avg_rating'], s=10, c=df['is_paid'], cmap='Accent', alpha=0.5)
plt.title("Paid: Reviews and Rating Plot")
plt.xlabel("Number of Reviews")
plt.ylabel("Average Rating")

plt.subplot(222)  # Top Right.
plt.scatter(df['num_comments'], df['num_subscribers'], s=10, c=df['is_paid'], cmap='Accent', alpha=0.5)
plt.title("Paid: Comments and Subscribers Plot")
plt.xlabel("Number of Comments")
plt.ylabel("Number of Subscribers")

# Displaying Plots.
plt.savefig(df_path[:-4] + "_p2.png")
plt.show()
