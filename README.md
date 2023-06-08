# Overview

This project is part of the Titanic Challenge proposed by [Kaggle](https://www.kaggle.com/competitions/titanic/data)

It was developed for a begginer to understand and do a simple machine learning module to analyse the people who died and who survived and determine if there is a certain randomness to the people who could save themselves or if they were chosen somewhow.

Titanic was a ocean liner that transported 2243 people (crew and passengers) from England no the United States, but while passing through an iceberg area, it was hit by one and sank. The tragedy made more than 1500 victims, making it the deadliest ocean linear/cruise ship disaster in "times of peace". It had the capacity to hold 48 lifeboats, but instead was equipped with only 20 that could hold 1178 passengers, not nearly enough to save everyone abord.

# The Project
This project was divided in two parts:

## First Part
### The descritive analysis.
I started with taking a look at the raw data of the CSV files and they are divided in two: the train.csv and the test.csv.

The train.csv contains 891 passengers with all their information, while the test.csv is a smaller group of passengers (419) but it doesn't have the output 'Survived'.

The output variable gives me a 0 or 1 value saying if the passenger survived or not with a 0/1 value.

This is used to test a Machine Learning module so we can know if our code have a good accuracy. But for now, this will not be important because this corresponds to the second part of the project.

Below, I'll show you some of the descritive analysis that I did and some conclusions that I could give with the information that I have.

Remember that none of those are final and need more information that will be given by the second part of this project, involving both a deep exploratory analysis and the finalization of the Machine Learning module.

### The Data:
 - This data do not represent ALL of the passengers abord the Titanic, this is roughly 1/2 of the total.
 - There are 891 rows within the train data and 419 within the test data, with 12 variables:
     - PassengerId: identification number
     - Survived: this is the output, saying if the passenger survived the incedent or not
     - Pclass goes from 1 to 3 and it is the kind of ticket bought by the passenger (1st classe, 2nd class, and so on)
     - Name: person's name
     - Sex: person's sex
     - Age: person's age
     - Sibsp: if the person had siblings or spouses
     - Parch: number of parents or sibling within the ship
     - Ticket: ticket number
     - Fare: ticket fare paid by the passenger
     - Cabin: cabin number that the passenger stayed in
     - Embarked: what platform the passenger entered the ship

Categorical variables:
 - Survived, Sex, Embarked, Pclass

Numerical variables:
 - Age, Fare, Sibsp, Parch
 
Nominal variable:
 - Name, Ticket, Cabin
 
### Univariate Analysis

This analysis only looked at the train.csv since it has more information.

We can have a look at each of the numerical variables within the table below:

![Central tendency](Images/central_tendency_num.png 'Central Tendency')

Looking at this table we can see that:
 - Most of the variables have all of 891 values, except for the variable 'age'
 - Still about the 'age':
     - There are 177 passengers with missing age information, this will need to be filled later on
     - The minimum value that was encoutered here was 0.42, meaning that there were babies with less than 1 year old abord the ship. This will need to be compared to the output 'survived' to see if the children were prioritized.
     - The average age of the passengers aboard the ship is 29 years old, meaning that most of the passengers were adults
         - This can be confirmed by looking at the graph, there is a concentration of passengers between 20 and 40 years
         - There is a surprisingly high number of children aboard
     - The oldest person within the data is 80 years old

 - Looking at the variable 'sex':
     - There were 577 (64.76%) males and 314 (35.24%) females

 - For 'Sibsp':
     - 68.2% of the passengers did not have sibling or spouses
     - The maximun number for this variable is 8, meaning that were some families aboard

 - The same can be seen within the variable 'Parch':
     - 76.1% (678 passengers) of the people aboard did not have siblings/parents within the ship

 - For 'Fare':
     - There is a maximum fare of 512, a big discrepancy compared to the rest of the data, most of which are around 0 and 60
     - This variable showed the greated standard deviation, which shows that some passengers paid much more than the others
     - The 1.7% of values represented by 0 were probably crew members

 - For 'Embarked':
     - 644 passengers used the platform S to embark, while 168 used C and 77 used Q, some explanation for the use of different platforms:
         - Platform Q and/or C could be a 'crew only' entrance.
         - Platform Q and or C could also be reserved for high value tickets

 - For 'Survived':
     - Within the train.csv we have 38.38% of survivability

We can look at the distribution of the variables within the histogram grid below:

![Histogram](Images/histogram.png 'Histogram')

### Bivariate Analysis:
Here we need to have a look at the correlation table:

![Correlation table](Images/correlation_table.png 'Correlation Table')

And correlation matrix:

![Correlation matrix](Images/correlation_matrix.png 'Correlation Matrix')

Those two alone do not make a full bivariate analysis, but we can have a look at some correlations:
 - As expected, we can see that SibSp and Parch have a high correlation (coefficient: 0.45), both show values for siblings, and spouses/parents
 - The Pclass and Fare are highly correlatable, as the "high class" tickets are more expensive
 - Pclass and Embarked variables have a coefficient of 0.26, meaning that the gate which people embarked could be based on the type of ticket they had (this was supposed with the univariate analysis)
 - Parch and Age have a high negative correlation. It could mean that the youngest the person, higher the chance of having parents
 - The output variable 'survived' and 'sex' have a 0.54 coefficiente, meaning that one sex had more people surviving (probably male because they represent almost 65% of the crew/passengers)
 - Some numerical variables are natural numbers (1, 2, 3...), making them hard to compare (SibSp, Parch)
    - This can be improved by making scatter plots points with transparency so we can have a better look at how those variables interact

We can take a look at the pairplot generated to see the interaction of the variables:

![Pairplot](Images/pairplot.png 'Pairplot')

From we can assume some things:
 - Those who paid much more than the other passengers, survived. This can be seein within the comparison of the variables 'Fare' and 'Survived'
 - 'Age' doens't seem to correlate with the output 'Survived'.
 - There is a correlation with people who embarked in the third platform and survided.
 - PassengerId is just a identification number and the comparisons generated with the other variables don't indicate anything.
 - The 'Sibsp' and 'Parch' indicate that most of the families had between 1 and 4 members, but some can be seen having 8 members.

The correlation matrix and table, as well as other comparisons can be seem within this [report](Images/report.html) here.

## Second Part
This involves a Machine Learning module.

To start it I cleaned the data, removing some variables judged non valuables, like the PassengerId, Cabin, Ticket and Name.

After that, it was necessary to fill the empty cells, so I used the median to do the inputation.

*I understand that this is not the best way to make an inputation*, but this is not the objective of the project and it is listed to be developed in future projects, but this model was made to be simple and viable.

I also needed to fill the Embarked variable, and for the empty cells I chose 'U' to repersent the 'unknown'.

The next step was to transform those categories ('Sex' and 'Embarked') in numeric values. That is why I used a loop function with a .fit_transform() command.
 - This made 'male' and 'female' 1 and 0 respectively.
 - The embarked signs ('C', 'Q', 'S', 'U') became 0, 1, 2 and 3.

The next image shows how it turned out:

![Data Clean](Images/data_clean.png 'Data Clean')

I than chose a Logistic Regression as the module for this machile learning. I chose this because it can deal with boolean variables and the interaction between the variables doesn't need to be linear, so it has more room to work.

The data was than divided into 75/25 for train and test respectively. I separated the 'Survived' column from the test data so I could compare to the results of the code later.

After the calculations, the model ended with a 84% accuracy, which is good and bad:
 - Comparing with the work I could do by hand, 84% is a really good result.
 - But, in a real life situation, the prediction of who lives and dies has a heavy weight and cannot be measured.
 
The output variables were stored within a csv file that now contains a 'Output' column, indicating the results obtained by the code, a 'Survived' column that is the real occurancy and a 'Results Comparison', indicating with a true or false if the algorithm got hte right results.

Everything can be seen in the file [results](results.csv)