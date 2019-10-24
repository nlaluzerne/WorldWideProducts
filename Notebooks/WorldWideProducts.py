import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Constants
data_path = r'..\Data\Historical_Product_Demand.csv'

# Load the data set
data = pd.read_csv( data_path )

# Split the Date Column into 3 fields, then add new columns for Year and Month (omit day)
dates = data[ 'Date' ].str.split( '/', expand = True )
data[ 'Year' ] = dates[ 0 ]
data[ 'Month' ] = dates[ 1 ]
data = data.drop( columns = [ 'Date' ] )

# Remove lines that are missing data
data = data.dropna()

# Transform all the columns into scaled numeric ids
labelEncoder = preprocessing.LabelEncoder()
data[ 'Product_Code' ] = labelEncoder.fit_transform( data[ 'Product_Code' ] )
data[ 'Warehouse' ] = labelEncoder.fit_transform( data[ 'Warehouse' ] )
data[ 'Product_Category' ] = labelEncoder.fit_transform( data[ 'Product_Category' ] )
data[ 'Year' ] = labelEncoder.fit_transform( data[ 'Year' ] )
data[ 'Month' ] = labelEncoder.fit_transform( data[ 'Month' ] )
data[ 'Order_Demand' ] = labelEncoder.fit_transform( data[ 'Order_Demand' ] )

# Select 3 random products to forecast the Order_Demand for
products = np.random.choice( data[ 'Product_Code' ], 3, replace = False )

# Create new data frames for each product
Adata = data.loc[ data[ 'Product_Code' ] == products[ 0 ] ]
Bdata = data.loc[ data[ 'Product_Code' ] == products[ 1 ] ]
Cdata = data.loc[ data[ 'Product_Code' ] == products[ 2 ] ]

# Predict Product Demand for Product A
# Create features and labels
x = Adata[ [ 'Product_Code', 'Warehouse', 'Product_Category', 'Year', 'Month' ] ]
y = Adata[ 'Order_Demand' ]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2 )

# Create a Linear Regressor
lr = LinearRegression()

# Train the regressor and predict the Product Demand
lr.fit( x_train, y_train )
y_predict = lr.predict( x_test )

# Compute the accuracy of the prediction
score = lr.score( y_test, y_predict ) # pls help
print( 'R\u00b2 Score of Product A: {0:.1f}'.format( score ) )