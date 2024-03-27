#!/usr/bin/env python
# coding: utf-8

# # NUMPY

# In[1]:


# program to create array from numpy library
import numpy as np


# In[2]:


# creating an array of integers.
myarray=np.array([1,2,3,4,5,6,7,8,9,10],int)
print("The array of integers is : \n",myarray)


# In[3]:


#creating an array of characters.
myarray1=np.array(['a','b','c','d','e'],str)
print("The array of the characters is : \n",myarray1)


# In[4]:


# creating an array of strings 
myarray2=np.array(['amity_noida','amity_haryana','amity_kolkata','amity_mumbai','amity_dubai'],str)
print("The array of the strings is : \n",myarray2)


# In[5]:


# creating a float array
myarray3=np.array([12.2,12.3,12.4,34.5],float)
print("The array of the float is : \n",myarray3)


# In[6]:


# creating an array using view() function from an existing array.
myarray4=myarray2.view()
print("Using view function to create a new array from an existing array is : \n",myarray4)


# In[7]:


# using copy() function to create a new array from an existing array.
myarray5=myarray3.copy()
print("using copy function to create a new array from an existing array is : \n",myarray5)


# In[8]:


n=np.array([1,"guru",2.3])  # yaha dekho array waise toh same data type contain karta hai par agar usme multiple data type hai toh vo sabhi ko ek hi data type mai convert kar dega, jo convertable ho 
print(n)
print(type(n))


# In[9]:


# creating an integer array using arange() function.
myarray6=np.arange(1,11)
print("The array of the integers using arange() function is : \n",myarray6)


# In[10]:


# creating an float array using arange() function.
myarray7=np.arange(14.56,40.89)
print("The array of the float using arange() function is : \n",myarray7)


# In[11]:


# creating an integers array using linspace() function for equal spacing.
myarray8=np.linspace(1,100,25)
print("The array of the integers using linspace() function is : \n",myarray8)


# In[12]:


# creating an array using zeros(shape) function.
myarray9=np.zeros((10))
print("The array using zeros(shape) function is : \n",myarray9)


# In[13]:


# creating an array using ones(shape) function.
myarray10=np.ones((10))
print("The array using ones(shape) function is : \n",myarray10)


# # PROGRAM FOR USING FUNCTIONS ON NUMPY ARRAY.

# In[14]:


import numpy as np


# In[15]:


myarray11=np.array([12,23,34,45,56,67,78,89,90,11,22,33,44,55,66,77,88,99,100],int)
print("original array is : \n",myarray11)


# In[16]:


# Modifying an element at a specified index.
myarray11[1]=30
print("modified array is : \n",myarray11)


# In[17]:


# DETERMINING THE CHARACTERISTICS OF THE ARRAY.

# The size returns the size of the array.
print("The size of the array is : \n",myarray11.size)


# In[18]:


# The dtype returns the data type of the array.
print("The data type of the array is : \n",myarray11.dtype)


# FUNCTIONS RETURNING ONE SINGLE NUMERICAL OUTPUT.

# In[19]:


# The mean() function return the mean of all the array elements.
print("The mean of the array is : \n",myarray11.mean())
print("The mean of the array is : \n",np.mean(myarray11))


# In[20]:


# The median() function return the median of all the array elements.
print("The median of the array is : \n",np.median(myarray11))


# In[21]:


# The min() function return the minimum element of all the array elements.
print("The minimum element of the array is : \n",myarray11.min())
print("The minimum element of the array is : \n",np.min(myarray11))


# In[22]:


# The max() function return the maximum element of all the array elements.
print("The maximum element of the array is : \n",myarray11.max())
print("The maximum element of the array is : \n",np.max(myarray11))


# In[23]:


# The sum() function return the sum of all the array elements.
print("The sum of element of the array is : \n",myarray11.sum())
print("The sum of element of the array is : \n",np.sum(myarray11))


# In[24]:


# The prod() function return the product of all the array elements.
print("The product element of the array's element is : \n",myarray11.prod())
print("The product of the array's element is : \n",np.prod(myarray11))


# In[25]:


# The var() function return the variance of all the array elements.
print("The variance of all the array elements is : \n",myarray11.var())
print("The variance of all the array elements is : \n",np.var(myarray11))


# In[26]:


# The cov() function return the co-variance of all the array elements.
print("The co-variance of all the array elements is : \n",np.cov(myarray11))


# In[27]:


# The std() function return the standard deviation of all the array elements.
print("The standard deviation of all the array elements is : \n",myarray11.std())
print("The standard deviation of all the array elements is : \n",np.std(myarray11))


# MATHEMATICAL FUCNTIONS RETURN AN ARRAY OUTPUT (ALL ELEMENTS).
# 

# In[28]:


# The sort() function returns the elements in a sorted order
print("The sorted array is : \n",np.sort(myarray11))


# In[29]:


# The power() function returns all elements raised to power of a number.
print("Power raised to element (2) of array : \n",np.power(myarray11,2))


# In[30]:


# The sqrt() function rerurns the square root of alla elements
print("Square root of the array elements is : \n",np.sqrt(myarray11))


# In[31]:


# abs() function returns the absolute value of all the elements.
print("Absolute value of the array elements is : \n",np.abs(myarray11))


# TRIGONOMETRIC FUNCTIONS RETURN AN ARRAY OUTPUT (ALL OUTPUT) 

# In[32]:


# The sin() , cos() , tan() , functions returns the sine , cosine , tan value of all the elements.
print("Sin value of the array elements is : \n",np.sin(myarray11))
print("Cos value of the array elements is : \n",np.cos(myarray11))
print("Tan value of the array elements is : \n",np.tan(myarray11))
print("cosec value of the array elements is : \n",1/np.sin(myarray11))
print("sec value of the array elements is : \n",1/np.cos(myarray11))
print("cot value of the array elements is : \n",1/np.tan(myarray11))


# LOGARITHMIC / EXPONENTIAL FUNCTIONS RETURN AN ARRAY OUTPUT 

# In[33]:


# The log() function returns log with exponential base of all elements
print("Log of all elements with 'e' base is : \n",np.log(myarray11))


# In[34]:


# The log10() function returns log with base 10 of all elements
print("Log of all elements with base 10 is : \n",np.log10(myarray11))


# In[35]:


# The exp() function returns expoenetial value of all the elements
print("Exponential of the array elements is : \n",np.exp(myarray11))


# # MATHEMATICAL OPERATORS FOR 1-D ARRAY

# In[36]:


# Mathematical operations on one dimensional array.
myarray12=np.arange(25,34)
print("Original array is : \n",myarray12)
print("Adding an element to array element is : \n",myarray12+45)
print("subtracting an element to array element is : \n",myarray12-10)
print("Multiplyiin an element to array element is : \n",myarray12*3)
print("Dividing an element to array element is : \n",myarray12/4)
print("Using module operator an element to array element is : \n",myarray12%4)
print("Using integer operator an element to array element is : \n",myarray12//4)


# In[37]:


# Program to use mathematical operators on multiple arrays

myarray13=np.array([11,22,33],int)
myarray14=np.array([40,50,60],int)


print("Adding an array element is : \n",myarray13+myarray14)
print("subtracting an array element is : \n",myarray13-myarray14)
print("Multiplying an array element is : \n",myarray13*myarray14)
print("Dividing an array element is : \n",myarray13/myarray14)
print("Using module operator to array element is : \n",myarray13%myarray14)
print("Using integer operator to array element is : \n",myarray13//myarray14)


# In[38]:


# Program to use Relational operators on array.
myarray13=np.array([11,22,33],int)
myarray14=np.array([40,50,60],int)


print("Adding an array element is : \n",myarray13==myarray14)
print("subtracting an array element is : \n",myarray13>=myarray14)
print("Multiplying an array element is : \n",myarray13<=myarray14)
print("Dividing an array element is : \n",myarray13>myarray14)
print("Using module operator to array element is : \n",myarray13<myarray14)
print("Using integer operator to array element is : \n",myarray13!=myarray14)


# # MULTIDIMENSIONAL ARRAYS

# In[39]:


# Program to show multidimensional array.
import numpy as np


# In[40]:


# creating a two-dimensional array using array() function.
multiarray=np.array([[100,200,300,400,500],[600,700,800,900,1000]])
print("Multidimensioanl array is : \n",multiarray)


# In[41]:


# creating a two dimensional array using matrix() function.
multiarray1=np.matrix('11 22;33 44;55 66;77 88')
multiarray2=np.matrix('12 23 33 44;34 45 32 43;56 67 65 76;61 52 31 21')
print("Multidimensional array using matrix function is : \n",multiarray1,'\n')
print("Multidimensional array using matrix function is : \n",multiarray2)


# In[42]:


# creating a two-dimensional array using reshape() function.
multiarray3=np.reshape(myarray,(5,2))
multiarray4=np.reshape(myarray,(2,5))
print("Multidimensional array using reshape is : \n",multiarray3,'\n')
print("Multidimensional array using reshape is : \n",multiarray4,'\n')


# In[43]:


# creating a multidimensional array using zeros((a,b)) and ones((a,b)) function.
multiarray5=np.zeros((5,5))
print("Multidimensional array using zeros is : \n",multiarray5,'\n')

multiarray6=np.ones((5,5))
print("Multidimensional array using ones is : \n",multiarray6)


# Accessing Elements in Multi-Dimensional Array

# In[44]:


# Programming to show indexing in multidimensional arrays.
multiarray7=np.array([[100,200,300,400,500],[600,700,800,900,1000],[123,234,345,456,467],[454,765,987,432,234]])
print("The array is : \n",multiarray7)


# In[45]:


# Performing indexing.
print("Element of first row and fourth coulmn is : \n",multiarray7[0][3],'\n')
print("second row and fourth coulmn is : \n",multiarray7[1][3])


# In[46]:


# Program to show slicing in multidimensional array.
print("The multidimensional array is : \n",multiarray7[:,:])


# In[47]:


print("first , second row and all columns is : \n",multiarray7[0:2,:])


# In[48]:


print("first , second row and odd columns is : \n",multiarray7[0:2,::2])


# In[49]:


print("first , last row and all columns is : \n",multiarray7[0::3,:])


# In[50]:


print("the element of (3,3) is : \n",multiarray7[3,3])


# In[51]:


print("array is : \n",multiarray7[1:3,1:4]) # array ke bich me se ek chota sa array extract kiya hai bas or kuch nahi.


# Functions on Multi-Dimensional Array

# In[52]:


multiarray7


# In[53]:


# The transpose() function returns the transpose of a matrix.
print("Transposed matrix is : \n",multiarray7.T)


# In[54]:


# The ndim determines the dimension of the array.
print("Dimension of the matrix is : \n",multiarray7.ndim)


# In[55]:


# The flatten() flattens matrix and converts into one-dimensional
print("Flattened matrix is : \n",multiarray7.flatten())


# In[56]:


# The sort() function with axis=1(default) and axis=0 sort the matrix on row basis and on column basis.
print("Sorted matrix on row basis : \n",np.sort(multiarray7,axis=1),'\n')
print("Sorted matrix on column basis : \n",np.sort(multiarray7,axis=0))


# In[57]:


# The diagonal() functions helps to determine the diagonal elements.
print("Diagonal elements are : \n",np.diagonal(multiarray7))


# # PANDAS LIBRARY

# Program for creating series using series() function of pandas library.

# In[58]:


import pandas as pd


# In[59]:


# creating a list of prices.
pricelist=[100,200,300,400,500,600]

#creating a series
productseries=pd.Series(pricelist, index=['pen','shirt','book','mouse','keyboard','monitor'])

#displaying the series
print(productseries)


# CREATING A DATAFRAME

# In[60]:


# syntax------DataFrame(Data,columns=list of columns names)


# In[61]:


df=pd.DataFrame([[123,456,789,741],[452,856,365,154]],columns=['monitor','cpu','laptop','mobile'],index=['sale of 2023','sale of 2024'])
print("Dataframe of product is : \n",df)


# In[62]:


# Displaying the dimensions of the DataFrame using shape.
print("Dimension of the dataframe is : \n",df.shape)


# In[63]:


# Displaying the size of dataframe using size.
print("Size of the dataframe is : \n",df.size)


# In[64]:


# Displaying the name of the columns ,index of the dataframe using keys(),columns and index function.
print("names of the columns of the dataframe is : \n",df.keys(),'\n')
print("names of the columns of the dataframe is : \n",df.columns,'\n')
print("names of the index of the dataframe is : \n",df.index)


# Adding Rows and Columns to the DataFrame

# In[65]:


# Program for adding rows and columns to the data frame.


# In[66]:


# creating a new dataframe 
df1=pd.DataFrame([[12,23,34,45],[56,67,78,89]],columns=['pen','eraser','scale','pencil'])
print("New data frame is : \n",df1,'\n')
print("Dimension of the dataframe is : \n",df1.size)


# In[67]:


#Adding rows to the dataframe by adding other dataframe.
df2=df.append(df1)
print(df2,'\n')
print("dimension of the bew dataframe is : \n",df2.shape)


# In[68]:


df2


# In[69]:


# Adding column named "mobile" to the dataframe.
df2['mobile']=[11,22,33,44]


# In[70]:


df2


# In[71]:


# Displaying the size of the new dataframe using size.
print("Size of the new dataframe is : \n",df2.size)


# ADDING ROWS AND COLUMNS FROM THE DATAFRAME.

# In[72]:


#Program for deleting rows and columns from the data frame.


# In[73]:


# Deleting multiple columns from data frame.
df3=df2.drop(columns=['pencil','eraser'])
print("dataframe after deleting columns is : \n",df3,'\n')
print("dimension of the dataframe after deleting columns from the dataframe is : \n",df3.shape)


# In[74]:


# Deleting ROWS from the dataframe
df4=df2.drop(index=[0,'sale of 2024'])
print("dataframe after deleting rows from the dataframe is : \n",df4,'\n')
print("dimension of the dataframe after deleting rows from the dataframe is : \n",df4.shape,'\n')
print("size of the new dataframe df4 is : \n",df4.size)


# IMPORT OF DATA

# In[75]:


# importing "csv" file and storing in data frame.
liver=pd.read_csv("ILPD.csv")


# In[76]:


# displaying the csv file
liver


# In[77]:


# determining the size and shape of the dataset.
print("size of the dataset is : \n",liver.size,'\n')
print("shape of the dataset is : \n",liver.shape)


# In[78]:


# determinig the columns and rows of the dataset.
print("columns of the datset is : \n",liver.keys(),'\n')
print("columns of the dataset is : \n",liver.columns,'\n')
print("row are : '\n'",liver.index)


# Functions of the Dataframe

# In[79]:


# Displaying the complete information of the dataset using info() function.
print("Information of the dataset is : \n",liver.info())


# In[80]:


# Displaying the complete dataset using describe function.
print("Details of the dataset is : \n",liver.describe,'\n')

# Displaying the descriptive statistical values of the numerical columns using describe() function.
print("Description of the dataset is : \n",liver.describe())


# In[81]:


# Displaying the first and last records from dataset -using head() and tail() function.
print("First records of the dataset are : \n",liver.head(),'\n')
print("last records pf the datset are : \n",liver.tail())


# In[82]:


# Determining all the values of Alkphos column only.
print("Values for Alkphos column are : \n",liver['alkphos'].values)


# In[83]:


liver.columns


# In[84]:


# Program to use different functions for specified column.


# In[85]:


# Determine number of records based on gender in percentage form.
print("number of records for gender : \n",liver['gender'].value_counts())


# In[86]:


# Determine number of records based on gender in percentage.
print("number of records based on gender in percentage : \n",liver['gender'].value_counts()/len(liver['gender']))


# In[87]:


# Descriptive stats of 'TB' column
print("descriptive stats of 'tot_bilirubin' : \n",liver['tot_bilirubin'].describe())


# In[88]:


# Program to converting data frame to a list.
agelist=liver['age'].tolist()
print("list of age : \n",agelist)


# MATHEMATICAL AND STATISTICAL FUNCTIONS

# In[89]:


# Program to use Mathematical and Statistical functions on filtered data.


# In[90]:


# Determining mean of Age column using mean() function.
print("Mean of age is : \n",liver['age'].mean())


# In[91]:


# Determining median of age column using median() function
print("Median of age is : \n",liver['age'].median())


# In[92]:


# Determining Minimum of alkphos using min() function.
print("minimum of alkphos is : \n",liver['alkphos'].min())


# In[93]:


# Determining Maximum of alkphos using max() function.
print("maximum of alkphos is : \n",liver['alkphos'].max())


# In[94]:


# Determining sum of 'tot_bilirubin' column sum() function.
print("sum of 'tot_bilirubin' is : \n",liver['tot_bilirubin'].sum())


# In[95]:


# Determining product of 'tot_bilirubin' column prod() function.
print("product of 'tot_bilirubin' is : \n",liver['tot_bilirubin'].prod())


# In[96]:


# Determining n smallest values of 'tot_bilirubin' using nsmallest() function.
print("smallest values of 'tot_bilirubin' is : \n" ,liver['tot_bilirubin'].nsmallest().head())


# In[97]:


# Determining n largest values of 'tot_bilirubin' using nlargest() function.
print("largest values of 'tot_bilirubin' is : \n" ,liver['tot_bilirubin'].nlargest().head())


# SORT FUNCTIONS

# In[98]:


# Sorting in descending order.
print("sorting of 'tot_proteins' in descending order is : \n",liver.sort_values(by='tot_proteins',ascending=False))


# In[99]:


liver.sort_values(by='tot_proteins') # without descending order 


# DATA EXTRACTION

# In[100]:


# Displaying the records where gender is male.
male_data=liver[liver['gender']=='Male']
print("male_data is : \n",male_data.head(7))


# In[101]:


# Displaying the records where age is <=50.
age_data=liver[liver['age']>=50]
print("male_data is : \n",age_data.head(3))


# USING------ iloc ------INDEXERS

# In[102]:


# Displaying single column of single row. (begrow:endrow , begcol:endcol)
print("Third column of sixth record : \n",liver.iloc[5,2])


# In[103]:


# Displaying specific column of range of rows.
print("Range of records for sixth column : \n",liver.iloc[7:9,5])


# USING----loc----INDEXERS

# In[104]:


# Retrieving different multiple rows by loc method.
print("Displaying multiple specified records : \n",liver.loc[0:6,::2])


# In[105]:


# Retrieving different multiple rows by loc method.
print("Displaying multiple specified records : \n",liver.loc[0:6,'age':'sgot':2])


# In[106]:


# Retrieving different multiple rows by loc method.
print("Displaying multiple specified records : \n",liver.loc[0:6,'age':'sgot'])


# groupby FUNCTIONALITY

# In[107]:


# Using groupby() to group records on basis of categorical variable.


# In[108]:


# Count the number of records on the basis of Gender using groupby() function.
print("Number of records based on different gender are : \n",liver['gender'].groupby(liver['gender']).count())


# In[109]:


# Grouping on basis of "Gender" and using sum() function for 'tot_bilirubin'.
print("Grouping of observations on basis of Gender and calculating sum of tot_bilirubin : \n",liver['tot_bilirubin'].groupby(liver['gender']).sum())


# In[110]:


# Grouping on basis of "is_patient" and using min() function for 'direct_bilirubin'
print("Grouping on basis of 'is_patient' and calculating minimum of 'direct_bilirubin' is : \n",liver['direct_bilirubin'].groupby(liver['is_patient']).min())


# In[111]:


# Grouping on basis of "is_patient" and using min() function for 'albumin '
print("Grouping on basis of 'is_patient' and calculating minimum of 'albumin' is : \n",liver['albumin'].groupby(liver['is_patient']).min())


# In[112]:


# Grouping on basis of "is_patient" and using max() function for 'albumin '
print("Grouping on basis of 'is_patient' and calculating maximum of 'albumin' is : \n",liver['albumin'].groupby(liver['is_patient']).max())


# In[113]:


# Grouping on basis of "is_patient" and using mean() function for 'tot_proteins'
print("Grouping on basis of 'is_patient' and calculating mean of 'tot_proteins' is : \n",liver['tot_proteins'].groupby(liver['is_patient']).mean())


# CREATING CHARTS FOR DATAFRAME

# In[114]:


liver.hist(column='alkphos',by ='gender')


# In[115]:


liver.plot.scatter('tot_bilirubin','direct_bilirubin')


# In[116]:


print(pd.crosstab(liver.gender,liver.is_patient,margins=True).plot(kind='bar',figsize=(7,5)))


# MISSING VALUES

# In[117]:


loan=pd.read_csv("loan.csv")


# In[118]:


print("Displaying the dataset : \n",loan)


# In[119]:


# Displaying the dimension of the dataset using shape.
print("Dimension of the dataset is : \n",loan.shape)


# In[120]:


# Displaying the null values present in the dataset using isnull() and isna().
print("Number of null values present in the dataset are : \n",loan.isnull().sum())


# In[121]:


# Deleting observations containing Missing values


# In[122]:


newloan=loan.copy()


# In[123]:


# Removing the complete observations containing missing values.
newloan.dropna(inplace=True)


# In[124]:


print(newloan.isnull().sum(),'\n')
print('dimension of newloan after removing missing values is : \n',newloan.shape)


# In[ ]:





# # DATA VISUALIZATION
# 

# # 1. MATPLOTLIB

# In[1]:


import matplotlib.pyplot as plt


# In[7]:


#creating a line chart for single list.
p=plt.plot([12,23,34,54,56,78,46,78,78])

# adding label on y-axis and x-axis
plt.ylabel("Random number")
plt.xlabel("x - axis")
plt.show()


# In[8]:


# program for creating a chart considering ggplot from style
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')


# In[17]:


# creating a line chart with ggplot
plt.plot([12,45,23,67,90,89,76,66,55,100,61,11,12],color="g")
plt.ylabel('Random numbers')
plt.xlabel('x-axis')
#displaying a chart
plt.show()


# In[16]:


# program for creating line chart with grids and special effects.
plt.plot([7,5,6,2,7,11,13],[5,7,9,12,15,16,17],color='r',linewidth=1.0)

# displaying title of the chart
plt.title("chart with user defined color and width of the line")

# label on x-axis
plt.xlabel('X-AXIS')

# label on y-aixs
plt.ylabel("Y-AXIS")

# displaying a grid
plt.grid(True)

# displaying text on the plot area
plt.text(5,12,"green colored line")

# displaying the chart
plt.show()


# In[26]:


# Program to draw dot and lines on the same chart with axes limit
plt.plot([6,9,7,11,13],[8,11,13,12,15],'ro',[6,9,7,11,13],[8,11,13,12,15],'m')

# adding label to the x-axis
plt.xlabel("X-AXIS")
# adding label to the y-axis
plt.ylabel("Y-AXIS")

# Adding title to the chart
plt.title("Dot and line chart together")

# Setting the limit of both the axes
plt.axis([5,15,5,17])

# Displaying the chart
plt.show()


# In[27]:


# creating four lists
a=list(range(1,11))
b=list(range(5,55,5))
c=list(range(10,110,10))
d=list(range(20,210,20))
print("first list: ",a)
print("second list: ",b)
print("third list: ",c)
print("fourth list: ",d)


# creating a chart with different colors and effects for different data.
plt.plot(a,a,'g^',a,b,'bs', a, c, 'r--',a,d, 'mo')

# set limits of x-axis.
plt.xlim(0,11)
# set limits of y-axis
plt.ylim(0,220)

# adding labels on x-axis and y-axis
plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")

# title
plt.title("Multiple lines with different shapes and colors")

# displaying grid in chart
plt.grid(True,color='k')

# displaying the chart
plt.show()


# PIE CHART
# 
# --> Pie charts visualize absolute and relative frequencies

# In[33]:


# Program for creating Pie chart
import matplotlib.pyplot as plt
list1=[10,30,20,40,50,70,90,10]

# creating and displaying Pie chart
plt.pie(list1, labels=list1)
plt.show()


# VIOLIN PLOT
# 
# --> This plot is the combination of Box and kernel density plot. it is drawn using violinplot() from matplotlib.pyplot

# In[41]:


# Program to draw a violin plot
plt.violinplot([10,20,11,23,17,18,16,34,37,54,62,87,99,70,40,50,61,64],showmeans=True,showextrema=True)

# Displaying the chart
plt.show()


# SCATTER PLOT
# 
# --> Scatter plot is used to display the bivariate data. This scatter plot is created using scatter() function. Scatter plot show many points plotted in the cartesian plane. Each points represents the values of two variables. One variable is chosen in the horizontal axis and another in the vertical axis.

# In[47]:


# Program to draw a scatter plot
ecommerce=['Myntra','Snapdeal','Alibaba','Amazon','Flipkart']
q1=[35,45,100,70,40]
q2=[38,40,105,65,45]
q3=[30,42,120,72,50]
q4=[25,34,115,60,48]

# creating a scatter plot
plt.scatter(ecommerce,q1,color='green')
plt.scatter(ecommerce,q2,color='blue')
plt.scatter(ecommerce,q3,color='pink')
plt.scatter(ecommerce,q4,color='red')

# Adding title to the chart and labels to the axis
plt.xlabel("Organization Name")
plt.ylabel("Profit")
plt.title("Scatter Plot")

# Displaying a scatter chart
plt.show()


# HISTOGRAM
# 
# --> Histogram is based on the idea to categorize data into different groups and plot the bars for each category with height.

# In[87]:


# Program to display a histogram 
import matplotlib.pyplot as plt

#creating a histogram using hist2d() function
list1=list(range(1,100,20))
list2=[12,15,17,19,15]
plt.hist2d(list2,list1,color='g')
# displaying labels
plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")

# displaying title 
plt.title("histogram")

plt.grid(True)
# Displaying a chart
plt.show()


# BAR CHART

# In[1]:


import matplotlib.pyplot as plt


# In[13]:


# Creating a horizontal barplot.
plt.barh([2,4,5,6,7],width=[1,2,3,2,1],color='g')
# Displaying the chart
plt.show()


# In[30]:


# Program for drawing multiple bar charts on one image
ecommerce=['Myntra','Snapdeal','Alibaba','Amazon','Flipkart']
q1=[35,45,100,70,40]
q2=[38,40,105,65,45]
q3=[30,42,120,72,50]
q4=[25,34,115,60,48]

# Creating different bar charts on one image using subplot() function.
plt.figure(1,figsize=(18,13))

# Creating bar chart in first cell of figure having 2 rows, 3 columns.
plt.subplot(221)
plt.bar(ecommerce,q1,color='g')
plt.title("Quarter 1 profit")

# Creating bar chart in second cell. 
plt.subplot(222)
plt.bar(ecommerce,q2,color='r')
plt.title("Quarter 2 profit")

# Creating bar chart in third cell. 
plt.subplot(223)
plt.bar(ecommerce,q3,color='y')
plt.title("Quarter 3 profit")

# Creating bar chart in fourth cell. 
plt.subplot(224)
plt.bar(ecommerce,q4,color='b')
plt.title("Quarter 4 profit")

# Adding a Main title for the figure.
plt.suptitle("Profits on quarter basis.")


# Displaying the chart.
plt.show()


# In[33]:


# Program for drawing multiple bar charts on one image
ecommerce=['Myntra','Snapdeal','Alibaba','Amazon','Flipkart']
q1=[35,45,100,70,40]
q2=[38,40,105,65,45]
q3=[30,42,120,72,50]
q4=[25,34,115,60,48]

# Creating different bar charts on one image using subplot() function.
plt.figure(1,figsize=(18,13))

# Creating bar chart in first cell of figure having 2 rows, 3 columns.
plt.subplot(221)
plt.barh(ecommerce,q1,color='g')
plt.title("Quarter 1 profit")

# Creating bar chart in second cell. 
plt.subplot(222)
plt.barh(ecommerce,q2,color='r')
plt.title("Quarter 2 profit")

# Creating bar chart in third cell. 
plt.subplot(223)
plt.barh(ecommerce,q3,color='y')
plt.title("Quarter 3 profit")

# Creating bar chart in fourth cell. 
plt.subplot(224)
plt.barh(ecommerce,q4,color='b')
plt.title("Quarter 4 profit")

# Adding a Main title for the figure.
plt.suptitle("Profits on quarter basis.")


# Displaying the chart.
plt.show()


# STACKED BAR CHART

# In[37]:


# Program to create a stacked bar chart.
import matplotlib.pyplot as plt

plt.figure(1,figsize=(5,10))
countries=['A','B','C','D','E']
Population_1930=[10,20,24,30,20]
Population_1940=[12,27,30,42,26]
Population_1950=[15,35,34,50,33]
Population_1960=[27,43,43,57,38]
Population_1970=[35,45,46,62,40]
Population_1980=[38,49,55,65,45]
Population_1990=[42,52,60,72,50]
Population_2000=[48,54,65,75,58]
Population_2010=[53,58,70,77,63]


# Creating a stacked bar chart
plt.bar(countries,Population_2010,color='green')
plt.bar(countries,Population_2000,color='red')
plt.bar(countries,Population_1990,color='yellow')
plt.bar(countries,Population_1980,color='blue')
plt.bar(countries,Population_1970,color='brown')
plt.bar(countries,Population_1960,color='orange')
plt.bar(countries,Population_1950,color='purple')
plt.bar(countries,Population_1940,color='magenta')
plt.bar(countries,Population_1930,color='pink')

# Adding labels and title to the chart
plt.xlabel("Countries")
plt.ylabel("Population in different years")
plt.title("Stacked Bar Chart")

#Displaying a Bar Chart
plt.show()


# In[40]:


import matplotlib.pyplot as plt


# AREA PLOT / STACK PLOT

# In[42]:


# Program to create an area plot
month_sales=[1,2,3,4,5,6,7,8,9,10,11,12]
electronics=[8,11,9,8,10,9,10,11,12,9,10,11]
clothing=[8,7,8,6,8,7,5,8,5,7,8,6]
food=[5,6,2,4,6,5,2,4,2,3,4,5]
books=[3,4,5,3,3,6,6,2,6,5,3,4]

plt.plot([],[],color='pink',label=books",linwidth=5)
plt.plot([],[],color='blue',label='od',linwidth=5)
plt.plot([],[],color='red',labelclothing',linwidth=5)
plt.plot([],[],color='cyan',label'electronics',linwidth=5)

plt.stackplot(month_sales,electronics,clothing,food,books,color=['pink','blue','red','cyan'])
plt.xlabel("MONTHS")
plt.ylabel("SALES")
plt.title("Area Chart for Sales of different categories")
plt.legend()

# Displaying the chart
plt.show()


# In[48]:


# Program to create an area plot
month_sales=[1,2,3,4,5,6,7,8,9,10,11,12]
electronics=[8,11,9,8,10,9,10,11,12,9,10,11]
clothing=[8,7,8,6,8,7,5,8,5,7,8,6]
food=[5,6,2,4,6,5,2,4,2,3,4,5]
books=[3,4,5,3,3,6,6,2,6,5,3,4]

plt.plot([],[],color='pink',label='books', linewidth=5)
plt.plot([],[],color='blue',label='food', linewidth=5)
plt.plot([],[],color='red',label='clothing',linewidth=5)
plt.plot([],[],color='cyan',label='electronics',linewidth=5)

plt.stackplot(month_sales,electronics,clothing,food,books,color=['pink','blue','red','cyan'])
plt.xlabel("MONTHS")
plt.ylabel("SALES")
plt.title("Area Chart for Sales of different categories")
plt.legend()

# Displaying the chart
plt.show()


# MESH GRID 

# In[62]:


# Program for creating a meshgrid
import matplotlib.pyplot as plt
import numpy as np


# Creating large datasets
var1=np.linspace(1,100,1000)
var2=np.linspace(5,200,2000)

# Creating a Meshgrid chart
A,B=np.meshgrid(var1,var2)
plt.figure(1,figsize=(10,5))
plt.subplot(131)
plt.imshow(A+B)
plt.subplot(132)
plt.imshow(A-B)
plt.subplot(133)
plt.imshow(A*B)


# # SEABORN LIBRARY

# Relational plot
# 
# ~ to see the statistics relation between 2 or more variables.
# 
# ~ Bivariate Analysis
# 
# Plot under this Session
# 
# ~ scatter plot
# 
# ~ Lineplot

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[10]:


tips=sns.load_dataset('tips')
tips


# In[ ]:





# In[20]:


# scatter plot -->axis level function
sns.scatterplot(data=tips,x='total_bill',y='tip',hue='sex',style='time',size='size')


# In[21]:


# relplot --> figure level plot -->square shape
sns.relplot(data=tips,x='total_bill',y='tip',hue='sex',style='time',size='size')


# In[29]:


# Line plot --> axis level function
gap=px.data.gapminder()
temp_df=gap[gap['country']=='India']
temp_df


# In[36]:


sns.lineplot(data=temp_df,x='year',y='lifeExp')


# In[41]:


temp_df=gap[gap['country'].isin(['India','Brazil','Germany'])]
temp_df


# In[43]:


sns.relplot(kind='line',data=temp_df,x='year',y='lifeExp',hue='country',style='continent',size='continent')


# In[40]:


sns.lineplot(data=temp_df,x='year',y='lifeExp',hue='country')


# In[ ]:





# In[35]:


sns.relplot(data=temp_df,x='year',y='lifeExp',kind='line')


# In[51]:


# FACET PLOT
sns.relplot(data=tips,x='total_bill',y='tip',kind='scatter',col='smoker',row='sex')


# In[54]:


# FACET PLOT  ----> work with relplot not with scatterplot and line plot
sns.relplot(data=tips,x='total_bill',y='tip',kind='scatter',col='sex',row='day')


# DISTRIBUTION PLOT
# 
# ~ used for univariate analysis
# 
# ~ used to find out the distribution
# 
# ~ range of the distribution
# 
# ~ Central tendency
# 
# ~ is the data dimodal?
# 
# ~ are there outliers?
# 
# 
# PLOT UNDER DISTRIBUTION PLOT
# ~ histplot
# 
# ~ kdeplot
# 
# ~ rugplot 

# In[55]:


# figure level plot--> displot
# axes level plot--> histplot, kdeplot, rugplot


# In[59]:


# plotting univariate histogram
sns.histplot(data=tips,x='total_bill')


# In[60]:


# plotting univariate histogram
sns.displot(data=tips,x='total_bill',kind='hist',bins=20)


# In[65]:


# plotting univariate histogram
sns.displot(data=tips,x='tip',kind='hist',hue='sex')


# In[67]:


titanic=sns.load_dataset("titanic")
titanic


# In[69]:


sns.displot(data=titanic,x='age',kind='hist',hue='sex',element='step')


# In[70]:


sns.displot(data=titanic,x='age',kind='hist',hue='sex',element='step',col='sex')


# In[72]:


# KDE plot
sns.kdeplot(data=tips,x='total_bill')


# In[74]:


# KDE plot
sns.kdeplot(data=tips,x='total_bill',hue='sex',fill=True)


# In[75]:


# RUGPLOT -->

sns.kdeplot(data=tips,x='total_bill')
sns.rugplot(data=tips,x='total_bill')


# In[80]:


# bivariate histplot

sns.histplot(data=tips,x='total_bill',y='tip')


# In[82]:


# bivariate kdeplot
sns.kdeplot(data=tips,x='total_bill',y='tip')


# MATRIX PLOT
# 
# ~ Heatmap
# 
# ~ Clustermap

# In[89]:


# heatmap---> plot rectangular data as a color-encoded matrix---> it is axes level function
gap


# In[88]:


temp_df=gap.pivot(index='country',columns='year',values='lifeExp')


# In[103]:


plt.figure(figsize=(15,15))
sns.heatmap(temp_df)


# In[105]:


eu=gap[gap['continent']=='Europe'].pivot(index='country',columns='year',values='lifeExp')


# In[111]:


plt.figure(figsize=(15,15))
sns.heatmap(eu,annot=True,linewidth=0.5,cmap='autumn')


# In[114]:


# CLUSTERMAP----> plot a matrix dataset as a hierarchically-clustered heatmap
# this function requires scipy to be available.

iris=px.data.iris()
iris


# In[116]:


sns.clustermap(iris.iloc[:,[0,1,2,3]])


# In[118]:


# import datasets
tips=sns.load_dataset('tips')
iris=sns.load_dataset("iris")


# In[119]:


tips


# In[120]:


iris


# CATEGORICAL PLOTS
# 
# Categorical Scatter Plot
# 
# ~ Stripplot
# 
# ~ Swarmplot
# 
# Categorical Distribution Plots
# 
# ~ Boxplot
# 
# ~ Violinplot
# 
# Categorical Estimate Plot--> for central tendency
# 
# ~ Barplot
# 
# ~ Pointplot
# 
# ~ countplot

# In[122]:


# strip plot
sns.stripplot(data=tips,x='day',y='total_bill',jitter=False)  # axes level 


# In[126]:


# using catplot ---> figure level

sns.catplot(data=tips,x='day',y='total_bill',kind='strip')


# In[131]:


# jitter --can customized by using values
sns.catplot(data=tips,x='day',y='total_bill',kind='strip',jitter=.01,hue='sex')


# In[135]:


# swarm plot
sns.catplot(data=tips,x='day',y='total_bill',kind='swarm') # figure level


# In[137]:


sns.swarmplot(data=tips,x='day',y='total_bill',hue='sex') # axes level


# BOX PLOT--

# In[143]:


# Box plot
sns.boxplot(data=tips,x='day',y='total_bill',hue='sex')  # axes level


# In[142]:


# using catplot

sns.catplot(data=tips,x='day',y='total_bill',kind='box')


# In[144]:


# single boxplot --> numerical col
sns.boxplot(data=tips,y='total_bill')


# VIOLIN PLOT --> (Boxplot + KDEplot)

# In[148]:


# violin plot
sns.violinplot(data=tips,x='day',y='total_bill')


# In[149]:


# using catplot
sns.catplot(data=tips,x='day',y='total_bill',kind='violin',hue='sex')


# In[155]:


# Bar plot
import numpy as np
sns.barplot(data=tips,x='sex',y='total_bill',hue='smoker',estimator=np.min)


# In[159]:


# point plot
sns.pointplot(data=tips,x='sex',y='total_bill',hue='smoker')


# In[158]:


sns.pointplot(data=tips,x='sex',y='total_bill',hue='smoker',estimator=np.min)


# In[160]:


sns.pointplot(data=tips,x='sex',y='total_bill',hue='smoker',estimator=np.max)


# In[162]:


# count plot
sns.countplot(data=tips,x='sex',hue='day')


# Regression plots
# 
# ~ regplot
# 
# ~ lmplot
# 
# In the simplest invocation, both functions draw a scatteredplot of two variables. x and y, and then fit the regression model y~x and plot the resulting regression line and a 95% confidence interval for that regression.

# In[169]:


# using regplot---hue parameter is not available---> axes level
sns.regplot(data=tips,x='total_bill',y='tip')


# In[172]:


# using lmplot---hue parameter available 
sns.lmplot(data=tips,x='total_bill',y='tip',hue='sex')


# In[174]:


# residplot
sns.residplot(data=tips,x='total_bill',y='tip')


# Multi grid plot

# In[175]:


# facet grid 
sns.catplot(data=tips,x='sex',y='total_bill',kind='violin',col='day',row='time')


# In[180]:


g=sns.FacetGrid(data=tips,col='day',row='time',hue='smoker')
g.map(sns.violinplot,'sex','total_bill')
g.add_legend()


# Plotting Pairwise Relationship (PairGrid Vs Pairplot)

# In[181]:


iris


# In[185]:


sns.pairplot(iris,hue='species')


# In[184]:


sns.pairplot(tips,hue='sex')


# In[189]:


# Pair Grid 
g=sns.PairGrid(data=iris,hue='species')

#g map
g.map(sns.scatterplot)


# In[199]:


# map diag ---->map offdiag

g=sns.PairGrid(data=iris,hue='species')
g.map_diag(sns.violinplot)
g.map_offdiag(sns.scatterplot)


# JointGrid Vs Jointplot

# In[200]:


sns.jointplot(data=tips,x='total_bill',y='tip')


# In[204]:


sns.jointplot(data=tips,x='total_bill',y='tip',kind='kde',hue='sex')


# In[202]:


sns.jointplot(data=tips,x='total_bill',y='tip',kind='reg')


# In[203]:


sns.jointplot(data=tips,x='total_bill',y='tip',kind='hex')


# In[208]:


# joint grid
g=sns.JointGrid(data=tips,x='total_bill',y='tip')
g.plot(sns.scatterplot,sns.histplot)


# # PLOTLY

# In[15]:


import plotly.graph_objects as go
import seaborn as sns


# In[8]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3,4,5],y=[3,4,5,6,7], mode='markers'))
fig.show()


# In[9]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3,4,5],y=[3,4,5,6,7], mode='lines'))
fig.show()


# In[14]:


fig = go.Figure()
fig.add_trace(go.Bar(x=[1,2,3,4,5],y=[3,4,5,6,7]))
fig.show()


# In[18]:


tips=sns.load_dataset("tips")
tips


# In[27]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=['tip'],y=tips['total_bill']))
fig.show()


# In[29]:


fig = go.Figure(data=[go.Histogram(x=tips.total_bill)])
fig.show()


# In[33]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=tips.total_bill,y=tips.tip, mode='markers'))
fig.show()


# In[38]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=tips.total_bill,y=tips.tip, mode='markers',marker_size=10*tips['size']))
fig.show()


# In[39]:


fig = go.Figure()
fig.add_trace(go.Scatter3d(x=tips.total_bill,y=tips.tip, mode='markers',z=tips['size']))
fig.show()


# In[42]:


fig = go.Figure()
fig.add_trace(go.Scatter3d(x=tips.total_bill,mode='lines',y=tips.tip,z=tips['size']))
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




