# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:39:14 2019

@author: Diip
"""

def raise_to_power(x,y):
    return x**y

raise_to_power(2,3)

raise_to_power = lambda x,y : x**y

raise_to_power(2,3)

def power(a_list):
    return a_list**2

a_list = [10,20,30]

#power(a_list)

rs = map(lambda a:a**2,a_list)
rs

list(rs)


nums = [5,6,10,1,2]
def more_than_5(x):
    y=[]
    for i in x:
        if i>5:
            y.append(i)
    return y

more_than_5(nums)

morethan_5 = filter(lambda num: num>5,nums)
morethan_5
list(morethan_5)
type(morethan_5)


#reduce()
a_list_prime_num= [2,3,5,7,11,13,17,19,23,29]

product = 1
for i in a_list_prime_num:
    product = product*i
product


from functools import reduce
re = reduce(lambda x,y:x*y,a_list_prime_num)
re


# exception and errors

def sqrt(x):
    try:
        print(x**0.5)
    except:
        print('x must be an integer or float')
        
sqrt(9)
sqrt('ABCD')


def sqrt(x):
    if x<0:
        raise ValueError('x must be non-negative')
    else:
        print(x**0.5)

sqrt(9)
sqrt(-9)

#iterables and iteratots

names = ['Ram','Rabson','Robert','Dilip','Nancy']

for i in names:
    print(i)

avoid_for = iter(names)
#list(avoid_for)
#type(avoid_for)
next(avoid_for)
next(avoid_for)
next(avoid_for)
next(avoid_for)
next(avoid_for)


a_str = 'Python'

a_str_itr = iter(a_str)
next(a_str_itr)
next(a_str_itr)
next(a_str_itr)
next(a_str_itr)
next(a_str_itr)
next(a_str_itr)

# Enumerate function

DataScience = ['PYTHON','R','MATHS','STATS','ML','DL','NLU','NLP','AI']

print(enumerate(DataScience))

DataScience_list = list(DataScience)
type(DataScience_list)

for index,value in enumerate(DataScience):
    print(index,value)

std_no = (10,20,30)
std_name = ('Ram','Nancy','Rabson')
std_marks = (100,200,300)

std_data = zip(std_no,std_name,std_marks)
list(std_data)
type(std_data)
std_data1 = zip(std_no,std_name,std_marks)
for value1,value2,value3 in std_data1:
    print(value1,value2,value3)

type(value1)
std_data1 = zip(std_no,std_name,std_marks)
value1,value2,value3 = zip(*std_data1)
value1
value2
type(value3)

print(std_no == value1)


# List Comprehensions

nums = [10,20,30,40,50]

new_nums = []

for num in nums:
    new_nums.append(num**2)
new_nums

rs_map = map(lambda num:num**2,nums)
list(rs_map)

new_nums1 = [num**2 for num in nums]
new_nums1

#list comprehensions to avoid nested loops

pairs_1 = []

for num1 in range(0,2):
  for num2 in range(6,8):
      pairs_1.append((num1,num2))
pairs_1


pairs_2 = [(num1,num2) for num1 in range(0,2) for num2 in range(6,8)]
pairs_2

# list comprehensions using conditions

print([num1 for num1 in range(10) if num1%2==0])

print({num:-num for num in range(10)})

num_list = ((num,num**2) for num in range(10))
num_list
type(num_list)
list(num_list)

# generators
std = ['Ram','Nancy','Rabson','Dilip','Omkar','Ajjagutta']

l = [num for num in std if len(num)>5]
l
g = (num for num in std if len(num)>5)
list(g)
next(g)
next(g)


# Numpy - Numerical Python

height = [1.6,1.7,1.8,1.66,1.8]
weight = [55,66,74,85,63]

BMI = weight/height **2

import numpy as np
np_height = np.array(height)
np_weight = np.array(weight)
BMI = np_weight/np_height**2
BMI
print(BMI[1])
print(BMI[1:3])
print(BMI[BMI>22])


l = [1,2,3.9]
n = np.array([1,2,3.9])
print(l)
print(n)

a = np.arange(6)
b = np.arange(12).reshape(4,3)
c = np.arange(24).reshape(2,3,4)
d = np.arange(24).reshape(2,3,2,2)

print(a)
print(b)
print(c)
print(d)

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

# working with 2D NumPy array

np_2d_array = np.array([[1.71,1.58,1.51,1.73,1.92],[65.9,58.8,66.8,83.4,68.7]])
print(type(np_2d_array))
print(np_2d_array.shape)
print(np_2d_array[0])
print(np_2d_array[0][2])
print(np_2d_array[:,0:2])
print(np_2d_array[:,3])
print(np_2d_array[1,:])


# working with basic statistics

x = [1,4,8,10,12]
print(np.mean(x))
print(np.median(x))
print(np.std(x))

# working with random numbers with out SEED

print(np.random.randint(10,50,5))

print(np.random.randint(10,50,5))

# working with random numbers with SEED

np.random.seed(1)
print(np.random.randint(10,50,5))
np.random.seed(1)
print(np.random.randint(10,50,5))



import pandas as pd

df = pd.read_csv("pandas_sales.csv",index_col='month')
df

df['eggs']
df['spam']
df['eggs']['May']

df

df.salt[0]
df.salt[0:3]
df.salt[[0,3,5]]

df.loc['Jan','eggs']
df.loc['Jan':'Jun','eggs':'spam']
df.loc[['Jan','May'],['eggs','spam']]
df
df.iloc[1,1]
df.iloc[1:3,1:3]
df.iloc[[1,3],[0,2]]
df_new = df[['eggs','spam']]
df_new
series = df['eggs']
series

print(df.salt>30)
print(df[df.salt > 30])

print(df[(df.salt >30) & (df.spam >30)])
print(df[(df.salt >30) | (df.spam >30)])

df2 =df.copy()
print(df2==df)

df2['cake'] = [0,0,50,60,70,80,90,100,110,120,130,140]
print(df2)
print(df2.loc[:,df2.all()])
df3 = df2.copy()
df3['milk'] = [0,0,0,0,0,0,0,0,0,0,0,0]
print(df3)

print(df3.loc[:,df3.any()])

print(df.loc[:,df.isnull().any()])
print(df.loc[:,df.notnull().all()])
print(df3.dropna(how='any'))

print(df.eggs[df.salt>30])
df.eggs[df.salt>30] += 5
print(df)



import pandas as pd  
df = pd.DataFrame({'A':{0:'a',1:'b',2:'c'}, 
                   'B':{0:1,1:3,2:5}, 
                   'C':{0:2,1:4,2:6}, 
                   'D':{0:7,1:9,2:11}, 
                   'E':{0:8,1:10,2:12}}) 
df
# convert dataframe using melt 
df_melt = df.melt(id_vars = ["A"], 
        value_vars=["B","C","D","E"], 
        var_name = "my_var", 
        value_name ="my_val") 
df 
df_melt











































































































































































































































































