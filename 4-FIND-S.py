# Algorithm
# Find-S Algorithm
# Step 1: Start
# Step 2: Import the required modules such as pandas.
# Step 3: Create data set and save it as pandas DataFrame.
# Step 4: Initialize h with most specific hypothesis.
# Step 5: Implement Find-S algorithm to find the most general hypothesis 
# Step 6: Display the maximally hypothesis
# Step 7: Stop

import pandas as pd

f = pd.read_csv('data.csv')
le = len(f.columns)
hypothesis = [0]*(le-1)
print("Hypothesis at beginning: ", hypothesis)

arr = []
f = open('data.csv', 'r')
for i in f.readlines():
	arr.append(i.split(',')

for i in range(len(arr[0])-1):
  hypothesis[i] = arr[0][i]

 for i in range(1, len(arr)):
  if(arr[i][-1][:-1] == "No"):
    pass
  else:
    for j in range(0, len(arr[i])-1):
      if hypothesis[j] != arr[i][j]:
        hypothesis[j] = "?"
  print("Hypothesis:", hypothesis)

  print("The maximally specific hypothesis is:", hypothesis)