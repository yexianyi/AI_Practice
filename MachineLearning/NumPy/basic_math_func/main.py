import numpy as np

iris_data = np.loadtxt('iris_data.csv', delimiter=',')
print("file content:" + str(iris_data))

# Sort
iris_data.sort()
print("Sorting Result: \n", iris_data)

# len
print("Length of data: ", len(iris_data))

# remove duplicate
after_rm_dup = np.unique(iris_data)
print("After remove duplicate: \n", after_rm_dup)

# sum array
print("Sum: ", np.sum(iris_data))

# cumsum
print("accumulative sum: \n", np.cumsum(iris_data))

# avg
print("average value: ", np.mean(iris_data))

# var
print("variance value: ", np.var(iris_data))

# min
print("min value: ", np.min(iris_data))

# max
print("max value: ", np.max(iris_data))