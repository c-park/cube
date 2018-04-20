import numpy as np

list = [(32,21),(10,10),(12,30),(8,20),(17,23),(22,32),(31,11),(20,11),(28,33)]

list_sorted = sorted(list,key=lambda k: k[0])
print list_sorted
list1 = list_sorted[0:3]
list2 = list_sorted[3:6]
list3 = list_sorted[6:9]
print list1
print list2
print list3
list1_sorted = sorted(list1,key=lambda k: k[1])
list2_sorted = sorted(list2,key=lambda k: k[1])
list3_sorted = sorted(list3,key=lambda k: k[1])
list_final = list1_sorted + list2_sorted + list3_sorted
print list_final