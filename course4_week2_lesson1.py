import tensorflow as tf

def print_dataset(dataset, end='\n'):
    for val in dataset:
        print(val.numpy(), end=end)


dataset = tf.data.Dataset.range(10)
print_dataset(dataset, end=" ")  # 0 1 2 3 4 5 6 7 8 9
print()

tmp = dataset.window(5, shift=1)
for window in tmp:
    print_dataset(window, end=" ")
    print()
"""
0 1 2 3 4 
1 2 3 4 5 
2 3 4 5 6 
3 4 5 6 7 
4 5 6 7 8 
5 6 7 8 9 
6 7 8 9 
7 8 9 
8 9 
9 
"""

dataset = dataset.window(5, shift=1, drop_remainder=True)
for window in dataset:
    print(type(window))
    print_dataset(window, end=" ")
    print()
"""
0 1 2 3 4 
1 2 3 4 5 
2 3 4 5 6 
3 4 5 6 7 
4 5 6 7 8 
5 6 7 8 9 
"""

dataset = dataset.flat_map(lambda window: window.batch(5))
print_dataset(dataset)
"""
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
"""

dataset = dataset.map(lambda window: (window[:-1], window[-1]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
"""
[0 1 2 3] 4
[1 2 3 4] 5
[2 3 4 5] 6
[3 4 5 6] 7
[4 5 6 7] 8
[5 6 7 8] 9
"""

dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
    print(x.numpy(), y.numpy())
"""
[4 5 6 7] 8
[1 2 3 4] 5
[2 3 4 5] 6
[3 4 5 6] 7
[0 1 2 3] 4
[5 6 7 8] 9
"""

dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x=", x.numpy(), "\ny=", y.numpy())
    print()
"""
x= [[5 6 7 8]
 [3 4 5 6]] 
y= [9 7]

x= [[4 5 6 7]
 [0 1 2 3]] 
y= [8 4]

x= [[2 3 4 5]
 [1 2 3 4]] 
y= [6 5]
"""
