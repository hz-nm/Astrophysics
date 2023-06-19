import os

with open('test.txt', 'r') as f:
    lines = f.readlines()
    


columns = lines[0].split(' ')
vals = lines[1].split(' ')

print(columns)
print(vals)
