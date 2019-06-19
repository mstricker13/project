import os

name = 'M3C_other.csv'
name_save = 'M3C_other_no_meta.csv'
no_meta_index = 6
with open(name, 'r') as file:
    text = file.read()
with open(name_save, 'w+') as file:
    for line in text.split('\n'):
        values = line.split(',')
        new_line = ','.join(values[no_meta_index:]) + '\n'
        file.write(new_line)

#TODO remove last empty line manually