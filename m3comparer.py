import os
import sys

#load files
path_m3 = os.path.join('hiwidata', 'M3C_monthly.csv')
path_theta = os.path.join('hiwidata', 'theta_25_h3_m3m4.csv')

with open(path_m3) as file:
    m3 = file.read()

with open(path_theta) as file:
    theta = file.read()

#remove last empty line
m3_lines = m3.split('\n')
if m3_lines[-1] == '':
    m3_lines = m3_lines[:-1]

#remove last empty line
theta = theta.split('\n')
if theta[-1] == '':
    theta = theta[:-1]
#remove the useless information fields without calculated values
theta = theta[2:]
theta = [line.split(',')[1:] for line in theta]
#remove the empty values
theta_final = list()
for line in theta:
    values = [float(val) for val in line if val != 'NA']
    theta_final.append(values)

#remove the useless information fields without calculated values
m3_lines = [','.join(line.split(',')[6:]) for line in m3_lines]
new_m3 = list()
for line in m3_lines:
    values = line.split(',')
    #remove the empty values
    values = [float(val) for val in values if val != '']
    #skip the first 25 percent
    skip = int(len(values)*0.25)
    values = values[skip:]
    new_m3.append(values)

#for line in new_m3:
#    print(line)

#for line in theta_final:
#    print(line)

print(len(new_m3), len(theta_final))
i = 1
for line_t, line_m in zip(theta_final, new_m3):
    if(len(line_t) != len(line_m)):
        print('Error ' + str(i))
        i += 1

#TODO maybe be Theta code does a wrong 0.25% split because it also takes the ,'', into account?
#-> yes it does, therefore also my "expert" values are wrong?
#one line to few in output of theta file?
#how is split into train/test/val handled? Are samples overlapping with test set removed?
#I also have sometimes small validation set