fname1 = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/physionet.org/files/epilepsy-data/chbmit/epilepsy-data/1.0.0/RECORDS-WITH-SEIZURES"
fname2 = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/physionet.org/files/epilepsy-data/chbmit/epilepsy-data/1.0.0/RECORDS"
count1 = 0
count2 = 0
with open(fname1, 'r') as f:
    for line in f:
        count1 += 1

with open(fname2, 'r') as f:
    for line in f:
        count2 += 1
print("Total number of lines is:", count1, count2)