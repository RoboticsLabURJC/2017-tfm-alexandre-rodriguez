import os
import glob

path = "dets/*.*"
last_created_file = (max(glob.glob(path), key=os.path.getmtime)).split('/')[1][:-4]
for i in range(int(last_created_file)):
    if not os.path.isfile('dets/'+str(i)+'.txt'):
        with open('dets/'+str(i) + '.txt', 'a') as logfile:
            continue  # create empty file if not found