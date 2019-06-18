# generate patients
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,70,72,
         78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]
count=0
for i in range(10041):
  doc_num=random.randint(1,5)
  for j in range(doc_num):
    count+=1
    doc_ind = random.randint(1,100)
    if i == 0 and j == 0:
      print("INSERT INTO Patients (PatID,PName,DName) VALUES (%d,'P%d','D%d')," %(count,i+1,doc_ind))
    elif i < 10000:
      print("(%d,'P%d','D%d')," %(count,i+1,doc_ind))
    elif i ==10040 and j != doc_num-1:
      print("(%d,'D%d','D%d')," %(count,Ill_Doc[i-10000],doc_ind))
    elif i == 10040 and j == doc_num-1:
      print("(%d,'D%d','D%d');" %(count,Ill_Doc[i-10000],doc_ind))
    else:
      print("(%d,'D%d','D%d')," %(count,Ill_Doc[i-10000],doc_ind))
      


# generate illness
import random
import numpy as np
for i in range(1000):
  if i == 0:
    print("INSERT INTO Illness (IName) VALUES ('I%d')," %(i+1))
  elif i > 0 and i < 999:
    print("('I%d')," %(i+1))
  else:
    print("('I%d');" %(i+1))


# genrate P_T_I
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,70,72,
         78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]

for i in range(10041):
  ill_num=random.randint(0,3)
  if ill_num == 0:
    if i == 0:
      print("INSERT INTO P_T_I (PName,IName) VALUES ('P%d','Healthy')," %(i+1))
    elif i > 0 and i < 10000:
      print("('P%d','Healthy')," %(i+1))
    elif i > 9999 and i < 10040:
      print("('D%d','Healthy')," %(Ill_Doc[i-10000]))
    else:
      print("('D%d','Healthy');" %(Ill_Doc[i-10000]))
  else:
    for j in range(ill_num):
      ill_ind = random.randint(1,1000)
      if i == 0 and j == 0:
        print("INSERT INTO P_T_I (PName,IName) VALUES ('P%d','I%d')," %(i+1,ill_ind))
      elif i == 0 and j > 0:
        print("('P%d','I%d')," %(i+1,ill_ind))
      elif i > 0 and i < 10000: 
        print("('P%d','I%d')," %(i+1,ill_ind))
      elif i > 9999 and i < 10040:
        print("('D%d','I%d')," %(Ill_Doc[i-10000],ill_ind))
      elif i ==10040 and j != ill_num-1:
        print("('D%d','I%d');" %(Ill_Doc[i-10000],ill_ind))
      else:
        print("('D%d','I%d');" %(Ill_Doc[i-10000],ill_ind))

# genrate Treatment
import random
import numpy as np
for i in range(750):
  if i == 0:
    print("INSERT INTO Treatment (TName) VALUES ('T%d')," %(i+1))
  elif i > 0 and i < 749:
    print("('T%d')," %(i+1))
  else:
    print("('T%d');" %(i+1))

# genrate P_T_T
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,70,72,
         78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]

for i in range(10041):
  trt_num=random.randint(1,500)
  for j in range(trt_num):
    trt_ind = random.randint(1,750)
    if i == 0 and j == 0:
      print("INSERT INTO P_T_T (PName,TName) VALUES ('P%d','T%d')," %(i+1,trt_ind))
    elif i == 0 and j > 0: 
      print("('P%d','T%d')," %(i+1,trt_ind))
    elif i > 0 and i < 10000:
      print("('P%d','T%d')," %(i+1,trt_ind))
    elif i > 9999 and i < 10040:
      print("('D%d','T%d')," %(Ill_Doc[i-10000],trt_ind))
    elif i ==10040 and j != trt_num-1:
      print("('D%d','T%d')," %(Ill_Doc[i-10000],trt_ind))
    else:
      print("('D%d','T%d');" %(Ill_Doc[i-10000],trt_ind))