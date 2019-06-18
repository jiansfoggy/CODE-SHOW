# insert doctors
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,70,72,
         78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]
for i in range(100):
  if i+1 in Ill_Doc: 
    print("CREATE (D:Doctors{Name:'D%d',If_Patient:'Yes'});" %(i+1))
  else:
    print("CREATE (D:Doctors{Name:'D%d',If_Patient:'No'});" %(i+1))

# insert patient
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,70,72,
         78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]
for i in range(10041):
  if i < 10000:
    ill_num = random.randint(0,3)
    doc_num = random.randint(1,5)
    if ill_num == 0:
      print("CREATE (P:Patients{Name:'P%d',Ill:'No Illness',Doc:'Do not need'});" %(i+1))
    else:
      ill_ls = []
      doc_ls = []
      for j in range(ill_num):
        ill_ls.append('I%d' %(random.randint(1,1000)))
      for j in range(doc_num):
        doc_ls.append('D%d' %(random.randint(1,100)))
      print("CREATE (P:Patients{Name:'P%(pid)d',Ill:%(ill)s,Doc:%(doc)s});" %\
            {"pid":i+1,"ill":ill_ls,"doc":doc_ls})

  else:
    ill_num = random.randint(1,3)
    doc_num = random.randint(1,5)
    ill_ls = []
    doc_ls = []
    for j in range(ill_num):
      ill_ls.append('I%d' %(random.randint(1,1000)))
    for j in range(doc_num):
      doc_ls.append('D%d' %(random.randint(1,100)))
    print("CREATE (P:Patients{Name:'D%(pid)d',Ill:%(ill)s,Doc:%(doc)s});" %\
            {"pid":Ill_Doc[i-10000],"ill":ill_ls,"doc":doc_ls})

# insert illness
import random
import numpy as np
for i in range(1000):
  print("CREATE (I:Illness{Name:'I%d'});" %(i+1))

# insert treatment
import random
import numpy as np
Ill_Doc=[8,14,15,16,19,21,23,24,30,38,43,50,51,54,55,56,58,60,62,64,65,
         70,72,78,80,81,82,83,85,86,87,88,89,90,92,93,94,95,96,97,100]
for i in range(750):
  pat_num = random.randint(1,500)
  pat_ls = []
  for j in range(pat_num):
    pat_ind = random.randint(1,10041)
    if pat_ind<10000:
      pat_ls.append('P%d' %(pat_ind))
    else:
      pat_ls.append('D%d' %(Ill_Doc[pat_ind-10001]))
  print("CREATE (T:Treatments{Name:'T%(trt)d',Pat:%(pat)s});" %\
          {"trt":i+1,"pat":pat_ls})