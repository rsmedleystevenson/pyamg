
import pdb
import numpy as np


Cpts = [1,4,5,7,9,10,12,13]
Cpts_size = len(Cpts)
n = 15

# Form set of F-points, given set of C-points. 
Fpts = np.zeros((n-Cpts_size,));
temp_Cind = 0;
temp_Find = 0;

C0 = 0;
C1 = Cpts[0];
for i in range(C0, C1):
	Fpts[temp_Find] = i;
	temp_Find += 1;

while (temp_Cind < (Cpts_size-1)):
	C0 = Cpts[temp_Cind];
	C1 = Cpts[temp_Cind+1];
	for i in range((C0+1), C1):
		Fpts[temp_Find] = i;
		temp_Find += 1;

	temp_Cind += 1;

C0 = Cpts[temp_Cind];
for i in range((C0+1), n):
	Fpts[temp_Find] = i;
	temp_Find += 1;



pdb.set_trace()
