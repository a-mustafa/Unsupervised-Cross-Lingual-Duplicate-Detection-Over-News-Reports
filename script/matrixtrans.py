import numpy as np
from numpy import linalg
import cPickle
import os
import sys
import time
import scipy.io









if __name__ == '__main__':
    if len(sys.argv) < 2:
    	print 'Usage: python pretrain_da.py datasetl datasetr'
    	print 'Example: python pretrain_da.py gauss basic'
    	sys.exit()
    
    #python matrixtrans.py /home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsleftAttro2enes.txt /home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsrightAttro2enes.txt
    filepathl= str(sys.argv[1]) 
    filepathr = str(sys.argv[2])
    filepathl='/home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsleftAttro2enes.txt'
    filepathr ='/home/ahmad/duplicate-detection/eventregistrydata/pairs/matchingwordsrightAttro2enes.txt'
    
    with open(filepathl, 'r') as myfile:
        content=myfile.readlines()#.replace('\n', '')
    
    def str2float(strng):
      if strng.strip()=="":
        return strng.strip()
      return float(strng.strip())
    
    data_l=[map(str2float,con.strip()[1:-1].split(",")) for con in content if "," in con]
    data_l=np.asarray(data_l)
    #print data_l
    with open(filepathr, 'r') as myfile:
        content=myfile.readlines()#.replace('\n', '')
    
    data_r=[map(str2float,con.strip()[1:-1].split(",")) for con in content if "," in con]
    data_r=np.asarray(data_r)
    #print data_r
    #data_l*A=data_r
    A=linalg.lstsq(data_l, data_r)
    dot(data_l[0],A[0])
    X=np.dot(data_l.T,data_l)
    Xprime=np.dot(data_r.T,data_r)
    A=linalg.solve(X, Xprime)
    cPickle.dump(A[0], open('/home/ahmad/duplicate-detection/eventregistrydata/pairs/AlstsqAttro2enes.p', 'wb'))
    
    
'''
from numpy import matrix
A = matrix( [[1,2,3],[11,12,13],[21,22,23]]) # Creates a matrix.
x = matrix( [[1],[2],[3]] )                  # Creates a matrix (like a column vector).
y = matrix( [[1,2,3]] )                      # Creates a matrix (like a row vector).
print A.T                                    # Transpose of A.
print A*x                                    # Matrix multiplication of A and x.
print A.I                                    # Inverse of A.
print A*linalg.solve(A, x)     # Solve the linear equation system.
'''