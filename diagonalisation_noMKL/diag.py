
import numpy as np
import time
from pprint import pprint

def QR_algo(A, n):
	Q, R = np.linalg.qr(A)
	vQ = Q
	
	for i in range(n-1):
	# ~ for i in range(1):
		A = np.matmul(R,Q)
		Q, R = np.linalg.qr(A)
		# ~ print(Q)
		vQ = np.matmul(vQ,Q)
	
	QR_eigvals = np.diag(A)
	s = np.argsort(QR_eigvals)
	
	return (QR_eigvals[s]), vQ[:,s]

# Sets of rotations for parallel Jacobi algorithm
def rotational_sets(top, bot, n):
	m = n/2
	
	top_new = np.zeros(m, dtype=int)
	bot_new = np.zeros(m, dtype=int)
	
	for k in range(0, m):
		if k==0:
			top_new[0] = 0
		elif k==1:
			top_new[k] = bot[0]
		else:
			top_new[k] = top[k-1]
		
		if k==m-1:
			bot_new[k] = top[k]
		else:
			bot_new[k] = bot[k+1]
	
	return top_new, bot_new

# Symmetric 2-by-2 Schur Decomposition
def sym_Schur2(A, p, q):
	if A[p,q] != 0:
		tau = (A[q,q] - A[p,p]) / (2.*A[p,q])
		
		if tau>=0:
			t =  1. / (tau + np.sqrt(1+tau**2))
		else:
			t = -1. / (-tau + np.sqrt(1+tau**2))
		
		c = 1 / np.sqrt(1+t**2)
		s = t*c
	else:
		c = 1
		s = 0
	return c, s

def offDiag(A, n):
	offDiag2Sum = 0
	for i in xrange(n):
		for j in range(i+1,n):
			offDiag2Sum += 2*A[i,j]**2
	
	return np.sqrt(offDiag2Sum)

# Parallel Jacobi algorithm
def Jacobi_algo(A, n):
	V = np.identity(n)
	eps = 1e-10
	maxIter = n
	it = 0
	
	while offDiag(A,n)>eps and it<=maxIter:
		it += 1
		
		Apq_max = 0
		for i in range(0,n):
			for j in range(i+1,n):
				if abs(A[i,j]) > Apq_max:
					Apq_max = abs(A[i,j])
					p = i
					q = j
		
		c, s = sym_Schur2(A, p, q)
		
		J = np.identity(n)
		J[p,p] = c
		J[q,q] = c
		J[p,q] = s
		J[q,p] = -s
		
		# A = J^T x A x J
		A = np.matmul(np.transpose(J), np.matmul(A,J))
		V = np.matmul(V,J)
	
	eigvals = np.diag(A)
	sort = np.argsort(eigvals)
	
	return (eigvals[sort]), V[:,sort]
		
	
# Parallel Jacobi algorithm
def Jacobi_parallel_algo(A, n):
	V = np.identity(n)
	eps = 1e-10
	maxIter = n
	it = 0
	
	top = np.arange(0,n,2)
	bot = np.arange(1,n,2)
	
	while offDiag(A,n)>eps and it<=maxIter:
		it += 1
		for sets in xrange(n-1):	# Independent rotational set (determines top and bot sets)
			# ~ start = time.time()
			for k in xrange(n/2):	# Loop through sets, this can be parallellised
				p = min(top[k], bot[k])
				q = max(top[k], bot[k])
				
				c, s = sym_Schur2(A, p, q)
				
				# Make Jacobi rotation matrix
				J = np.identity(n)
				J[p,p] = c
				J[q,q] = c
				J[p,q] = s
				J[q,p] = -s
				
				# Update A (3 loops to avoid if-tests)
				for i in range(0,p):
					Api = A[p,i]
					Aqi = A[q,i]
					
					A[p,i] = c*Api - s*Aqi
					A[i,p] = A[p,i]
					
					A[q,i] = c*Aqi + s*Api
					A[i,q] = A[q,i]
				for i in range(p+1,q):
					Api = A[p,i]
					Aqi = A[q,i]
					
					A[p,i] = c*Api - s*Aqi
					A[i,p] = A[p,i]
					
					A[q,i] = c*Aqi + s*Api
					A[i,q] = A[q,i]
				for i in range(q+1,n):
					Api = A[p,i]
					Aqi = A[q,i]
					
					A[p,i] = c*Api - s*Aqi
					A[i,p] = A[p,i]
					
					A[q,i] = c*Aqi + s*Api
					A[i,q] = A[q,i]
				
				
				App = A[p,p]
				Apq = A[p,q]
				Aqq = A[q,q]
				A[p,p] = c**2*App - 2*c*s*Apq + s**2*Aqq
				A[q,q] = s**2*App + 2*c*s*Apq + c**2*Aqq
				A[p,q] = 0
				A[q,p] = 0
				
				# Update V
				for i in range(0,n):
					Vip = V[i,p]
					Viq = V[i,q]
					V[i,p] = c*Vip - s*Viq
					V[i,q] = s*Vip + c*Viq
			
			top, bot = rotational_sets(top, bot, n)
	
	eigvals = np.diag(A)
	sort = np.argsort(eigvals)
	
	return (eigvals[sort]), V[:,sort]
	# ~ return np.diag(A), V

def make_mat(n):
	A = np.zeros(shape=(n,n))
	
	for i in xrange(n):
		for j in xrange(n):
			A[i,j] = (i + j) * np.sqrt(i + j) / 1e3
	return A

def make_mat_example():
	return np.matrix([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]]) 

N = 46
mat = make_mat(N)
mat_NP = make_mat(N)

# ~ mat = make_mat_example()
# ~ mat_NP = make_mat_example()

start = time.time()
# ~ eigVals, eigVecs = QR_algo(mat, N)
eigVals, eigVecs = Jacobi_parallel_algo(mat, N)
# ~ eigVals, eigVecs = Jacobi_algo(mat, N)
end = time.time()
time_custom = end - start

start = time.time()
eigVals_NP, eigVecs_NP = np.linalg.eigh(mat_NP)
end = time.time()
time_NP = end - start

max_val = 0

print "\n"
for i in xrange(N):
	
	max_val_i = abs(eigVals_NP[i] - eigVals[i])
	if max_val_i > max_val:
		max_val = max_val_i
	
	# ~ print eigVals[i]
	# ~ print eigVals_NP[i], "\n"
	
	# ~ print "\n",eigVecs[i]
	# ~ print eigVecs_NP[i]
print max_val

print "\nTimes:" 
print "My routine:", time_custom
print "np routine:", time_NP
