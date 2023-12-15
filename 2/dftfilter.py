import numpy as np

# Size padding
def paddedsize(array):
	"""Perform estimation of array size for zero-padding
	required to avoid circular effects during filtering
	in frequency domain.
	   array - input 2d signal
	   returns new size
	"""
	m, n = np.shape(array)
	
	return (2 * m, 2 * n)
	
def dftuv(size):
	"""Generates meshgrid for filter in frequency domain
	   size - requred size
	   returns U, V - meshgrid for filter calculation
	"""
	M, N = size
	u = np.arange(0, M, 1)
	v = np.arange(0, N, 1)
	u[(M // 2):] = u[(M // 2):] - M
	v[(N // 2):] = v[(N // 2):] - N
	U, V = np.meshgrid(u, v)
	
	return U, V

# Created Low-pass filter
def lp_filter(type, size, D0, *args):
	""" Creates low pass filter.
	    type - `ideal', `gaussian', `btw' (Batterwoth)
		size - filter size (M, N)
		D0 - filter radius
		n - Butterworth filter order (optional, n = 2 by default)
	"""
	U, V = dftuv(size)
	D = np.sqrt(U * U + V * V)
	#D = np.fft.fftshift(D)
	
	if ('ideal' == type):
		H = (D <= D0)
		
	elif ('gaussian' == type):
		H = np.exp(-(D * D) / (2 * (D0 * D0)))
	
	elif ('btw' == type):
		if (len(args) == 0):
			n = 2	# default order
		else:
			n = args[0]
		H = 1 / (1 + (D / D0) ** (2 * n))
	else:
		print('Unsupported filter type: {}, '
		      'creating ideal filter'.format(type))
		H = (D <= D0)

	return H

# Creates High-pass filter
def hp_filter(type, size, *args):
	""" Creates high pass filter.
	    type - `ideal', `gaussian', `btw' (Batterwoth), `laplacian'
		size - filter size (M, N)
		D0 - filter radius
		n - Butterworth filter order (optional, n = 2 by default)
	"""
	if ('laplacian' == type):
		U, V = dftuv(size)
		H = -(U * U + V * V)
		return H
	
	if (len(args) == 1):
		return (1 - lp_filter(type, size, args[0]))
		
	if (len(args) == 2):
		return (1 - lp_filter(type, size, args[0], args[1]))

# Filtering
def filter(f, H):
	"""Perform filtering in frequncy domain.
	   f - image in spatial domain
	   H - filter in frequncy domain, H may be 
	       generated by `lp_filter' or `hp_filter' functions
	"""
	m, n = f.shape
	M, N = H.shape

	# Transition f to the frequency domain
	F = np.fft.fft2(f, [M, N])

	# Convolution in frequency domain
	G = F * H

	# Transition G to the spatial domain
	g = np.fft.ifft2(G)

	# Cropping 
	g = g[0:m, 0:n].real

	return g

# Simplified freqz2 procedure
def freqz2(h, M, N):
	"""h - filter in spatial domain
	returns filter in frequency domain of size (M, N)
	"""
	m, n = np.shape(h)
	a = np.zeros((M, N))
	
	i = int(np.ceil((M - m) / 2))
	j = int(np.ceil((N - n) / 2))
	a[i:(i + m), j:(j + n)] = h;
	
	H = np.fft.fft2(np.fft.fftshift(a))

	return H
