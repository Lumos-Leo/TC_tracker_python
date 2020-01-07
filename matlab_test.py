import matlab.engine
import matlab
import numpy as np
from scipy import io as io 
# eng = matlab.engine.start_matlab()
# tf = eng.isprime(37)
mat_path = 'you_mat'
a = np.ones((2,4))
# io.savemat(mat_path, {'name':a})
print(a)
# mat = io.loadmat('your')
b = matlab.double(a.tolist())
print(b)
# eng.quit()