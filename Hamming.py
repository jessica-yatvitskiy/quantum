#Simulates Hamming Error Correction
import numpy as np

#Generate matrix of all n-bit binary arrays
binary_arrays = []
def get_bin_arrs(n, bin_str=''):
    if len(bin_str) == n:
        binary_arrays.append([eval(i) for i in list(bin_str)])
    else:
        get_bin_arrs(n, bin_str + '0')
        get_bin_arrs(n, bin_str + '1')

#From above method, generate Hamming matrix
get_bin_arrs(3)
h=np.transpose(np.array(binary_arrays)[1:][:])

#Uncorrupted message:
m=np.transpose(np.array([0, 0, 0, 1, 1, 1, 1]))
#Find location of error in message (should be 0, for no error)
error_loc_bin_arr=np.dot(h,m)%2
error_loc_bin = ''.join(str(x) for x in error_loc_bin_arr)
error_loc_dec=int(error_loc_bin ,2)
print("uncorrupted message...")
print("location of error (0 means no error):")
print(error_loc_dec)

#Corrupted message (changed 3rd bit from left)
m=np.transpose(np.array([0, 0, 1, 1, 1, 1, 1]))
#Find location of error in message (should be 3)
error_loc_bin_arr=np.dot(h,m)%2
error_loc_bin = ''.join(str(x) for x in error_loc_bin_arr)
error_loc_dec=int(error_loc_bin ,2)
print("corrupted message...")
print("location of error (0 means no error):")
print(error_loc_dec)
