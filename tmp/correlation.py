from numpy import *
set_printoptions(precision=3, suppress=True)

# Define a function which pulls out a 3x3 submatrix and turns it into a array of length 9. 
def submatrix(M,i,j):
        return M[i-1:i+2,j-1:j+2].ravel()

# Make a 400x408 CCD with a Poisson distributed mean in each pixel.   
mean = 2000
a = random.poisson(mean, (400,408) )

print "\nCalculate 2D spatial Autocorrelation for random numbers with mean", mean
print "Shape is", a.shape

# Loop over all the pixels of the CCD and make a sequence of the 1D arrays.
# The second line ignores the edges.
q = [ submatrix(a,i,j) for i,j in ndindex(a.shape)
      if i != 0 and j != 0 and i != a.shape[0]-1 and j != a.shape[1]-1]

# Stack the 1D arrays up in columns. So this is like having many measurements of X,Y etc etc..
y = column_stack(q)

# Calculate the corr. coefficients (1 upper-left corner, 5 is center-pixel, 9 lower-right)
fullCorrelation = corrcoef(y)

# Pull out the correllations of the center pixel with all the others (row 5)
# and reshape into 3x3 matrix so it looks like what we want to check.
centerPixelCorrelation = fullCorrelation[4].reshape(3,3)

# Print it out
print centerPixelCorrelation
