from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)
f = interpolate.interp2d(x, y, z, kind='cubic')

xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 5.01, 1e-2)
znew = f(*(xnew, ynew))
znew = interpolate.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], xnew, ynew)[0]
print(znew.shape)
plt.plot(x, z[0, :], 'ro-', xnew, znew, 'b-')
plt.show()