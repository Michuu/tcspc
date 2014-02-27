import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import sys

if(len(sys.argv) != 3):
    print 'error: give 2 filenames'
    print 'usage: tcspc.py irf.txt data.txt'
    sys.exit(0)
    
data = [line.strip().split() for line in open(sys.argv[1], 'r')]
irfX =[]
irfY =[]
for line in data:
    if(line[0] != '' and line[1] != ''):
        irfX.append(float(line[0]))
        irfY.append(float(line[1]))

data = [line.strip().split() for line in open(sys.argv[2], 'r')]
X =[]
Y =[]
E =[]
for line in data:
    if(line[0] != '' and line[1] != ''):
        X.append(float(line[0]))
        Y.append(float(line[1]))
        if(float(line[1])>0):
            E.append(np.sqrt(float(line[1])))
        else:
            E.append(1.0)			

if(irfX!=X):
    print 'error: incompatible data and irf'
    sys.exit(0)
    
X=np.array(X)
Y=np.array(Y)
E=np.array(E)
irfX=np.array(irfX)
irfY=np.array(irfY)



def residuals(p,Y,X,E,irf):
    model=p[1]*np.exp((-1)*X/p[0])+p[2]
    return (np.fft.irfft(np.fft.rfft(irf)*np.fft.rfft(model))-Y)/E
    
p0=[3500.0,1000.0,1.0]
fit=leastsq(residuals,p0,args=(Y,X,E,irfY),full_output=True)
if(fit[4]<=4 and fit[4] >=1):
    print 'fit succesful!'
    print 'converged in '+str(fit[2]['nfev'])+' steps'
else:
    print 'fit unsuccesful'
    print 'residuals evaluated '+str(fit[2]['nfev'])+' times'
    print fit[3]

cov = fit[1]
p=fit[0]
fitres_model=p[1]*np.exp((-1)*X/p[0])+p[2]
fitres=np.fft.irfft(np.fft.rfft(irfY)*np.fft.rfft(fitres_model))
s_sq = (fit[2]['fvec']**2).sum()/(len(Y)-len(p))
i_sq = (residuals(p0,Y,X,E,irfY)**2).sum()/(len(Y)-len(p))
print 'reduced chi squared = '+str(s_sq)
print 'reduced chi squared at initial point = '+str(i_sq)
print 'tau='+str(round(p[0],1))+' +/- '+str(round(s_sq*cov[0][0],1))+' ps'


f=open(sys.argv[2][:-4]+'_results.txt','w')
i=0
for point in fitres:
    f.write(str(X[i]))
    f.write(' ')
    f.write(str(point))
    f.write("\n")
    i+=1
f.close()
print 'fit data saved to '+sys.argv[2][:-4]+'_results.txt'


plt.plot(X,Y,'o')
plt.plot(irfX,irfY)
plt.plot(X,fitres)
plt.xlabel('Time [ps]')
plt.ylabel('Counts')
plt.title('Fluorescence decay fit results for '+sys.argv[2]+' and '+ sys.argv[1])
plt.legend(['Experimental Data','IRF',r'Fit: $ \tau ='+str(round(p[0],1))+'\pm'+str(round(s_sq*cov[0][0],1))+'$ ps'])
plt.show()
