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

print 'give number of exponents:'
numexp=int(sys.stdin.readline())

print 'function to fit: f(t)='
for i in range(numexp):
    print 'A_'+str(i) +' exp(-t / tau_'+str(i)+') +'
print 'B'

p0=[]
print 'initial conditions:'
for i in range(numexp):
	print 'tau_'+str(i)
	p0.append(float(sys.stdin.readline()))
	print 'A_'+str(i)
	p0.append(float(sys.stdin.readline()))
print 'B'
p0.append(float(sys.stdin.readline()))

def model(p,X,numexp):
    ret=p[2*(numexp-1)+2]
    for i in xrange(numexp):
        ret+=p[2*i+1]*np.exp((-1)*X/p[2*i])
    return ret
        
def residuals(p,Y,X,E,irf,numexp):
    return (np.fft.irfft(np.fft.rfft(irf)*np.fft.rfft(model(p,X,numexp)))/len(Y)-Y)/E

fit=leastsq(residuals,p0,args=(Y,X,E,irfY,numexp),full_output=True)
if(fit[4]<=4 and fit[4] >=1):
    print 'fit succesful!'
    print 'converged in '+str(fit[2]['nfev'])+' steps'
else:
    print 'fit unsuccesful'
    print 'residuals evaluated '+str(fit[2]['nfev'])+' times'
    print fit[3]

cov = fit[1]
p=fit[0]
fitres_model=model(p,X,numexp)
fitres=np.fft.irfft(np.fft.rfft(irfY)*np.fft.rfft(fitres_model))/len(Y)
s_sq = (fit[2]['fvec']**2).sum()/(len(Y)-len(p))
i_sq = (residuals(p0,Y,X,E,irfY,numexp)**2).sum()/(len(Y)-len(p))
print 'reduced chi squared = '+str(s_sq)
print 'reduced chi squared at initial point = '+str(i_sq)
for i in range(numexp):
    print 'tau_'+str(i)+'='+str(round(p[2*i],1))+' +/- '+str(round(s_sq*cov[2*i][2*i],1))+' ps'
    print 'A_'+str(i)+'='+str(p[2*i+1])+' +/- '+str(s_sq*cov[2*i+1][2*i+1])
print 'B='+str(p[2*numexp])+' +/- '+str(s_sq*cov[2*numexp][2*numexp])

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
legendstring='Fit:\n'
for i in range(numexp):
    legendstring+=r'$\tau_'+str(i)+' ='+str(round(p[2*i],1))+'\pm'+str(round(s_sq*cov[2*i][2*i],1))+'$ ps\n'
plt.legend(['Experimental Data','IRF',legendstring])
plt.show()
