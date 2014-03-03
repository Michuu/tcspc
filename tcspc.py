#!/usr/bin/python
# -*- coding: utf-8 -*-
#     
#    tscpc.py
#    usage: tscpc.py irf.hhd data.hhd
#    
#    Time Correlated Single Photon Counting data analysis software
#    V2
#    Reads data from hhd files and tries to fit multiple exponential 
#    decay convoluted with irf to data
#    Michał Parniak <mparniak@gmail.com>
#
#    hhd read part adapted from Matlab implementation by Peter Kapusta, 
#    PicoQuant GmbH, June 2008
#
######################################################################

import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import sys
from datetime import datetime

hrl = '_____________________________________'

def model(p,X,numexp):
    '''
    p: model parameters
    X: time axis
    numexp: number of exponential to fit
    calculates the value of multiple exponential decay function at points given by X
    '''
    ret=p[2*(numexp-1)+2]
    for i in xrange(numexp):
        ret+=p[2*i+1]*np.exp((-1)*X/p[2*i])
    return ret
        
def residuals(p,Y,X,E,irf,numexp):
    '''
    p: model parameters
    Y: experimental data
    X: time axis
    E: errors
    irf: instument response function
    numexp: number of exponential to fit
    return: convolution of irf and model minus data weighted by errors
    convolution is calculated as an inverse fourier transform of the product of fourier transforms of irf and model
    one needs to divide the result by Y to obtain the correct result, however amplitude is not crucial in this calculation
	'''
    return (np.fft.irfft(np.fft.rfft(irf)*np.fft.rfft(model(p,X,numexp)))/len(Y)-Y)/E
    
def readhhd(filename,k):
	'''
	filname: name of hhd file
	k: curve index to read
	this function reads data of k-th curve from a hhd file given in filename
	return: curve data, resolution, comment, timestamp of measurement
	'''
	f=open(filename,'rb')

	Ident=f.read(16)

	FormatVersion=f.read(6).strip(chr(0))

	if(FormatVersion != '2.0'):
		print 'this program is only for file format 2.0'
		sys.exit(0)

	CreatorName=f.read(18)

	CreatorVersion=f.read(12)

	FileTime=f.read(18)

	CRLF=f.read(2)

	Comment=f.read(256)

	NumberOfCurves = struct.unpack('i',f.read(4))[0]

	BitsPerRecord = struct.unpack('i',f.read(4))[0]

	ActiveCurve = struct.unpack('i',f.read(4))[0]

	MeasurementMode = struct.unpack('i',f.read(4))[0]

	SubMode = struct.unpack('i',f.read(4))[0]

	Binning = struct.unpack('i',f.read(4))[0]

	Resolution = struct.unpack('d',f.read(8))[0]

	Offset = struct.unpack('i',f.read(4))[0]

	Tacq = struct.unpack('i',f.read(4))[0]

	StopAt = struct.unpack('I',f.read(4))[0];

	StopOnOvfl = struct.unpack('i',f.read(4))[0]

	Restart = struct.unpack('i',f.read(4))[0]

	DispLinLog = struct.unpack('i',f.read(4))[0]

	DispTimeAxisFrom = struct.unpack('I',f.read(4))[0]

	DispTimeAxisTo = struct.unpack('I',f.read(4))[0]

	DispCountAxisFrom = struct.unpack('I',f.read(4))[0]

	DispCountAxisTo = struct.unpack('I',f.read(4))[0]

	DispCurveMapTo=[]
	DispCurveShow=[]
	for i in range(8):
		DispCurveMapTo.append(struct.unpack('i',f.read(4))[0])
		DispCurveShow.append(struct.unpack('i',f.read(4))[0])

	ParamStart=[]
	ParamStep=[]
	ParamEnd=[]
	for i in range(3):
		ParamStart.append(struct.unpack('f',f.read(4))[0])
		ParamStep.append(struct.unpack('f',f.read(4))[0])
		ParamEnd.append(struct.unpack('f',f.read(4))[0])

	RepeatMode = struct.unpack('i',f.read(4))[0]

	RepeatsPerCurve = struct.unpack('i',f.read(4))[0]

	RepatTime = struct.unpack('i',f.read(4))[0]

	RepeatWaitTime = struct.unpack('i',f.read(4))[0]

	ScriptName = f.read(20)

	#HW

	HardwareIdent = f.read(16)

	HardwarePartNo = f.read(8)  
		
	HardwareSerial = struct.unpack('i',f.read(4))[0]

	nModulesPresent = struct.unpack('i',f.read(4))[0]

	ModelCode=[]
	VersionCode=[]
	for i in range(10):
		ModelCode.append(struct.unpack('i',f.read(4))[0])
		VersionCode.append(struct.unpack('i',f.read(4))[0])

	BaseResolution = struct.unpack('d',f.read(8))[0]

	InputsEnabled = struct.unpack('Q',f.read(8))[0]

	InpChansPresent  = struct.unpack('i',f.read(4))[0]

	RefClockSource  = struct.unpack('i',f.read(4))[0]

	ExtDevices  = struct.unpack('i',f.read(4))[0]

	MarkerSettings  = struct.unpack('i',f.read(4))[0]

	SyncDivider = struct.unpack('i',f.read(4))[0]

	SyncCFDLevel = struct.unpack('i',f.read(4))[0]

	SyncCFDZeroCross = struct.unpack('i',f.read(4))[0]

	SyncOffset = struct.unpack('i',f.read(4))[0]

	#CH
	InputModuleIndex=[]
	InputCFDLevel=[]
	InputCFDZeroCross=[]
	InputOffset=[]
	for i in range(InpChansPresent):
		InputModuleIndex.append(struct.unpack('i',f.read(4))[0])   
		InputCFDLevel.append(struct.unpack('i',f.read(4))[0])
		InputCFDZeroCross.append(struct.unpack('i',f.read(4))[0])   
		InputOffset.append(struct.unpack('i',f.read(4))[0])
	 
	#CURVE
	CurveIndex=[]
	TimeOfRecording=[]
	HardwareIdent=[]
	HardwarePartNo=[]
	HardwareSerial=[]
	nModulesPresent=[]
	ModelCode=[]
	VersionCode=[]
	BaseResolution=[]
	InputsEnabled=[]
	InpChansPresent=[]
	RefClockSource=[]
	ExtDevices=[]
	MarkerSettings=[]
	SyncDivider=[]
	SyncCFDLevel=[]
	SyncCFDZeroCross=[]
	SyncOffset=[]
	InputModuleIndex=[]
	InputCFDLevel=[]
	InputCFDZeroCross=[]
	InputOffset=[]
	InpChannel=[]
	MeasurementMode=[]
	SubMode=[]
	Binning=[]
	Resolution=[]
	Offset=[]
	Tacq=[]
	StopAfter=[]
	StopReason=[]
	P1=[]
	P2=[]
	P3=[]
	SyncRate=[]
	InputRate=[]
	HistCountRate=[]
	IntegralCount=[]
	HistogramBins=[]
	DataOffset=[]
	for i in range(NumberOfCurves):
		CurveIndex.append(struct.unpack('i',f.read(4))[0])
		TimeOfRecording.append(struct.unpack('L',f.read(4))[0])
		HardwareIdent.append(f.read(16))
		HardwarePartNo.append(f.read(8))
		HardwareSerial.append(struct.unpack('i',f.read(4))[0])
		nModulesPresent.append(struct.unpack('i',f.read(4))[0])
		ModelCode.append([])
		VersionCode.append([])
		for j in range(10):
			ModelCode[i].append(struct.unpack('i',f.read(4))[0]) 
			VersionCode[i].append(struct.unpack('i',f.read(4))[0])
		BaseResolution.append(struct.unpack('d',f.read(8))[0])
		InputsEnabled.append(struct.unpack('Q',f.read(8))[0])
		InpChansPresent.append(struct.unpack('i',f.read(4))[0])
		RefClockSource.append(struct.unpack('i',f.read(4))[0])
		ExtDevices.append(struct.unpack('i',f.read(4))[0])
		MarkerSettings.append(struct.unpack('i',f.read(4))[0])
		SyncDivider.append(struct.unpack('i',f.read(4))[0])
		SyncCFDLevel.append(struct.unpack('i',f.read(4))[0])
		SyncCFDZeroCross.append(struct.unpack('i',f.read(4))[0])
		SyncOffset.append(struct.unpack('i',f.read(4))[0])
		InputModuleIndex.append(struct.unpack('i',f.read(4))[0])
		InputCFDLevel.append(struct.unpack('i',f.read(4))[0])
		InputCFDZeroCross.append(struct.unpack('i',f.read(4))[0])
		InputOffset.append(struct.unpack('i',f.read(4))[0])
		InpChannel.append(struct.unpack('i',f.read(4))[0])
		MeasurementMode.append(struct.unpack('i',f.read(4))[0])
		SubMode.append(struct.unpack('i',f.read(4))[0])
		Binning.append(struct.unpack('i',f.read(4))[0])
		Resolution.append(struct.unpack('d',f.read(8))[0])
		Offset.append(struct.unpack('i',f.read(4))[0])
		Tacq.append(struct.unpack('i',f.read(4))[0])
		StopAfter.append(struct.unpack('i',f.read(4))[0])
		StopReason.append(struct.unpack('i',f.read(4))[0])
		P1.append(struct.unpack('f',f.read(4))[0])
		P2.append(struct.unpack('f',f.read(4))[0])
		P3.append(struct.unpack('f',f.read(4))[0])
		SyncRate.append(struct.unpack('i',f.read(4))[0])
		InputRate.append(struct.unpack('i',f.read(4))[0])
		HistCountRate.append(struct.unpack('i',f.read(4))[0])
		IntegralCount.append(struct.unpack('q',f.read(8))[0])
		HistogramBins.append(struct.unpack('i',f.read(4))[0])
		DataOffset.append(struct.unpack('i',f.read(4))[0])

	Counts=[]
	f.seek(DataOffset[k],0)
	for j in range(HistogramBins[k]):
		Counts.append(struct.unpack('I',f.read(4))[0]);
	f.close()
	return [Counts,Resolution[k],Comment.strip(chr(0)),TimeOfRecording[k]]


# ----------------- the program starts here ----------------

# check parameters

if(len(sys.argv) != 3):
    print 'error: give 2 filenames'
    print 'usage: tcspc.py irf.hhd data.hhd'
    sys.exit(0)

# load data and irf

[irfY,irfRes,Comment,ToR] =readhhd(sys.argv[1],0)
print hrl
print 'IRF loaded from '+sys.argv[1]+', time of measurement: '+str(datetime.fromtimestamp(ToR))
print 'Comment:\n'+Comment.strip()
[Y,Res,Comment,ToR] =readhhd(sys.argv[2],0)
print hrl
print 'Data loaded from '+sys.argv[1]+', time of measurement: '+str(datetime.fromtimestamp(ToR))
print 'Comment:\n'+Comment
print hrl

# calculate errors as sqrt(Y)

E =[]
for pt in Y:
        if(pt>0):
            E.append(np.sqrt(pt))
        else:
            E.append(1.0)			

if(len(irfY)!=len(Y)):
    print 'error: incompatible data and irf'
    sys.exit(0)

if(irfRes!=Res):
	print 'error: incompatible resolutions'
	sys.exit(0)

X=[i*Res for i in xrange(len(Y))]

X=np.array(X)
Y=np.array(Y)
Y=Y.astype('float')
E=np.array(E)
irfX=X
irfY=np.array(irfY)

# ask the user for additional data

# number of exps to fit
print 'give number of exponents:'
numexp=int(sys.stdin.readline())

print 'function to fit: f(t)='
for i in range(numexp):
    print 'A_'+str(i) +' exp(-t / tau_'+str(i)+') +'
print 'B'

# initial conditions

print hrl
p0=[]
print 'initial conditions:'
for i in range(numexp):
	print 'tau_'+str(i)
	p0.append(float(sys.stdin.readline()))
	print 'A_'+str(i)
	p0.append(float(sys.stdin.readline()))
print 'B'
p0.append(float(sys.stdin.readline()))

# fit

fit=leastsq(residuals,p0,args=(Y,X,E,irfY,numexp),full_output=True)

# evaluate fit results

if(fit[4]<=4 and fit[4] >=1):
    print 'fit succesful!'
    print 'converged in '+str(fit[2]['nfev'])+' steps'
else:
    print 'fit unsuccesful'
    print 'residuals evaluated '+str(fit[2]['nfev'])+' times'
    print fit[3]

# calculate the goodness of fit and uncertainties

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

# write results to text file

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

# plot

# experimental data
plt.plot(X,Y,'o')
# irf
plt.plot(irfX,irfY)
# fire results
plt.plot(X,fitres)

# labels
plt.xlabel('Time [ps]')
plt.ylabel('Counts')

# title
plt.title('Fluorescence decay fit results for '+sys.argv[2]+' and '+ sys.argv[1])

# legend
legendstring='Fit:\n'
for i in range(numexp):
    legendstring+=r'$\tau_'+str(i)+' ='+str(round(p[2*i],1))+'\pm'+str(round(s_sq*cov[2*i][2*i],1))+'$ ps\n'
plt.legend(['Experimental Data','IRF',legendstring])

plt.show()
