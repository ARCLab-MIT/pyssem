# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:39:13 2019

@author: mkb1g12
@author: Matthew Brown, UoB SERENE

A.Rossi 2006 Presentation @ 24th IADC
NASA Breakup Model Implementation Comparison of Results
https://openportal.isti.cnr.it/data/2006/120331/2006_120331.pdf
"""

import csv
import numpy

##############################
### Johnson 2001 Equations ###
##############################

def catastrophicOrNot(M1, M2):
    
    V2 = 93.1225 # (9.65 km/s)**2
    
    if M1*V2/M2 >= 8e-8:
        return True
    return False
    

def normalDist(mu, sigma):
    """
    Returns a normal distribution with mean mu and sigma sigma
    """
    #chi = sigma*numpy.random.normal(loc=0, scale=1)+mu #manipulate a standard normal distribution
    chi = numpy.random.normal(mu, sigma)
    return chi

def AxfromLc(Lc):
    """
    Area from characteristic length, equations 8 and 9
    
    Lc in m
    Ax in m^2
    """
    if Lc < 0.00167:
        Ax = 0.540424 * Lc * Lc
    else:
        Ax = 0.556945 * (Lc**2.0047077)
    return Ax

def distFuncUpperStageLargerThan11cm(Lc):
    """
    Produces a random A/M ratio for an upper stage of given characteristic length larger than 11cm
    Uses the distribution function for upper stage functions with Lc larger than 11cm (Equation 5 Johnson 2001)
    """
    lambdaC = numpy.log10(Lc)  
    
    if lambdaC <= -1.4:
        alphaRB = 1.0
    elif lambdaC >= 0:
        alphaRB = 0.5
    else:
        alphaRB = 1.0-0.3571*(lambdaC+1.4)
        
    if lambdaC <= -0.5:
        mu1RB = -0.45
    elif lambdaC >= 0:
        mu1RB = -0.9
    else:
        mu1RB = -0.45-0.9*(lambdaC+0.5)
        
    sigma1RB = 0.55
    mu2RB = -0.9
    
    if lambdaC <= -1.0:
        sigma2RB = 0.28
    elif lambdaC >= 0.1:
        sigma2RB = 0.1
    else:
        sigma2RB = 0.28-0.1636*(lambdaC+1.0)
        
    DAtoMRB = 10**(alphaRB*normalDist(mu1RB, sigma1RB)+(1-alphaRB)*normalDist(mu2RB, sigma2RB))
    #Random A/M ratio from the distribution for a rocket body fragment >11cm
    return DAtoMRB

def distFuncSpacecraftLargerThan11cm(Lc):
    """
    Produces a random A/M ratio for a spacecraft of given characteristic length larger than 11cm
    Uses the distribution function for spacecraft functions with Lc larger than 11cm (Equation 6 Johnson 2001)
    """
    lambdaC = numpy.log10(Lc) 
    
    if lambdaC <= -1.95:
        alphaSC = 0.0
    elif lambdaC >= 0.55:
        alphaSC = 1.0
    else:
        alphaSC = 0.3+0.4*(lambdaC+1.2)
        
    if lambdaC <= -1.1:
        mu1SC = -0.6
    elif lambdaC >= 0:
        mu1SC = -0.95
    else:
        mu1SC = -0.6-0.318*(lambdaC+1.1)
        
    if lambdaC <= -1.3:
        sigma1SC = 0.1
    elif lambdaC >= -0.3:
        sigma1SC = 0.3
    else:
        sigma1SC = 0.1+0.2*(lambdaC+1.3)
        
    if lambdaC <= -0.7:
        mu2SC = -1.2
    elif lambdaC >= -0.1:
        mu2SC = -2.0
    else:
        mu2SC = -1.2-1.333*(lambdaC+0.7)
    
    if lambdaC <= -0.5:
        sigma2SC = 0.5
    elif lambdaC >= -0.3:
        sigma2SC = 0.3
    else:
        sigma2SC = 0.5-(lambdaC+0.5)
        
    DAtoMSC = 10**(alphaSC*normalDist(mu1SC, sigma1SC)+(1-alphaSC)*normalDist(mu2SC, sigma2SC))
    #Random A/M ratio from the distribution for a spacecraft fragment >11cm
    return DAtoMSC

def distFuncSmallerThan8cm(Lc):
    """
    Produces a random A/M ratio for an object of given characteristic length smaller than 8cm
    Uses the distribution function for upper stage or rocket body functions with Lc smaller than 8cm (Equation 7 Johnson 2001)
    """
    lambdaC = numpy.log10(Lc)
    
    if lambdaC <= -1.75:
        muSOC = -0.3
    elif lambdaC >= -1.25:
        muSOC = -1.0
    else:
        muSOC = -0.3-1.4*(lambdaC+1.75)
        
    if lambdaC <= -3.5:
        sigmaSOC = 0.2
    else:
        sigmaSOC = 0.2+0.1333*(lambdaC+3.5)
        
    DAtoMSOC = 10**normalDist(muSOC, sigmaSOC)
    #Random A/M ratio from the distribution for a fragment <8cm
    return DAtoMSOC

def distFuncAtoM(Lc):
    """
    Produces a random A/M ratio for a spacecraft of given characteristic length
    Uses distribution functions for larger than 11cm and smaller than 8cm.
    Uses a homebrew bridging function for spacecraft between 8 and 11cm
    
    Returns DAtoM which is a random A/M ratio for the given Lc
    """
    if Lc >= 0.11:
#        DAtoM = distFuncSpacecraftLargerThan11cm(Lc)
        DAtoM = distFuncUpperStageLargerThan11cm(Lc)
    elif Lc <= 0.08:
        DAtoM = distFuncSmallerThan8cm(Lc)
    else: #bridging function
       mu    = -0.9798961012694807 - 0.02010389873051932*((0.11-Lc)/(0.03))
       sigma = 0.5 + 0.0203318952660261*((0.11-Lc)/(0.03))
       DAtoM = 10**normalDist(mu, sigma)
    return DAtoM

def createDeltaVCollision(AtoM):
    """
    Produces a random deltaV for a fragment of given A/M ratio created by a collision
    Uses the distribution function given by Johnson 2001 in equation 12
    """
    chi       = numpy.log10(AtoM)
    muColl    = 0.9*chi+2.9
    sigmaColl = 0.4
    
    # DdeltaVColl = normalDist(muColl, sigmaColl) #m/s
    # return DdeltaVColl/1000 #km/s

    # DdeltaVColl = normalDist(muColl, sigmaColl)/1000 #km/s
    # return 10**DdeltaVColl
    
    DdeltaVColl = 10**normalDist(muColl, sigmaColl) #m/s
    return DdeltaVColl/1000 #km/s

def createFragmentFromLc(Lc):
    """
    Creates all the relevant characteristics for a fragment of given characteristic length
    """
    AtoM  = distFuncAtoM(Lc)
    Ax    = AxfromLc(Lc)
    Mfrag = Ax/AtoM #Mass found by equation 10 of Johnson 2001
    DV = numpy.real(createDeltaVCollision(AtoM)) #Sample for deltaV
    
    return AtoM, Ax, Mfrag, DV

def numOfFragsOfGivenLcAndLarger(M, Lc):
    """
    Power law distribution for the number of fragments of a given size and larger created by a collision
    involving a given total mass
    """
    N = 0.1 * (M**0.75) * (Lc**-1.71)
    return N
    
def binnedNumbersOfFragments(M, minLc, maxLc, numBins):
    """
    Creates an array of fragments within specificed Lc bins.
    Orbital Debris Quarterly News Volume 15, Issue 4, October 2011 says the correct implementation
    is to distribute fragments from 1mm to 1m following the power law distribution
    """
    binSides = numpy.logspace(minLc, maxLc, numBins+1)
    
    bins = numpy.zeros_like(binSides[1:])
    
    for binNum, (smallSide, largeSide) in enumerate(zip(binSides[:-1], binSides[1:])):
        largeNumFrags = numOfFragsOfGivenLcAndLarger(M, smallSide)
        smallNumFrags = numOfFragsOfGivenLcAndLarger(M, largeSide)
        bins[binNum]  = largeNumFrags-smallNumFrags
    
    return bins

#############################################################
### UNCERTAIN PARTS NOT EXPLICITLY STATED IN JOHNSON 2001 ###
#############################################################

def randomPointInLogBin(logBinSide1, logBinSide2):
    """
    Pull a random point from a bin
    Mainly used to sample Lc bins
    
    Currently not working correctly, doesn't match NASA example data
    """
    
    randomPoint = numpy.random.uniform(logBinSide1, logBinSide2)
    
#    side1 = numpy.log10(logBinSide1)
#    side2 = numpy.log10(logBinSide2)
#    randomPoint = 10**numpy.random.uniform(side1, side2)
    
    return randomPoint

#################################
### Creating larger fragments ###
#################################
    
def LcFromAx(Ax):
    """
    Johnson 2001 Equation 9 reversed. Only eq 9 and not 8 as this will be applied to Lc>1m
    """
    
    Lc = (Ax/0.556945)**(0.49882583879934217)
    
    return Lc
    

def createRandomMasses(massTarget):
    
    numFragmentsLargerThan1m = numpy.random.randint(2,9) #Need 2 to 8 fragments larger than 1m in size according to ODQN
    # Least massive 1m fragment is around 2kg, so don't want smaller than that
    
    #https://www.reddit.com/r/CasualMath/comments/3xah7i/how_would_i_divide_a_number_into_n_numbers_such/
    #"Choose a random number, x between 0 and 300, then choose a y between 0 and 300-x, repeat 50 times. Simple recursion. Can't be an integer answer though."
    #Adapted by adding a factor of 0.7 so it is harder to pull all the large fragments at the start and leave smaller than 1 towards the end
    
    massTargetCheck = massTarget
    
    masses = []
    for i in range(1, numFragmentsLargerThan1m):
        masses += [numpy.random.uniform(1,0.7*massTargetCheck)]
        massTargetCheck -= masses[-1]  
    #Add last mass by conservation
    masses += [massTarget-sum(masses)]

    return masses

####################################################################
### Creating the full spectrum of fragments from a fragmentation ###
####################################################################

def createFragmentsForGivenLcBin(numFrags, minLcBinSide, maxLcBinSide, totalMassOfFrags, totalMassInput):
    """
    Creates the fragments for a given Lc bin
    Following the NASA example data's format, rather than sampling a huge amount of small fragments,
    group them off into 100s, then 10s. Also follows the same layout as the example data.
    
    [count, size of sample, characteristic length, mass, area, A/M ratio, deltaV]
    
    MassOfFragsLcBin keeps track of the total mass
    """
    
    
    frags = []

    Nsamp = 1    
    if numFrags/10 >= 1: #Can fit groups of ten in
        Nsamp = 10
    if numFrags/100 >= 1: #Can fit groups of one hundred in
        Nsamp = 100
    
    for ct in range(int(numFrags/Nsamp)): #Loop with Nsamp fragments using same sample
        Lc = randomPointInLogBin(minLcBinSide, maxLcBinSide) #Sample Lc in the given bin
        AtoM, Area, Mfrag, DV = createFragmentFromLc(Lc) #Sample AtoM then calculate Ax and M
        if totalMassOfFrags+Nsamp*Mfrag > totalMassInput: #Mass conservation check
#            print (numFrags, Nsamp, "Mass conservation")
            return frags, totalMassOfFrags
        totalMassOfFrags += Nsamp*Mfrag
        frags += [[ct, Nsamp, Lc, Mfrag, Area, AtoM, DV]]
    if Nsamp != 1: #Fill up the number of samples. If Nsamp=1 then it's all handled in the loop above 
        Nsamp = int(numFrags) % Nsamp #Then fill up the number of samples to the total with a number between 1 and 99
        Lc = randomPointInLogBin(minLcBinSide, maxLcBinSide) #Sample Lc in the given bin
        AtoM, Area, Mfrag, DV = createFragmentFromLc(Lc) #Sample AtoM then calculate Ax and M
        if totalMassOfFrags+Nsamp*Mfrag > totalMassInput: #Mass conservation check
#            print (numFrags, Nsamp, "Mass conservation")
            return frags, totalMassOfFrags
        totalMassOfFrags += Nsamp*Mfrag
        frags += [[ct+1, Nsamp, Lc, Mfrag, Area, AtoM, DV]]
        
    return frags, totalMassOfFrags
        
def createFragmentsLargerThan1m(massInput, totalMassFragsLessThan1m):
    """
    Creates the 2 to 8 fragments larger than 1m as suggested by Johnson 2001
    
    Sampling of these objects isn't specified so this is a best effort until I ask Sam.
    Output is same layout as smaller fragments
    """
    fragmentMasses = createRandomMasses(massInput-totalMassFragsLessThan1m)
    
    largerFragments = []
    
    for fragmentMass in fragmentMasses:
        
        AtoM = 10**(numpy.random.uniform(-1.3,-1.7)) #random sample, ####Not sure how####
        Area = fragmentMass*AtoM
        Lc   = LcFromAx(Area)
        DV = createDeltaVCollision(AtoM) #Sample for deltaV
        
        largerFragments += [[0, 1, Lc, fragmentMass, Area, AtoM, DV]]
    
    return largerFragments

def createFragmentsLikeNASAdata(M1, M2, numBins=100, printOut=0):
    """
    Creates fragments for a collision between objects of mass M1 and M2.
    Following the NASA example data's format, rather than sampling a huge amount of small fragments,
    group them off into 100s, then 10s. Also follows the same layout as the example data.
    
    [count, size of sample, characteristic length, mass, area, A/M ratio, deltaV]
    
    Orbital Debris Quarterly News Volume 15, Issue 4, October 2011 says the correct implementation
    is to distribute fragments from 1mm to 1m following the power law distribution
    2 to 10 fragments larger than 1m are then calculated.
    """
    
    if M1<M2 :
        smallMass = M1
        largeMass = M2
    else:
        smallMass = M2
        largeMass = M1
    
    ### Replace the below with a an average of the orbit? ###
    Vsmall = 9.65 #km/s Value from Rossi 1994

    #Catastrophic decision
    if ((smallMass*Vsmall*Vsmall)/(largeMass)) > 0.08: #40,000 Jkg^-1 collision catastrophic limit, factor of 2 from KE, divide by (1000)**2 so V can be input in km/s
        catastrophic  = 1
        M = M1+M2 #Total mass of input objects
    else: #Non-catastrophic
        catastrophic  = 0
        M = smallMass*Vsmall*Vsmall #This line is the ODQN October 2011 correction. From just smallMass*Vsmall
    
    LcBinSides   = numpy.logspace(-3, 1, numBins+1)
    
    fragmentsOut = []
    
    #Create the fragments in the 1mm to 1m range
    
    #Large piece in
    fragDist = binnedNumbersOfFragments(M, -3, 1, numBins)
    M1fragmentsOut = []
    totalMassOfFrags = 0.0
    for i, numFragsGivenSize in enumerate(fragDist):
        frags, totalMassOfFrags = createFragmentsForGivenLcBin(numFragsGivenSize, LcBinSides[i], LcBinSides[i+1], totalMassOfFrags, M1)
        M1fragmentsOut += frags
    fragmentsOut += M1fragmentsOut

    #Small piece in
    fragDist = binnedNumbersOfFragments(M2, -3, 1, numBins)
    M2fragmentsOut = []
    totalMassOfFrags = 0.0
    for i, numFragsGivenSize in enumerate(fragDist):
        frags, totalMassOfFrags = createFragmentsForGivenLcBin(numFragsGivenSize, LcBinSides[i], LcBinSides[i+1], totalMassOfFrags, M2)
        M2fragmentsOut += frags
    fragmentsOut += M2fragmentsOut
    
    totalMassOfFrags = numpy.dot(numpy.array(fragmentsOut)[:,1], numpy.array(fragmentsOut)[:,3])
    
    if M - totalMassOfFrags > 1.0: # If still more than 1kg of fragments to create
        largerFragments = createFragmentsLargerThan1m(M, totalMassOfFrags)
        fragmentsOut += largerFragments
        
    fragmentsOut = numpy.array(fragmentsOut)
        
    if printOut:
        
        print ("Comparison against cross-validation")
        print ("NASA Breakup Model Implementation Comparison of results - Presentation by A.Rossi 2006 at the 24th IADC\n")
        numLargeInFrags = numpy.sum(numpy.array(M1fragmentsOut)[:,1])
        print ("Fragments from object of mass 1000kg: {}, target is 2872327, {}".format(int(numLargeInFrags), 2872327/numLargeInFrags))
        numSmallInFrags = numpy.sum(numpy.array(M2fragmentsOut)[:,1])
        print ("Fragments from object of mass 10kg: {}, target is 84832, {}".format(int(numSmallInFrags), 84832/numSmallInFrags))  
        print ("Total number of fragments >1mm: {}, target is 2957159".format(int(numLargeInFrags+numSmallInFrags)))
        print ("Total Mass: {} kg, target is 1010kg".format(numpy.dot(fragmentsOut[:,1], fragmentsOut[:,3])))
        numFragsLargerThan10cm = 0
        for fragment in fragmentsOut:
            if fragment[2]>0.1:
                numFragsLargerThan10cm += fragment[1]
        print ("Number of fragments >10cm: {}, target is 862".format(int(numFragsLargerThan10cm)))
        numFragsLargerThan1g = 0
        for fragment in fragmentsOut:
            if fragment[3]>0.001:
                numFragsLargerThan1g += fragment[1]
        print ("Number of fragments >1g: {}, target is 12600".format(int(numFragsLargerThan1g)))
        numFragsLargerThan1cm2 = 0
        for fragment in fragmentsOut:
            if fragment[4]>0.0001:
                numFragsLargerThan1cm2 += fragment[1]
        print ("Number of fragments >1cm2: {}, target is 28892".format(int(numFragsLargerThan1cm2)))
        numFragsFasterThan1kms = 0
        for fragment in fragmentsOut:
            if fragment[6]>1.0:
                numFragsFasterThan1kms += fragment[1]
        print ("Number of fragments >1km/s: {}, target is 638537".format(int(numFragsFasterThan1kms)))
        
    return fragmentsOut, catastrophic

######################################################
### Adapting and manipulating the output fragments ###
######################################################

def binMasses(massBinSides, fragments):
    """
    Put masses created by createFragmentsLikeNASAdata into mass bins defined by massBinSides
    """
    
    masses     = fragments[:,3]
    massCounts = fragments[:,1]
    
    massBins = numpy.zeros_like(massBinSides[1:])
    numBins  = len(massBins)
    
    for count, mass in zip(massCounts, masses):
        for binNum in range(numBins):
            if massBinSides[binNum+1] > mass >= massBinSides[binNum]:
                massBins[binNum] += count
                break
    
    return massBins

def fragmentsToFile(fragments, filename='fragments.out'):
    """
    Write the fragments created by createFragmentsLikeNASAdata to a csv file
    Does so similarly to the NASA example data's format, without headers
    """
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for fragment in fragments:
            writer.writerow(fragment)
            
##########################
### Plotting fragments ###
##########################
            
def xplotFragments(fragments):
    """
    Plot the fragments in similar graphs as those from the cross-validation
    This allows for quick by eye validation of the SBM model
    
    """
    import matplotlib.pyplot as plt
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    #Plotting Lcs
    LcImg = plt.imread("LcValidationPic.png")
    ax1.set_xlim([-3,1])
    ax1.set_ylim([0, 700000])
    ax1.imshow(LcImg, extent=(-3,1,0,700000), aspect="auto")
    Lcs = numpy.log10(fragments[:,2])
    ax1.hist(Lcs, bins=100, weights=fragments[:,1], color="green",alpha=.5)
#    ax1.plot([-3,1],[680000,680000], label="ESA")
#    ax1.plot([-3,1],[450000,450000], label="NASA")
#    ax1.plot([-3,1],[360000,360000], label="ASI")
#    ax1.legend()
    ax1.set_xlabel("Log$_{10}$L [m]")
    ax1.set_ylabel("Number of objects")
    
    massImg = plt.imread("MassValidationPic.png")
    ax2.set_xlim([-8,4])
    ax2.set_ylim([0, 600000])
    ax2.imshow(massImg, extent=(-8,4,0,600000), aspect="auto")
    masses = numpy.log10(fragments[:,3])
    ax2.hist(masses, bins=100, weights=fragments[:,1], color="green", alpha=.5)
#    ax2.plot([-8,4],[530000,530000], label="ESA")
#    ax2.plot([-8,4],[310000,310000], label="NASA")
#    ax2.plot([-8,4],[250000,250000], label="ASI")
#    ax2.legend()
    ax2.set_xlabel("Log$_{10}$M [kg]")
    ax2.set_ylabel("Number of objects")

    areaImg = plt.imread("AreaValidationPic.png")
    ax3.set_xlim([-7,2])
    ax3.set_ylim([0, 700000])
    ax3.imshow(areaImg, extent=(-7,2,0,700000), aspect="auto")
    areas = numpy.log10(fragments[:,4])
    ax3.hist(areas, bins=100, weights=fragments[:,1], color="green", alpha=.5)
#    ax3.plot([-7,2],[670000,670000], label="ESA")
#    ax3.plot([-7,2],[440000,440000], label="NASA")
#    ax3.plot([-7,2],[350000,350000], label="ASI")
#    ax3.legend()
    ax3.set_xlabel("Log$_{10}$A [m$^2$]")
    ax3.set_ylabel("Number of objects")

    deltaVImg = plt.imread("DeltaVValidationPic.png")    
    ax4.set_xlim([-3.5,1])
    ax4.set_ylim([0, 300000])
    ax4.imshow(deltaVImg, extent=(-3.5,1,0,300000), aspect="auto")
    deltaVs = numpy.log10(fragments[:,5])
    print(deltaVs.shape)
    ax4.hist(deltaVs, bins=100, weights=fragments[:,1], color="green", alpha=.5)
#    ax4.plot([-3.5,1],[250000,250000], label="ESA")
#    ax4.plot([-3.5,1],[110000,110000], label="NASA")
#    ax4.plot([-3.5,1],[ 80000, 80000], label="ASI")
#    ax4.legend()
    ax4.set_xlabel("Log$_{10}\Delta$V [km/s]")
    ax4.set_ylabel("Number of objects")
       
def numOfFragmentsBiggerThan1gramRandomTest(runs):
    
    StoreAllNumFragsLargerThan1g = []
    
    for i in range(runs):
        if i % 100 == 0:
            print("Run {}".format(i))
        fragments, catastrophic = createFragmentsLikeNASAdata(1000, 10, 100, printOut=0)
        numFragsLargerThan1g = 0
        for fragment in fragments:
            if fragment[3]>0.001:
                numFragsLargerThan1g += fragment[1]
        StoreAllNumFragsLargerThan1g += [numFragsLargerThan1g]
        #print ("Number of fragments >1g: {}, target is 12600".format(int(numFragsLargerThan1g)))
        
    StoreAllNumFragsLargerThan1g = numpy.array(StoreAllNumFragsLargerThan1g)
    print ("NASA target is 12600")
    averageLargerThan1g = numpy.mean(StoreAllNumFragsLargerThan1g)
    errorLargerThan1g   = numpy.std(StoreAllNumFragsLargerThan1g)
    print ("After {} runs, our mean is {}±{}".format(runs, averageLargerThan1g, errorLargerThan1g))
    
    import matplotlib.pyplot as pyplot
    
    pyplot.figure()
    pyplot.hist(StoreAllNumFragsLargerThan1g, bins=50)
    pyplot.title("Histogram of my NASA SBM implementation, # objects larger than 1g")
    pyplot.xlabel("# Objects")
    pyplot.figtext(0.65,0.85,"NASA target is 12600")
    pyplot.figtext(0.65,0.8,"{} runs".format(runs))
    pyplot.figtext(0.65,0.75,"{:.2f}±{:.2f}".format(averageLargerThan1g, errorLargerThan1g))
    
    return StoreAllNumFragsLargerThan1g
     
if __name__ == "__main__":
    
    fragments, catastrophic = createFragmentsLikeNASAdata(1000, 10, 100, printOut=1)
    print (fragments.shape)