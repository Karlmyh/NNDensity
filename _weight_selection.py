'''
Weight Selection
'''

import numpy as np



def weight_selection(beta):
    
    potentialNeighbors=len(beta)
    alphaIndexMax=0
    lamda = beta[0]+1 
    Sum_beta = 0
    Sum_beta_square = 0

    # iterates for k
    while ( lamda>beta[alphaIndexMax] ) and (alphaIndexMax<potentialNeighbors-1):
        # update max index
        alphaIndexMax +=1
        # updata sum beta and sum beta square
        Sum_beta += beta[alphaIndexMax-1]
        Sum_beta_square += (beta[alphaIndexMax-1])**2
        
        # calculate lambda
        if  alphaIndexMax  + Sum_beta**2 - alphaIndexMax * Sum_beta_square>=0:
            lamda = (1/alphaIndexMax) * ( Sum_beta + np.sqrt( alphaIndexMax  + (Sum_beta**2 - alphaIndexMax * Sum_beta_square) ) )
        else:
            alphaIndexMax-=1
            break
    
    # estimation
    estAlpha=np.zeros(potentialNeighbors)
    if alphaIndexMax==0:
        return estAlpha, 0
    
    for j in range(alphaIndexMax):
        estAlpha[j]=lamda-beta[j]
    
    
    estAlpha=estAlpha/np.linalg.norm(estAlpha,ord=1)
    
    return estAlpha,alphaIndexMax


    
        
    