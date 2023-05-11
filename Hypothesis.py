#import packages
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

#import Random class
sys.path.append(".")
import Random as rng

#import Sorting class
sys.path.append(".")
import MySort as mys

# main function
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [options]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print ("   -seed [integer number]  seed")
        print ("   -Nmeas [integer number]  number of measurements per experiments")
        print ("   -Nexp [integer number]  number of experiments")
        print ("   -lambda [float number]  Parameter for the Null Hypothesis")
        print ("   -alpha [float number]  Parameter for the Alternative Hypothesis")
        print ("   -beta [float number]  Parameter for the Alternative Hypothesis")
        print
        sys.exit(1)
        
    #Initialize 
    seed = 5555
    
    Nmeas = 1 #number of measurements per experiment
    Nexp = 1 #number of experiments
    
    lamb = 1.0 #must be positive
    alpha = 1.0 #must be positive
    beta = 1.0 #must be positive
    
    #System Inputs
    if '-seed' in sys.argv:
        p = sys.argv.index('-seed')
        seed = sys.argv[p+1]
        
    if '-Nmeas' in sys.argv:
        p = sys.argv.index('-Nmeas')
        ptemp = int(sys.argv[p+1])
        Nmeas = ptemp

    if '-Nexp' in sys.argv:
        p = sys.argv.index('-Nexp')
        ptemp = int(sys.argv[p+1])
        Nexp = ptemp
        
    if '-lambda' in sys.argv:
        p = sys.argv.index('-lambda')
        ptemp = float(sys.argv[p+1])
        lamb = ptemp
    
    if '-alpha' in sys.argv:
        p = sys.argv.index('-alpha')
        ptemp = float(sys.argv[p+1])
        alpha = ptemp
    
    if '-beta' in sys.argv:
        p = sys.argv.index('-beta')
        ptemp = float(sys.argv[p+1])
        beta = ptemp  
              
    #class instance of Random and Sorting class 
    random = rng.Random(seed)
    Sorter = mys.MySort()

    #initialize data 
    data_simple = [] #[[exp1], [exp2], ...] each experiment in the simple hypothesis
    data_graph = [] #[meas1, meas1, ...] used to plot data from simple hypothesis
    
    data_complex = [] #[[exp1], [exp2], ...] each experiment in the complex hypothesis
    hist_complex = []  #[meas1, meas1, ...] every measurement in the complex hypothesis from all lambdas
    lamb_graph = [] #[lamb1, lamb2, ...] used to plot the distribution of lambdas
    
    #Generate Data for Simple Hypothesis (fixed lambda) 
    for e in range(0, Nexp): #each experiment
        data_exp_simple = [] #all measurements in a given experiment
        
        for m in range(0, Nmeas): #each measurement
    	    measurement_simple = float(random.Poisson(lamb))
    	    data_exp_simple.append(measurement_simple)
    	    data_graph.append(measurement_simple)
    	
        data_simple.append(data_exp_simple)
    
    #print results of simple hypothesis
    print(f"Number of measurements/experiment: {Nmeas}")
    print(f"Number of experiments: {Nexp}")
    #print(data_graph) #all measurements 
    #print(data_simple[0]) #experiment 1
    
    #Graph simple hypothesis data
    n, bins, patches = plt.hist(data_graph, 16, edgecolor = 'black', linewidth = 3, density = True, facecolor = 'orange', alpha=0.75)
  
    x = np.linspace(0, 20, 1000)
    y = []	
     
    def LogPoisson(x): #Log of Poisson Distribution + Stirling Approximation
        if x == 0:
            f = -1 * lamb
        
        else: 
            f = x * np.log(lamb) - 1/2 * np.log(2 * np.pi * x) - x * np.log(x) + x - lamb  
        
        return f
      
    for i in range(len(x)):
        y.append(np.exp(LogPoisson(x[i])))
    
    plt.plot(x, y, color = 'blue', label = 'Poisson Distribution')

    plt.xlabel('x', fontsize = 15)
    plt.ylabel('P(x | $\lambda$)', fontsize = 15)
    plt.title('Poisson Distribution', fontsize = 20)
    
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.savefig('SimpleData.png')
    plt.show()
    
    #Generate Data for Complex Hypothesis (lambda comes from a Gamma Distribution)
    for e in range(0, Nexp): #each experiment
        data_exp_complex = []  #all measurements in a given experiment
        
        lamb_complex = float(random.Gamma(alpha, beta)) #new lambda every experiment
      
        lamb_graph.append(lamb_complex)
        
        for m in range(0, Nmeas): #each measurement
    	    measurement_complex = float(random.Poisson(lamb_complex))
    	    hist_complex.append(measurement_complex)
    	    data_exp_complex.append(measurement_complex)
    	
        data_complex.append(data_exp_complex)
    
    #print results of complex hypothesis
    #print(data_complex[0]) #experiment 1
    
    #Graph distribution of lambda for Complex Hypothesis
    n, bins, patches = plt.hist(lamb_graph, 16, edgecolor = 'black', linewidth = 3, density = True, facecolor = 'orange', alpha=0.75)
  
    x2 = np.linspace(0, 20, 1000)
    y2 = []	
     
    def LogGamma(L): #Log of Gamma Distribution 
        f = alpha * np.log(beta) - np.log(math.gamma(alpha)) + (alpha - 1) * np.log(L) - beta * L 
        
        return f
      
    for i in range(len(x2)):
        y2.append(np.exp(LogGamma(x2[i])))
    
    plt.plot(x2, y2, color = 'blue', label = 'Gamma Distribution')

    plt.xlabel('$\lambda$', fontsize = 15)
    plt.ylabel('P($\lambda$ | $\\alpha, \\beta$)', fontsize = 15)
    plt.title('Gamma Distribution', fontsize = 20)
    
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.savefig('LambdaDistribution.png')
    plt.show()
    
    
    
    ####Log Likelihoods
    
    #Log Likelihood for Poisson Distribution (Simple Hypothesis)
    def LogLikelihoodPoisson(lamb, data): #data is a single experiment from data_simple or data_complex.
        f = 0 #initialize value
       
        for d in data: #measurements in a single experiment
            if d == 0:
                f += 0
                
            else:
                f += d * np.log(lamb)  - 1/2 * np.log(2 * np.pi * d) - d * np.log(d) + d - lamb 
        
        return f 
    
    #Log Likelihood for Complex Hypothesis done numerically using a normalized histogram
    
    #Graph Complex Histogram in order to save bin number and bin height
    n_save, bins_save, patches = plt.hist(hist_complex, 16, edgecolor = 'black', linewidth = 3, density = True, facecolor = 'orange', alpha=0.75)
    
    #n_save is height of histogram
    #bins_save = bin edges starting from the left
    
    plt.xlabel('x', fontsize = 15)
    plt.ylabel('Non-Normalized P(x | $\\alpha, \\beta$)', fontsize = 15)
    plt.title('Complex Hypothesis', fontsize = 20)
    
    plt.legend(loc = 'upper right')
    plt.grid(True)
    
    plt.savefig('ComplexHist.png')
    plt.show()
    
    #normalize histogram to get probabilities
    #print(f'Non-normalized heights of the histogram: {n_save}')
    
    tot = 0
    for n in n_save:
        tot += n
    
    print(f'Non-normalized value of the histogram: {tot}')
    
    for n in range(len(n_save)):
       n_save[n] /= tot
    
   # print(f'Normalized heights of the histogram: {n_save}')
    
    tot_normal = 0
    for n in n_save:
        tot_normal += n
    
    print(f'Normalized value of the histogram: {tot_normal}') #should equal 1 if normalized correctly.
    
    #Log Likelihood for Complex Hypothesis
    logprob_min = np.log(1/len(hist_complex))
    
    def ComplexLikelihood(data): #takes in one experiment from data_simple or data_complex
        g = 0 #initialize value
        
        for d in data: #measurements in an experiment
            bin_current = 0
            
            #protect against going over the farthest right bin edge
            if d > bins_save[len(bins_save) - 1]:
                logprob = logprob_min
            
            #find what bin the measurement falls in, starting from the left
            else:
                while d > bins_save[bin_current + 1]:
                    bin_current += 1
            
                #protect against bins w/ no counts
                if n_save[bin_current] <= 0:
                    logprob = logprob_min
            
                else:
                    logprob = np.log(n_save[bin_current]) #once normalized, height of the histogram = probability of measurement
            
            g += logprob
            
        return g
    
    #Log Likelihood Ratios
    LL_simple_simple = [] #Log Likelihood of the Simple Hypothesis using the data made from the Simple Hypothesis
    LL_complex_simple = [] #Log Likelihood of the Complex Hypothesis using the data made from the Simple Hypothesis
    
    LL_simple_complex = [] #Log Likelihood of the Simple Hypothesis using the data made from the Complex Hypothesis
    LL_complex_complex = [] #Log Likelihood of the Complex Hypothesis using the data made from the Complex Hypothesis
    
    LLR_simple = [] #[LLR1, LLR2, ...]  LLR_simple = LL_simple_simple[i] - LL_complex_simple[i]
    LLR_complex = [] #[LLR1, LLR2, ...] LLR_complex = LL_simple_complex[i] - LL_complex_complex[i]

    #Log Likelihood of the Simple Hypothesis using the data made from the Simple Hypothesis 
    for exp in data_simple: #each experiment
        LL = LogLikelihoodPoisson(lamb, exp) 
        LL_simple_simple.append(LL)
    
    #Log Likelihood of the Complex Hypothesis using the data made from the Simple Hypothesis
    for exp in data_simple: #each experiment
        LL = ComplexLikelihood(exp)
        LL_complex_simple.append(LL)
    
    #Log Likelihood of the Simple Hypothesis using the data made from the Complex Hypothesis   
    for exp in data_complex: #each experiment
        LL = LogLikelihoodPoisson(lamb, exp) 
        LL_simple_complex.append(LL)
    
    #Log Likelihood of the Complex Hypothesis using the data made from the Complex Hypothesis
    for exp in data_complex: #each experiment
        LL = ComplexLikelihood(exp)
        LL_complex_complex.append(LL)
    
    #LLR for simple hypothesis
    for i in range(len(LL_simple_simple)):
        LLR_simple.append(LL_simple_simple[i] - LL_complex_simple[i])

    #LLR for complex hypothesis
    for i in range(len(LL_simple_complex)):
        LLR_complex.append(LL_simple_complex[i] - LL_complex_complex[i])

    #Sort the LLRs   
    LLR_simple = Sorter.DefaultSort(LLR_simple) #sort LLR in ascending order.
    LLR_complex = Sorter.DefaultSort(LLR_complex) #sort LLR in ascending order.  
    
    #print(f"Sorted list of LLR under Null Hypothesis: {LLR_simple}")
    #print(f"Sorted list of LLR under Alternative Hypothesis: {LLR_complex}")
    
    #Define Confidence Level and find critical LLR value
    alpha_CL = 0.05 #Confidence Level = 95%
    
    LLR_alpha_CL = LLR_simple[math.ceil(Nexp * alpha_CL)] #critical LLR value
    
    print(f"Alpha value: {alpha_CL}")
    print(f"The critical LLR value is: {LLR_alpha_CL}")    
    
    #Find Beta and Power of Test   
    for i in range(len(LLR_complex)):
    	if LLR_complex[i] >= LLR_alpha_CL:
    	    LLR_Beta = LLR_complex[i]
    	    LLR_Beta_Position = i
    	    print(f"LLR_Beta: {LLR_Beta}")
    	    print(f"Position of LLR_Beta in LLR_complex: {LLR_Beta_Position}")
    	    break
                
    Beta_CL = (len(LLR_complex) - LLR_Beta_Position)/Nexp #Beta_CL = perctent of entries in LLR_complex above LLR_alpha_CL
    print(f"Beta value: {Beta_CL}")
    
    Power = 1 - Beta_CL
    print(f"Power of test is: {Power}")
    
    #Create LLR figure
    title = str(Nmeas) +  " measurements / experiment"
    
    plt.figure()
    plt.hist(LLR_simple, Nmeas + 1, density = True, facecolor = 'b', alpha = 0.5, label = "assuming $\\mathbb{H}_0$")
    
    plt.hist(LLR_complex, Nmeas + 1, density = True, facecolor = 'g', alpha = 0.7, label = "assuming $\\mathbb{H}_1$")
    
    plt.legend()

    plt.xlabel('$LLR = \\log({\\cal L}_{\\mathbb{H}_{0}}/{\\cal L}_{\\mathbb{H}_{1}})$')
    plt.ylabel('Probability')
    plt.title(title)
    
    plt.grid(True)
    plt.ylim(0, 0.04)

    plt.savefig('LLRPlot.png')
    plt.show()
    
    
