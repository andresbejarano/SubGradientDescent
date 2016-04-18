#
# CS578 - Statistical Machine Learning
# Homework 3
# 
# By: Andres Bejarano <abejara@cs.purdue.edu>
#

import sys
import csv
import numpy

# Define the names of the set files
trainingDataFile = "data/train.csv"
validatingDataFile = "data/validation.csv"
testingDataFile = "data/test.csv"
neutralWordsFile = "data/neutral.csv"

# The dictionaries
dictionary = {}
neutralDictionary = {}

# The data sets
trainingSet = []
validatingSet = []
testingSet = []


##	+--------------------------+
##	| Predict a single example |
##	+--------------------------+
def predict_one(weights, words, label, bias):
    
    # Calculate the dot product (learned label)
    dot = 0.0
    for word in words:
        dot += weights[dictionary[word]]
    
    if label * (dot + bias) <= 1:
        return 0
    else:
        return label


##	+---------------------+
##	| Parse the arguments |
##	+---------------------+
def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
	parseArgs([ 'template.py', '-i', '10', '-r', 'l1', '-s', '0.4', '-l', '0.5', '-f', '1' ]) = {'-i':'10', '-r':'l1', '-s:'0.4', '-l':'0.5', '-f':1 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map


##	+---------------------+
##	| Validate the inputs |
##	+---------------------+
def validateInput(args):
    args_map = parseArgs(args)
    
    maxIterations = 1000 # the maximum number of iterations. should be a positive integer
    regularization = 'l1' # 'l1' or 'l2'
    stepSize = 0.1 # 0 < stepSize <= 1
    lmbd = 0.1 # 0 < lmbd <= 1
    featureSet = 1 # 1: original attribute, 2: pairs of attributes, 3: both

    if '-i' in args_map:
      maxIterations = int(args_map['-i'])
    if '-r' in args_map:
      regularization = args_map['-r']
    if '-s' in args_map:
      stepSize = float(args_map['-s'])
    if '-l' in args_map:
      lmbd = float(args_map['-l'])
    if '-f' in args_map:
      featureSet = int(args_map['-f'])

    assert maxIterations > 0
    assert regularization in ['l1', 'l2']
    assert stepSize > 0 and stepSize <= 1
    assert lmbd > 0 and lmbd <= 1
    assert featureSet in [1, 2, 3]
	
    return [maxIterations, regularization, stepSize, lmbd, featureSet]


##	+-------------------------------------------------------------+
##	| Build the data set from the indicated file and the set type |
##	+-------------------------------------------------------------+
def buildSet(filename, featureSet, useDictionary=1):

	# Initialize the dictionary and the data
	wordCount = 0
	
	# The set to be built
	localSet = []
	nSet = 0
	pSet = 0
	
	# Read the given file
	with open(filename, 'rb') as csvFile:
		
		# Read the file delimited by commas, ignore quote marks
		reader = csv.reader(csvFile, delimiter=',', skipinitialspace=True)
		
		# For each line in the file
		for line in reader:
			
			# Get the sentence from the line and remove grammatical symbols
			sentence = line[0]
			sentence = sentence.replace(',', ' ');
			sentence = sentence.replace('.', ' ');
			sentence = sentence.replace('[', ' ');
			sentence = sentence.replace(']', ' ');
			sentence = sentence.replace(':', ' ');
			sentence = sentence.replace(';', ' ');
			sentence = sentence.replace(' - ', ' ');
			sentence = sentence.replace(' \'', ' ');
			
			# Split sentence into words
			words = str.split(sentence)
			
			# Keep only not neutral words
			words = [x for x in words if not x in neutralDictionary]
			
			# Save the label of the entry (+1 for +, -1 for -)
			yt = 1 if line[1] == '+' else -1
			if yt == 1:
				pSet += 1
			
			# Initialize the feature set
			xt = []
			
			# Build unigrams
			if featureSet == 1:
				
				# For each valid word in the sentence
				for word in words:
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
			
			# Build bigrams
			elif featureSet == 2:
				
				n = len(words) - 1
				for i in range(0, n):
					
					# Build the bigram
					word = (words[i], words[i + 1])
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
			
			# Build unigrams-bigrams
			else:
				
				# Add unigrams first
				for word in words:
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
				
				# Now add bigrams
				n = len(words) - 1
				for i in range(0, n):
					
					# Build the bigram
					word = (words[i], words[i + 1])
					
					# Add word to dictionary in case it's not there
					if useDictionary == 1 and not word in dictionary:
						dictionary[word] = wordCount
						wordCount += 1
					
					# Add word to xt
					xt.append(word)
				
			# Add tuple to the set
			if len(xt) > 0:
				localSet.append((xt, yt))
				nSet += 1
	
	# Return the generated set and the counters
	return localSet, nSet, pSet


##	+-----------------------------------------------+
##	| Generates a dictionary with the neutral words |
##	+-----------------------------------------------+
def buildNeutralDictionary(filename):
	with open(filename, 'rb') as csvFile:
		
		# Read the file delimited by commas, ignore quote marks
		reader = csv.reader(csvFile, delimiter=',', skipinitialspace=True)
		
		# For each line in the file
		for line in reader:
			neutralDictionary[line[0]] = 1


##	+---------------------------------------------------------------+
##	| Performs a classification recognition using the given set and |
##    |  weights vector                                               |
##	+---------------------------------------------------------------+
def experiment(set, weights, bias):

	# Keep count of matches and mismatches
	matches = 0
	mismatches = 0
	truePositives = 0
	trueNegatives = 0
	predictedPositives = 0
	
	# For each entry t in the testing data set
	for xt, yt in set[0]:
		
		# Calculate the dot product
		y = predict_one(weights, xt, yt, bias)
		if y == 1:
			predictedPositives += 1
		
		# Update the respective counter
		if y == yt:
			matches += 1
			if y == 1:
				truePositives += 1
			else:
				trueNegatives += 1
		else:
			mismatches += 1
	
	# Calculate metrics
	accuracy = (float(truePositives) / float(set[2])) * (float(trueNegatives) / float((set[1] - set[2])))
	precision = float(truePositives) / float(predictedPositives)
	recall = float(truePositives) / float(set[2])
	average = (precision + recall) / 2.0
	fScore = (2.0 * precision * recall) / (precision + recall)
	
	# Print results
	print "MATCHES: ", matches
	print "MISMATCHES: ", mismatches
	print "TRUE POSITIVES: ", truePositives
	print "PREDICTED POSITIVES: ", predictedPositives
	print "ACTUAL POSITIVES: ", set[2]
	print "ACCURACY: ", float(accuracy)
	print "PRECISION: ", precision
	print "RECALL: ", recall
	print "AVERAGE: ", average
	print "F-SCORE: ", fScore
	print ""
	
	# Return values
	#return [matches, mismatches, accuracy, precision, recall, average, fScore]


##	+----------------------------+
##	| Gradient Descent Algorithm |
##	+----------------------------+
def GD(maxIterations, regularization, stepSize, lmbd, featureSet):
    
    # Get the number of words in the dictionary
    nWords = len(dictionary)
    
    # Initialize weight vectors and bias    
    #W = array([0] * nWords)
    #W = random.rand(nWords)
    W = numpy.random.uniform(-1, 1, nWords)
    b = 0.0
    
    # Initiate the feature vector
    Wt = numpy.array([0.0] * nWords)
    
    # The feature set
    D = featureSet[0]
    n = featureSet[1]
    
    # Initialize gradient array
    G = numpy.array([0.0] * nWords)
    
    # Indicate the type of regularization
    r = True if regularization == "l1" else False
    
    # Repeat the algorithm the indicated number of iterations
    for i in range(0, maxIterations):
        
        # Initialize gradient of weights and bias
        G *= 0.0
        g = 0.0
        
        # Initialize error counter
        errors = 0
        
        # For each entry t in the data set
        for x, y in D:
            
            # Set the respective values to the feature vector
            dot = 0.0
            Wt *= 0.0
            for word in x:
                dot += W[dictionary[word]]
            
            # If prediction is not correct
            if (y * (dot + b)) <= 1.0:
                
                # Increment error counter
                errors += 1
                
                # Update weight gradient
                for word in x:
                    loc = dictionary[word]
                    Wt[loc] = 1.0
                    G[loc] += y      
                
                # Update bias derivative
                g += y
        
        # Calculate the regularization value
        if r:
            reg = numpy.linalg.norm(Wt, 1) / 2.0
        else:
            reg = numpy.linalg.norm(Wt, 2) / 2.0
        
        # Add in regularization term
        G -= (lmbd * reg)
        
        # Update weights
        W += (stepSize * G)
        
        # Update bias
        b += (stepSize * g)
        
        # Calculate percentage error for the current iteration
        p = (errors / float(n)) * 100.0
        
        # Print information about iteration
        print "Training iter:", i, "\terror:", errors, "\tgradient:", g, "\tbias:", b, "\t", p, "%"
        
        # Break the training process when error is 0
        if errors == 0:
            break
        
    # Return the weight vector and the bias
    return W, b
    

##	+----------------------------------------------------------+
##	| Validating weight vector with Gradient Descent Algorithm |
##	+----------------------------------------------------------+
def VGD(W, maxIterations, regularization, stepSize, lmbd, featureSet):
    
    # Get the number of words in the dictionary
    nWords = len(dictionary)
    
    # Initialize weight vectors and bias    
    #W = array([0] * nWords)
    #W = random.rand(nWords)
    b = 0.0
    
    # Initiate the feature vector
    Wt = numpy.array([0.0] * nWords)
    
    # The feature set
    D = featureSet[0]
    n = featureSet[1]
    
    # Initialize gradient array
    G = numpy.array([0.0] * nWords)
    
    # Indicate the type of regularization
    r = True if regularization == "l1" else False
    
    # Repeat the algorithm the indicated number of iterations
    for i in range(0, maxIterations):
        
        # Initialize gradient of weights and bias
        G *= 0.0
        g = 0.0
        
        # Initialize error counter
        errors = 0
        
        # For each entry t in the data set
        for x, y in D:
            
            # Set the respective values to the feature vector
            dot = 0.0
            Wt *= 0.0
            for word in x:
                dot += W[dictionary[word]]
            
            # If prediction is not correct
            if (y * (dot + b)) <= 1.0:
                
                # Increment error counter
                errors += 1
                
                # Update weight gradient
                for word in x:
                    loc = dictionary[word]
                    Wt[loc] = 1.0
                    G[loc] += y      
                
                # Update bias derivative
                g += y
        
        # Calculate the regularization value
        if r:
            reg = numpy.linalg.norm(Wt, 1) / 2.0
        else:
            reg = numpy.linalg.norm(Wt, 2) / 2.0
        
        # Add in regularization term
        G -= (lmbd * reg)
        
        # Update weights
        W += (stepSize * G)
        
        # Update bias
        b += (stepSize * g)
        
        # Calculate percentage error for the current iteration
        p = (errors / float(n)) * 100.0
        
        # Print information about iteration
        print "Validating iter:", i, "\terror:", errors, "\tgradient:", g, "\tbias:", b, "\t", p, "%"
        
        # Break the training process when error is 0
        if errors == 0:
            break
        
    # Return the weight vector and the bias
    return W, b


##	+-------------------+
##	| The main function |
##	+-------------------+
def main():
    
    global trainingSet
    global validatingSet
    global testingSet
    
    # Validate the arguments
    arguments = validateInput(sys.argv)
    maxIterations, regularization, stepSize, lmbd, featureSet = arguments
    #print maxIterations, regularization, stepSize, lmbd, featureSet
    
    # Indicate which feature set is used
    feat = "UNIGRAMS" if featureSet == 1 else "BIGRAMS" if featureSet == 2 else "BOTH"
    
    # Build the neutral words dictionary
    buildNeutralDictionary(neutralWordsFile)
    
    # Build the sets
    trainingSet = buildSet(trainingDataFile, featureSet)
    validatingSet = buildSet(validatingDataFile, featureSet)
    testingSet =  buildSet(testingDataFile, featureSet)
    
    # Print parameters
    print "+---------------------+"
    print "| EXPERIMENT SETTINGS |"
    print "+---------------------+"
    print "maxIterations: ", maxIterations
    print "regularization: ", regularization
    print "stepSize: ", stepSize
    print "lmbd: ", lmbd
    print "featureSet: ", featureSet, " ", feat
    
    ## Run Gradient Descent
    print "+-------------------------------------+"
    print "| TRAINING USING SUB-GRADIENT DESCENT |"
    print "+-------------------------------------+"
    print ""
    weights, bias = GD(maxIterations, regularization, stepSize, lmbd, trainingSet)    
    
    # Show performance results for the training set
    print "+------------------------------+"
    print "| PERFORMANCE FOR TRAINING SET |"
    print "+------------------------------+"
    print ""
    experiment(trainingSet, weights, bias)
    
    # Show performance results for the validating set
    print "+--------------------------------+"
    print "| PERFORMANCE FOR VALIDATING SET |"
    print "+--------------------------------+"
    print ""
    experiment(validatingSet, weights, bias)
    
    # Show performance results for the testing set
    print "+-----------------------------+"
    print "| PERFORMANCE FOR TESTING SET |"
    print "+-----------------------------+"
    print ""
    experiment(testingSet, weights, bias)
    
    ## Run Gradient Descent with validation set
    print "+---------------------------------------+"
    print "| VALIDATING USING SUB-GRADIENT DESCENT |"
    print "+---------------------------------------+"
    print ""
    weights, bias = VGD(weights, maxIterations, regularization, stepSize, lmbd, validatingSet)
    
    # Show performance results for the training set
    print "+-----------------------------------------------+"
    print "| PERFORMANCE FOR TRAINING SET AFTER VALIDATION |"
    print "+-----------------------------------------------+"
    print ""
    experiment(trainingSet, weights, bias)
    
    # Show performance results for the validating set
    print "+-------------------------------------------------+"
    print "| PERFORMANCE FOR VALIDATING SET AFTER VALIDATION |"
    print "+-------------------------------------------------+"
    print ""
    experiment(validatingSet, weights, bias)
    
    # Show performance results for the testing set
    print "+----------------------------------------------+"
    print "| PERFORMANCE FOR TESTING SET AFTER VALIDATION |"
    print "+----------------------------------------------+"
    print ""
    experiment(testingSet, weights, bias)


### Initiate everything
if __name__ == '__main__':
    main()
