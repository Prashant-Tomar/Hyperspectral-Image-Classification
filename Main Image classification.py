
import math
import scipy.io
import numpy as np


###########################################  uploading file  ###################################################################

ground_truth = scipy.io.loadmat(r"Indian_pines_gt.mat")

data_of_indian_pines = scipy.io.loadmat(r"Indian_pines_corrected.mat")

###############################################################################################################################


num_of_epoch = 60000
learning = 0.5
DATA = data_of_indian_pines["indian_pines_corrected"]
Ground__truth = ground_truth["indian_pines_gt"]
lamda= 0.0001


###################################### changing shapes of data of dimension (145*145*200 to 21025*2)###############################
DATA = DATA.reshape(21025, 200)
DATA= DATA.astype(np.float64)

Ground__truth = Ground__truth.reshape(21025)




#####################################   erasing all the elements of class 0  ############################################################
index = np.nonzero(Ground__truth)
Ground__truth = Ground__truth[index]
DATA = DATA[index]
Ground__truth = np.subtract(Ground__truth, 1)


######################################### Converting into one hot encoded ##############################################################
one_hot_ground_truth = np.zeros((Ground__truth.shape[0],16))
for i in range(0, Ground__truth.shape[0]):
	one_hot_ground_truth[i][Ground__truth[i]] = 1

##########################   scaling of feature #####################################################################################
for f in range(0, DATA.shape[1]):
	mean = np.mean(DATA[:,f])
	minimum = np.min(DATA[:,f])
	maximum = np.max(DATA[:,f])
	DATA[:,f] = (DATA[:,f]-mean)/(maximum- minimum)


################################      Splitting datset in train and test data_of_indian_pines #########################################

ground_train = Ground__truth[0:int(Ground__truth.shape[0]*0.5)]

ground_test = Ground__truth[int(Ground__truth.shape[0]*0.5):Ground__truth.shape[0]]
train_data_of_indian_pines = DATA[0:int(DATA.shape[0]*0.5)]
test_data_of_indian_pines = DATA[int(DATA.shape[0]*0.5):DATA.shape[0]]
train_ground_truth = one_hot_ground_truth[0:int(Ground__truth.shape[0]*0.5)]
test_get_one_h = one_hot_ground_truth[int(Ground__truth.shape[0]*0.5):Ground__truth.shape[0]]

####################################### Initialising Weights ############################################################
WEIGHTS = np.random.randn(16,200)

############################ Performing Gradient Descent #############################################################################
for epoch in range(0, num_of_epoch):
	print("Epoch", epoch+1)
	M = np.matmul(train_data_of_indian_pines, np.transpose(WEIGHTS))
	exponential_of_M = np.exp(M)
	Loss = 0
	############################ calculating loss by summing over each class for each training example #####################################
	for i in range(0, exponential_of_M.shape[0]):
		sum = (np.sum(exponential_of_M[i,:]))
		p = ground_train[i]
		Loss= Loss+ np.log((exponential_of_M[i][p]/sum))

	Loss = (-Loss/train_data_of_indian_pines.shape[0]) + lamda*np.sum(np.sum(np.matmul(WEIGHTS, np.transpose(WEIGHTS))))
	print(" Loss in  Training", Loss)
	probability = np.zeros((train_data_of_indian_pines.shape[0],16))
	for c in range(0, train_data_of_indian_pines.shape[0]):
		sum = (np.sum(exponential_of_M[c,:]))
		for p in range(0, 16):
			probability[c][p] = exponential_of_M[c][p]/sum

	####################################   Testing Accuracy of model #######################################################################
	M_test = np.matmul(test_data_of_indian_pines, np.transpose(WEIGHTS))
	exponential_of_M_test = np.exp(M_test)
	probability_test = np.zeros((test_data_of_indian_pines.shape[0],16))
	for c in range(0, test_data_of_indian_pines.shape[0]):
		sum = (np.sum(exponential_of_M_test[c,:]))
		for p in range(0, 16):
			probability_test[c][p] = exponential_of_M_test[c][p]/sum
	predict = np.argmax(probability_test, axis =1)
	accuracy = float(np.sum(predict==ground_test)/ test_data_of_indian_pines.shape[0])
	counter=0
	for a in range(0, predict.shape[0]):
		if(predict[a]==ground_test[a]):
			counter = counter+1
	print("Accuracy", float(counter)*100/test_data_of_indian_pines.shape[0])
    
 


	#############################################  Updating the WEIGHTS ##############################################################
	gradient = np.zeros((16,200))
	gradient = np.matmul(np.transpose(train_data_of_indian_pines), train_ground_truth-probability)
	WEIGHTS = WEIGHTS- (learning*np.transpose(gradient)*(-1)/train_data_of_indian_pines.shape[0]) + lamda*WEIGHTS

	print("\n"*2)



#########################################################################################################################################
    