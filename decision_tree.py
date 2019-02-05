from util import entropy, information_gain, partition_classes
import ast
from operator import itemgetter
import csv
import random 
class DecisionTree(object):
    
    global max_depth
    max_depth = 250 #set the maximum depth in here
    
    
    #print ("max_depth is set to", max_depth)
    depth_counter = 0 #initialize depth_counter 
    
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        #print ("initializing dictionary")
        self.depth = DecisionTree.depth_counter
        DecisionTree.depth_counter += 1
    

    def find_best_split(self,X_data, y_data):
    #def partition_classes(X, y, split_attribute, split_val):
    #return (left_list,right_list,y_left,y_right)
        
        best_info_gain = -999 # initialize information gain
        best_split_attribute = 999
        best_split_val = 999
        
        # ------------------- important -------------------------
        # ------------------- important -------------------------
        # random forest pick random attributes
        
        attributes_amount = 8
        
        random_attributes_index = random.sample(range(0,len(X_data[0]),1),attributes_amount)
        
        # ------------------- important -------------------------
        # ------------------- important -------------------------
        
        for column_index in random_attributes_index:
            #column val contains all the row in a column
            
            column_val = list(map(itemgetter(column_index),X_data))
            
            for row_index in range(0, len(column_val),1):
                (left_list,right_list,y_left,y_right) = partition_classes(X_data,y_data,column_index,column_val[row_index])
                info_gain = information_gain(y_data, [y_left,y_right])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split_attribute = column_index
                    best_split_val = column_val[row_index]
                        
        return (best_split_attribute, best_split_val, best_info_gain)


    def is_leaf_node(self,X_data, y_data,min_size):
        
        # this function determines if a leaf node is reach
        # input:
        #X_data, a list, contains all the attributes except the classfication
        #y_data, a list, contains all the classification (the last column of the csv)
        unique_y_values = set(y_data)  # find out the unique value in the y_data
        
        first_row_X = X_data[0][:]    #store the first row of the data, [:] means copy to avoid aliasing
        identical_attributes = True   #initialze if all X data is the same
        y_majority = max(set(y_data), key = y_data.count) # return majority of y
        
        
        # determine a minimal data in a tree node
        if len(y_data) <= min_size:
            if y_data.count(0) == y_data.count(1):  #if there is a tie in the y value,
                y_majority = random.randint(0,1)  # take random value
            return (True, y_majority, "#data pts less than threshold") # the last message is for debuging
        
        # if all the datapoints in X have the same class value y, return a leaft node that predicts y as output
        if len(unique_y_values) == 1:
            return (True, list(unique_y_values)[0], "all data in branch has same y value")
        else: # check for if all rows in X_data is the same
            for row in X_data[1:]:
                if row != first_row_X:
                    identical_attributes = False
                    return (False, "NA" ,"NA")
            
            if identical_attributes == True:
                if y_data.count(0) == y.data.count(1): #handle ties in y
                    y_majority = random.randint(0,1)
                return (True, y_majority, "ALL X are the same")   
#------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------             
    def learn(self, X, y):
    # TODO: Train the decision tree (self.tree) using the the sample X and labels y
    # You will have to make use of the functions in utils.py to train the tree
    
    # One possible way of implementing the tree:
    #    Each node in self.tree could be in the form of a dictionary:
    #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
    #    For example, a non-leaf node with two children can have a 'left' key and  a 
    #    'right' key. You can add more keys which might help in classification
    #    (eg. split attribute and split value)
    
    
    # From the CMU slide
    # We want know as pure as possible, meaning that we want to reduce the entropy
    # and we want to maximize the difference betwee nthe entropy of the parent node
    #and the expected entropy of the children
    
    # X is the csv files in list
    # y is the resonse variable in list 
    #print ("at the learn method")
        
        
        (leaf_node_boolean, leaf_node_yval, message) = self.is_leaf_node(X, y, min_size = 1)
            
            
        if leaf_node_boolean == True:
            #print (message)
            self.tree["leaf_node"] = True  
            self.tree["y_value"] = leaf_node_yval
            return ()
         
           
    # find the best split condition
        (best_split_attribute, best_split_val,best_info_gain) = self.find_best_split(X,y)
        
    # Check if there is really information again after spliting, base case #0
    # when there is no info gain, most likely one of the splitted side has length of zero
    # of info_gain greater than 0.0, perform split
        if best_info_gain > 0.001: # this value can be fine tuned
            
            self.tree["leaf_node"] = False  # set leaf_node to false since there is info_gain, so continue to split
            
            # check for the tree for depth
            if self.depth >= max_depth:
                self.tree["leaf_node"] = True
                if y.count(1) == y.count(0): # handles tie in y
                    self.tree["y_value"] =  random.randint(0,1)
                else:
                    self.tree["y_value"] = max(set(y), key = y.count)
                return ()
            
            
            (left_list,right_list,y_left,y_right)= partition_classes(X, y, best_split_attribute, best_split_val)
            
            left_branch = DecisionTree()  # create new Decisiontree class
            right_branch = DecisionTree()        
            left_branch.learn(left_list,y_left)    # recursion to learn
            right_branch.learn(right_list,y_right)
    
            
            self.tree["best_split_attribute"] = best_split_attribute
            self.tree["best_split_val"] = best_split_val
            self.tree["left_branch"] = left_branch
            self.tree["right_branch"] = right_branch
            
        else: 
            self.tree["leaf_node"] = True
            if y.count(1) == y.count(0):
                self.tree["y_value"] =  random.randint(0,1) # handles ties in y
            else:
                self.tree["y_value"] =  max(set(y), key = y.count)
            return ()



    
    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        # a record means a row a X data
        
        
        if self.tree["leaf_node"] == True:
            return self.tree["y_value"]
        
        if self.tree["leaf_node"] == False:
            best_split_attribute = self.tree["best_split_attribute"]
            
            if record[best_split_attribute] <= self.tree["best_split_val"]:
                return self.tree["left_branch"].classify(record)
        
            else:
                return self.tree["right_branch"].classify(record)
    
    

        
        # if all the datapoints in X have the same attributes value, return a leaf node that predicts the majority 
        # of the class values in Y as output
        

#def read_csv_file (filename):
#    row_counter = 0
#    f = open(filename, "r")
#    X = []
#    y = []
#    reader = csv.reader (f, delimiter = ',')
#    next(reader,None) # ignore header
#    for row in reader:
#        X.append(row[0:(len(row)-1)])
#        y.append(int((row[-1])))
#    f.close()
#    return (X,y)
#
#(X,y) = read_csv_file('hw4-data.csv')
#
#
#
#A = DecisionTree()    
#A.learn(X,y)
#print ("done with learning tree")
##row = X = [[1,1,1], [2,1,1],[10,2,5],[20,23,1],[30,23,1],[32,0,32]]
#y_predicted = []
#for i in X:
#    #print (i)
#    y_predicted.append (A.classify(i))
#    
#    
#    
#def calculate_accuracy(y_predicted, y):
#    assert len(y_predicted) == len(y)
#    correct_prediction = 0
#    for i in range(0,len(y),1):
#        if y_predicted[i] == y[i]:
#            correct_prediction += 1
##    print (len(y))
##    print (correct_prediction)
#    return (float(correct_prediction)/len(y))
#
#
#print (calculate_accuracy(y_predicted, y)) 