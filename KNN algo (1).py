from mymodules import*

def CalculateDistance(x,y):
    return distance.euclidean(x,y)

class marvellous():
    def fit(self,TrainingData,TrainingTarget):
        print("data training done...")
        self.TrainingData=TrainingData
        self.TrainingTarget=TrainingTarget
        
    def predict(self,TestData):
        prediction=[]
        for row in TestData:
            label=self.Shortest(row)
            prediction.append(label)
        return prediction
        
    def Shortest(self,row):
        minindex=0
        mindistance=CalculateDistance(row,self.TrainingData[0])
        
        for i in range(len(self.TrainingData)):
            Distance=CalculateDistance(row,self.TrainingData[i])
            if Distance<mindistance:
                mindistance=Distance
                minindex=i
                
        return self.TrainingTarget[minindex]
            
def marvellousKNN():
    Line="*"*50
    
    iris=load_iris()
    
    data=iris.data
    target=iris.target
  
    print(Line)
    print("actual dataset")
    print(Line)
    
    for i in range(len(iris.target)):
        print("ID:%d feature: %s , label:%s "%(i,iris.data[i],iris.target[i]))
        
    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)
   
    print(Line)
    print("training dataset...")
    print(Line)
    
    for i in range(len(data_train)):
        print("ID:%d feature: %s , label:%s "%(i,data_train[i],target_train[i]))
   
    print(Line)
    print("testing dataset...")
    print(Line)
    for i in range(len(data_test)):
        print("ID:%d feature: %s , label:%s "%(i,data_test[i],target_test[i]))
    
    print(Line)
    mobj=marvellous()
    
    mobj.fit(data_train,target_train)
    
    ret=mobj.predict(data_test)
    
    print("Result of Machine Learning Model")
    print(Line)
    for i in range(len(data_test)):
        print("ID : %d Expectation : %s, Prediction : %s" %(i, target_test[i],ret[i]))
    print(Line)

    icnt = 0
    for i in range(len(data_test)):
        if target_test[i] != ret[i]:
            icnt = icnt + 1
    print("Number of wrong answers by the ML model : ",icnt)
    print(Line)
    
    
    Accuracy=accuracy_score(target_test,ret)
    return Accuracy
    
    
def main():
    ret=marvellousKNN()
    print("Accuracy of KNN is: ",ret*100,"%")
    
   
   
if __name__=="__main__":
    main()

