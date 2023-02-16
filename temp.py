# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
class ADALINE():
    #construacor
    def __init__(self,d,xi,n_samples,wi,fac_ap,epochs,precision,w_tight):
        #self is  used to refer to a variable field within the class
        self.d=d
        self.xi=xi
        self.n_samples=n_samples
        self.wi=wi
        self.fac_ap=fac_ap
        self.epochs=epochs
        self.precision=precision
        self.y=0 #network exit
        self.w_tight=w_tight
    def traning(self):
        E=1 #exit error
        E_ac=0 #current error
        Error_prev=0 #previous mistake
        Ew=0 #mean square error
        E_red=[] #network error
        E_total=0# error total
        while (np.abs(E)>self.precision):
            Error_prev=Ew
            for i in range(self.n_samples):
                self.y=sum(self.xi[i,:]*self.wi) #calculation of the network output
                E_ac=(self.d[i]-self.y)#error calculation
                self.wi=self.wi+(self.fac_ap*E_ac*self.xi[i,:])
                E_total=E_total+((E_ac)**2)

            #calculate mean square error
            Ew=((1/self.n_samples)*(E_total))
            E=(Ew-Error_prev)#network error
            E_red.append(np.abs(E))
            self.epochs+=1
        return self.wi,self.epochs,E_red

    def F_operation(self):
        output=[]
        for j in range(self.n_samples):
            self.y=sum(self.xi[j,:]*self.w_tight)
            output.append(self.y)
        return output


#main circle
if __name__=="__main__":
#read excel table
    rand_arr=[]
    counter=0
    count=0
    with open('weatherHistory.csv','r') as csv_file:
        arr=csv_reader=csv.DictReader(csv_file,delimiter=',')
        for lines in csv_reader:
            counter=counter+1
            arr=lines['Apparent Temperature (C)']
        for i in range(int((counter*70)/100)):
            rand_arr=(random.choice(arr))
            count=count+1
    table=pd.ExcelFile("weatherHistory.csv")
    v_data=table.parse(rand_arr)
    #convert table data to array
    v_data=np.array(v_data)
    #input data
    xi=v_data[count]
    #desered values
    d=v_data[count]
    #number of samples
    n_samples=len(count)
    #set the weight vector w
    wi=np.array[count]
    #learning factor
    fac_ap=0.3
    epochs=0
    precision=0.0001
    w_tight=[]
    #initialize adaline network
    red=ADALINE(d,xi,n_samples,wi,fac_ap,epochs,precision,w_tight)
    w_tight,epochs,error=red.traning()
    #graph
    plt.ylabel('Error',Fontsize=12)
    plt.xlabel('epochs',Fontsize=12)
    plt.title("ADALINE ,rule")
    x=np.arange(epochs)
    plt.plot(x,error,'m->',label="Error")
    plt.legend(loc='upper right')
    plt.show()
    print("adjusted weights",w_tight)