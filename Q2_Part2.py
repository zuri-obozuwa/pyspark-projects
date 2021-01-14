
from pyspark.sql import SparkSession




spark = SparkSession.builder \
    .master("local[10]") \
    .config("spark.local.dir","/fastdata/acq19zfo") \
    .appName("Assgn2 Q2 Part 2") \
    .getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")


import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql.functions import isnull, col 
from pyspark.sql.functions import when
from pyspark.sql.types import DoubleType


insuranceRaw = spark.read.csv('train_set.csv', header=True)

#Question 2, Part 1 and 2

#Part 1:

###MAKING A BINARY CLAIM COLUMN
insuranceRaw = insuranceRaw.withColumn('binaryclaim', when (col("Claim_Amount")>0, 1).otherwise(0))


insuranceRaw = insuranceRaw.withColumn('binaryclaim', insuranceRaw['binaryclaim'].cast(DoubleType()))
insuranceRaw.printSchema



####
unwanted = ['Household_ID', 'Vehicle', 'Calendar_Year','Model_Year', 'Blind_Make' , 'Blind_Model', 'Blind_Submodel' ]
for c in unwanted:
    insuranceRaw = insuranceRaw.drop(c)

####



insuranceRaw.show(3)
insuranceRaw.printSchema()



cont = ['Var1','Var2', 'Var3','Var4','Var5', 'Var6','Var7','Var8', 'NVVar1','NVVar2', 'NVVar3', 'NVVar4' ]


print("contin")
insuranceRaw.select(cont).show(7)


categ = [ 'Cat1', 'Cat2' , 'Cat3' , 'Cat4', 'Cat5' , 'Cat6', 'Cat7' ,'Cat8', 'Cat9' , 'Cat10', 'Cat11' , 'Cat12', 'OrdCat', 'NVCat' ]

nonOrdCateg = [ 'Cat1', 'Cat2' , 'Cat3' , 'Cat4', 'Cat5' , 'Cat6', 'Cat7' ,'Cat8', 'Cat9' , 'Cat10', 'Cat11' , 'Cat12' , 'NVCat' ]


##Removal of categorical columns that have more than 30% missing values
rowcount = insuranceRaw.count()

categFilt = []
categDrop = []

for x in categ:
    nullcount = insuranceRaw.where(isnull(col(x))).count()
    qcount= insuranceRaw.select(x).filter((insuranceRaw[x]== "?"  )).count()
    proportion = ((nullcount + qcount)/rowcount)*100

    print("percentage of missing values in ", x, " is ", proportion, "%")
    if proportion < 30:
        categFilt.append(x)
    else:
        categDrop.append(x)


for c in categDrop:
    insuranceRaw = insuranceRaw.drop(c)


print(categFilt)

insuranceRaw.printSchema()


#imputation of most common value in a column to replace missing values in that column
#if null or '?', replace with most common value


from pyspark.sql.functions import when


for v in categFilt:

    counts = insuranceRaw.select(v).groupBy(v).count().orderBy("count")
    counts.show()
    mostCommon = str(counts.select(v).collect()[-1][v])
    print("most common value in ", v, "is ", mostCommon)
    
    insuranceRaw = insuranceRaw.withColumn(v, when(insuranceRaw[v]=="?",mostCommon).otherwise(insuranceRaw[v]))
    insuranceRaw = insuranceRaw.fillna(mostCommon, v)


    counts = insuranceRaw.select(v).groupBy(v).count().orderBy("count")
    counts.show()
    



insuranceRaw.printSchema()





#string indexing

from pyspark.ml.feature import StringIndexer

#we dont want to index the ordcat
categFilt.remove("OrdCat")

for t in categFilt:
    stringind =  StringIndexer(handleInvalid="keep", inputCol=t, outputCol = t + "_ind")
    insuranceRaw = stringind.fit(insuranceRaw).transform(insuranceRaw)
    

for c in categFilt:
    insuranceRaw = insuranceRaw.drop(c)


insuranceRaw.printSchema()

insuranceRaw.show(80)
###

####


from pyspark.ml.feature import OneHotEncoderEstimator



stringIndNames = []
encodedNames = []

for v in categFilt:

    encodedNames.append(v+"_indE")
    stringIndNames.append(v+"_ind")

print(categFilt)
print(stringIndNames)
print("\n", encodedNames)

encoder = OneHotEncoderEstimator(inputCols= stringIndNames, outputCols=encodedNames)

insuranceRaw = encoder.fit(insuranceRaw).transform(insuranceRaw)


insuranceRaw.select('Cat1_indE', 'Cat3_indE').show(30)


#vector assemble the categorical features
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=encodedNames, outputCol="catfeatures")

insuranceRaw = assembler.transform(insuranceRaw)

for x in stringIndNames:
    insuranceRaw = insuranceRaw.drop(x)



insuranceRaw.printSchema

insuranceRaw.show(100)




###NOW DEALING WITH CONTINOUS DATA


from pyspark.sql.functions import isnull, col 
from pyspark.sql.types import DoubleType

###

cont = ['Var1','Var2', 'Var3','Var4','Var5', 'Var6','Var7','Var8', 'NVVar1','NVVar2', 'NVVar3', 'NVVar4' ]

toDouble = ['Claim_Amount' , 'OrdCat', 'Var1','Var2', 'Var3','Var4','Var5', 'Var6','Var7','Var8', 'NVVar1','NVVar2', 'NVVar3', 'NVVar4' ]



###
#Change continous values to type double
#ALSO CHANGES ORDCAT TO DOUBLE and CLASS_AMOUNT to double
for g in range(len(toDouble)):
    insuranceRaw = insuranceRaw.withColumn(toDouble[g], insuranceRaw[toDouble[g]].cast(DoubleType()))


###



contFilt = []
contDrop = []


for x in cont:
    nullcount = insuranceRaw.where(isnull(col(x))).count()
    qcount= insuranceRaw.select(x).filter((insuranceRaw[x]== "?"  )).count()
    nancount= insuranceRaw.select(x).filter((insuranceRaw[x]== float('nan')  )).count()
    print("for ", x, ": " ,nullcount, "are null, ",qcount , "are ?, ", nancount, " are nan " )


    proportion = ((nancount + nullcount + qcount)/rowcount)*100

    print("percentage of missing values in ", x, " is ", proportion, "%")
    if proportion < 30:
        contFilt.append(x)
    else:
        contDrop.append(x)

print("We have no nan or ?; only null")



from pyspark.ml.feature import Imputer

contImp =  ['Var1_imp','Var2_imp', 'Var3_imp','Var4_imp','Var5_imp', 'Var6_imp','Var7_imp','Var8_imp', 'NVVar1_imp','NVVar2_imp', 'NVVar3_imp', 'NVVar4_imp' ]
imputer = Imputer(strategy = 'mean', inputCols = cont, outputCols=contImp)

insurance_imp = imputer.fit(insuranceRaw).transform(insuranceRaw)

print("Impution COMPLETE \n\n\n")


for c in cont:
    insurance_imp = insurance_imp.drop(c)




insurance_imp.printSchema()

insurance_imp.show(20)



print("\n\n drop cat features (leave combined vector from encoder):\n\n")

for c in encodedNames:
    insurance_imp = insurance_imp.drop(c)

insurance_imp.printSchema()

insurance_imp.show(20)


####REMOVES NULL CLAIM COLUMNS, 
#also removes null values from failed conversion of strings (mistakes) to floats

insurance_imp = insurance_imp.dropna('any')


####Vector assembler again to join the continous and cat values:

from pyspark.ml.feature import VectorAssembler

contImpNames = contImp 

allFeatures = contImp

allFeatures.append("catfeatures")
allFeatures.append("OrdCat")

assembler = VectorAssembler(inputCols= allFeatures, outputCol="finalvector")


insurance = assembler.transform(insurance_imp)

insurance.show(20)

for c in contImpNames:
    insurance = insurance.drop(c)

insurance.show(100)








###BALANCING THE DATA

from pyspark.sql.functions import rand
    

zeroclaims = insurance.filter(col("binaryclaim")==0)
claims = insurance.filter(col("binaryclaim")==1)
imbalance = int(zeroclaims.count()/claims.count())

zeroBalanced = zeroclaims.sample(False, 1/imbalance, 42)
insuranceBalanced = zeroBalanced.unionAll(claims)

insuranceBalanced = insuranceBalanced.orderBy(rand(seed=42))


###Linear regression Question 2 part 2

(trainingData, testData) = insuranceBalanced.randomSplit([0.7, 0.3], 42)


#Checking its same split of data as part 3
trainingData.select("Row_ID", "binaryclaim").show(150)


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='finalvector', labelCol='Claim_Amount', regParam=0.1, elasticNetParam = 0.5)
lrmodel = lr.fit(trainingData)
predictions = lrmodel.transform(testData)



from pyspark.ml.evaluation import RegressionEvaluator
MSEevaluator = RegressionEvaluator(labelCol='Claim_Amount', predictionCol="prediction", metricName="mse")
MAEevaluator = RegressionEvaluator(labelCol='Claim_Amount', predictionCol="prediction", metricName="mae")


mse = MSEevaluator.evaluate(predictions)
mae = MAEevaluator.evaluate(predictions)
print("MSE = " , mse)

print("MAE = " , mae)



print("done")
spark.stop()