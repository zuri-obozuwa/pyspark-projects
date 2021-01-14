
from pyspark.sql import SparkSession




spark = SparkSession.builder \
    .master("local[10]") \
    .config("spark.local.dir","/fastdata/acq19zfo") \
    .appName("Assgn2") \
    .getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")


import numpy as np
import matplotlib.pyplot as plt


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


higgsRaw = spark.read.csv('HIGGS.csv').toDF("label", "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb")




#higgsRaw.show(0)


#print("WOOOaaah \n \n")
#print(type(higgsRaw))

higgsRaw.show(2)
higgsRaw.printSchema()

dataNames = higgsRaw.schema.names
ncolumns = len(higgsRaw.columns)

for i in range(ncolumns):
    higgsRaw = higgsRaw.withColumn(dataNames[i], higgsRaw[dataNames[i]].cast(DoubleType()))



(print("now double:"))

higgsRaw.show(2)

higgsRaw.printSchema()

namesH = ["lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

assembler = VectorAssembler(inputCols = namesH, outputCol = "features")



higgs = assembler.transform(higgsRaw.dropna())
higgsData = higgs.select('features'  , 'label')


print("vectorised:  ")

higgsData.show(10)
higgsData.printSchema()






(trainingData, testData) = higgsData.randomSplit([0.7, 0.3], 42)



rfc = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=3, maxBins = 20, maxDepth = 7, featureSubsetStrategy = "all" )

clasEv= BinaryClassificationEvaluator()

mClasEv =  MulticlassClassificationEvaluator(metricName = "accuracy")


rfcModel = rfc.fit(trainingData)


pred = rfcModel.transform(testData)

AUC =   clasEv.evaluate(pred)
accuracy = mClasEv.evaluate(pred)

print("AUC: ", AUC)

print("Accuracy: ", accuracy)

##Relevant features:


featI = rfcModel.featureImportances

important_feat = np.zeros(len(namesH))
important_feat[featI.indices] = featI.values

for x in range(len(namesH)):
    print("feature ", namesH[x], "has a relative importance of ", important_feat[x])


print("done")


spark.stop()