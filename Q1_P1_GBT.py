



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



from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from  pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


higgsRaw = spark.read.csv('HIGGS.csv').toDF("label", "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb")






#print(type(higgsRaw))

higgsRaw.show(2)
higgsRaw.printSchema()

dataNames = higgsRaw.schema.names
ncolumns = len(higgsRaw.columns)

for i in range(ncolumns):
    higgsRaw = higgsRaw.withColumn(dataNames[i], higgsRaw[dataNames[i]].cast(DoubleType()))



(print("\n\n \n now double:\n\n\n"))

higgsRaw.show(2)

higgsRaw.printSchema()

namesH = ["lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

assembler = VectorAssembler(inputCols = namesH, outputCol = "features")



higgs = assembler.transform(higgsRaw.dropna())
higgsData = higgs.select('features'  , 'label')


print("vectorised:  ")

higgsData.show(10)
higgsData.printSchema()



smallData, bigData = higgsData.randomSplit([0.05 , 0.95], 42)




(trainingData, testData) = smallData.randomSplit([0.7, 0.3], 42)



gbt = GBTClassifier(labelCol="label", featuresCol="features" )


clasEv= BinaryClassificationEvaluator()

mClasEv =  MulticlassClassificationEvaluator(metricName = "accuracy")

paramGrid = ParamGridBuilder() \
.addGrid(gbt.maxDepth, [3, 5, 7])\
.addGrid(gbt.maxBins, [16, 20, 24])\
.addGrid(gbt.maxIter , [5, 10, 15])\
.build()



crossVal = CrossValidator()\
.setEstimator(gbt)\
.setEvaluator(clasEv)\
.setEstimatorParamMaps(paramGrid)\
.setNumFolds(3) \
.setParallelism(9)  


cvModel = crossVal.fit(trainingData)
bestModel = cvModel.bestModel

bestMaxDepth = bestModel._java_obj.getMaxDepth()
bestMaxIter = bestModel._java_obj.getMaxIter()
bestMaxBins = bestModel._java_obj.getMaxBins()

print("max depth: " ,bestMaxDepth)
print("max iterations ", bestMaxIter)

print("max bins " ,bestMaxBins)





AUC =   clasEv.evaluate(cvModel.transform(testData))


print("AUC: ", AUC)





print("done")

spark.stop()


