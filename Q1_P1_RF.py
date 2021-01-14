





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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from  pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


higgsRaw = spark.read.csv('HIGGS.csv').toDF("label", "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb")





#higgsRaw.show(0)


#print(type(higgsRaw))

higgsRaw.show(2)


dataNames = higgsRaw.schema.names
ncolumns = len(higgsRaw.columns)

for i in range(ncolumns):
    higgsRaw = higgsRaw.withColumn(dataNames[i], higgsRaw[dataNames[i]].cast(DoubleType()))



(print("now double:"))

higgsRaw.show(2)

higgsRaw.printSchema()



####

higgsRaw.select("label").show()

from pyspark.sql.functions import isnull, col 
from pyspark.sql.functions import when


zeros= higgsRaw.filter(col("label")==0).count()
ones = higgsRaw.filter(col("label")==1).count()
imbalance = ones/zeros

print("imbalance: ", imbalance)
print(ones, "ones, and ", zeros, "zeros")

higgsRaw.select("label").show(100)







namesH = ["lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

assembler = VectorAssembler(inputCols = namesH, outputCol = "features")



higgs = assembler.transform(higgsRaw.dropna())
higgsData = higgs.select('features'  , 'label')


print("vectorised:  ")

higgsData.show(10)
higgsData.printSchema()



smallData, bigData = higgsData.randomSplit([0.05 , 0.95], 42)




(trainingData, testData) = smallData.randomSplit([0.7, 0.3], 42)



rfc = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=3)


clasEv= BinaryClassificationEvaluator()

mClasEv =  MulticlassClassificationEvaluator(metricName = "accuracy")

paramGrid = ParamGridBuilder() \
.addGrid(rfc.maxDepth, [3, 5, 7])\
.addGrid(rfc.maxBins, [16, 20, 24])\
.addGrid(rfc.featureSubsetStrategy, ["all", "onethird", "log2"])\
.build()



crossVal = CrossValidator()\
.setEstimator(rfc)\
.setEvaluator(clasEv)\
.setEstimatorParamMaps(paramGrid)\
.setNumFolds(3) \
.setParallelism(9)  
#Increased parallelism, because we have lots of cores avaiable, and will speed up process
#by testing different configurations at the same time

cvModel = crossVal.fit(trainingData)
bestModel = cvModel.bestModel

bestMaxDepth = bestModel._java_obj.getMaxDepth()
bestFSS = bestModel._java_obj.getFeatureSubsetStrategy()
bestMaxBins = bestModel._java_obj.getMaxBins()

print("max depth: " ,bestMaxDepth)
print("FSS: ", bestFSS)
print("max bins: " ,bestMaxBins)





AUC =   clasEv.evaluate(cvModel.transform(testData))


print("AUC: ", AUC)





print("done")


spark.stop()

