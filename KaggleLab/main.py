import pyspark.sql.session
from pyspark.ml.feature import StringIndexer, IndexToString, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import *

spark = pyspark.sql.session.SparkSession.builder \
    .master("local") \
    .appName("KaggleLab") \
    .getOrCreate()

trainingData = spark.read.csv(
    "/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/SD701/KaggleLab/train-data.csv", header=True, inferSchema=True)

testData = spark.read.csv(
    "/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/SD701/KaggleLab/test-data.csv", header=True, inferSchema=True)

# stages in our Pipeline
stages = []

# Index labels, adding metadata to the label column.
# Fit on whole trainingData to include all labels in index.
labelIndexer = StringIndexer(inputCol="Cover_Type", outputCol="label").fit(trainingData)
stages += [labelIndexer]

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="Cover_Type_pred", labels=labelIndexer.labels)

# All columns
all_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
            "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6",
            "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13",
            "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20",
            "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27",
            "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
            "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]

# One-Hot Encoding
# Try to drop first : Soil_Type1
categoricalColumns = ["Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6",
                      "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12",
                      "Soil_Type13",
                      "Soil_Type14", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
                      "Soil_Type20",
                      "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26",
                      "Soil_Type27",
                      "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33",
                      "Soil_Type34",
                      "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"]
# Soil_Type7, Soil_Type15 : always the same value

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# Numerical columns : create vecAssembler
numericCols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                 "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
                 "Wilderness_Area4"]
# Transform all features into a vector using VectorAssembler
assemblerInputs = list(map(lambda c: c + "classVec", categoricalColumns)) + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [vecAssembler]

# Split existing trainingData into training and test sets (30% held out for testing)
(training, test) = trainingData.randomSplit([0.7, 0.3], seed=100)

# Train a RandomForest model.
# dt = RandomForestClassifier(labelCol="label", featuresCol="features", impurity='gini', seed=100, numTrees=100,
#                             maxDepth=30, maxBins=100)
dt = RandomForestClassifier(labelCol="label", featuresCol="features", impurity='gini', seed=100)

# Add stages
stages += [dt, labelConverter]

# Chain indexers and forest in a Pipeline
# pipeline = Pipeline(stages=[labelIndexer, vecAssembler, dt, labelConverter])
pipeline = Pipeline(stages=stages)

# Run stages in pipeline and train model
model = pipeline.fit(training)

# Make predictions on test so we can measure the accuracy of our model on new data
predictions_tr = model.transform(test)

# Display what results we can view
# predictions_tr.printSchema()

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")

accuracy = evaluator.evaluate(predictions_tr)
print("Decision Tree Classifier Accuracy before Cross Validation : ", accuracy)

# We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
# This will allow us to jointly choose parameters for all Pipeline stages.
# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# We use a ParamGridBuilder to construct a grid of parameters to search over.
print("Starting CrossValidation")
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [20, 25, 28])
             .addGrid(dt.maxBins, [65, 70, 75])
             .addGrid(dt.numTrees, [80, 100])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross-validation, and choose the best set of parameters.
cvModel = cv.fit(trainingData)

print("      | numTrees = ", cvModel.bestModel.numTrees)
print("      | depth = ", cvModel.bestModel.maxDepth)

# Make predictions on test so we can measure the accuracy of our model on new data
predictions_tr_cv = cvModel.transform(test)

accuracy_cv = evaluator.evaluate(predictions_tr_cv)
print("Decision Tree Classifier Accuracy after Cross Validation : ", accuracy_cv)

predictions = cvModel.transform(testData)

# Display what results we can view
# predictions.printSchema()

print(predictions.repartition(1).select('Id', col('Cover_Type_pred').alias('Cover_Type').cast('integer')))

# Select columns Id and prediction
(predictions
 .repartition(1)
 .select('Id', col('Cover_Type_pred').alias('Cover_Type').cast('integer'))
 .write
 .format('com.databricks.spark.csv')
 .options(header='true')
 .mode('overwrite')
 .save('/Users/thaianthantrong/submission'))
