package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** INSERT PATHS HERE **/
    val input = "/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/INF729/Spark/data_cleaned_corr"
    val output = "/Users/thaianthantrong/Documents/MS_BIG_DATA/Cours/INF729/Spark/trainer_results"


   /** CHARGER LE DATASET **/
   /* Load data into a DataFrame */
   /* Replace any false value with null */
   var df = spark.read.parquet(input)

    /** TF-IDF **/
    // Split text into words
    println(" ----- 1. TOKENIZER -----")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stop with StopWordsRemover
    println(" ----- 2. STOP WORDS REMOVER -----")
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // TF : CountVectorizer
    println(" ----- 3. TF : COUNT VECTORIZER -----")
    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("td")
      .setVocabSize(3)
      //.setMinDF(2)

    // IDF
    println(" ----- 4. IDF -----")
    val idf = new IDF()
      .setInputCol("td")
      .setOutputCol("tfidf")

    /** VECTOR ASSEMBLER **/

    // COUNTRY
    println(" ----- 5. CATEGORICAL COLUMN COUNTRY -----")
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    // CURRENCY
    println(" ----- 6. CATEGORICAL COLUMN CURRENCY -----")
    val indexer_currency= new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    println(" ----- 7. VECTOR ASSEMBLER -----")
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /** MODEL **/
    println(" ----- 8. MODEL -----")
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    println(" ----- PIPELINE -----")
    // Configure an ML pipeline, which consists of the eight stages
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexer_country, indexer_currency, assembler, lr))

    /** TRAINING AND GRID-SEARCH **/
    println(" ----- SPLIT TRAIN / TEST -----")
    // Prepare training and test data.
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)

    // Run train validation split, and choose the best set of parameters.
    println(" ----- RUN TRAIN VALIDATION SPLIT -----")
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    println(" ----- TEST DATA -----")
    val df_WithPredictions = model.transform(test)
      .select("features", "final_status", "predictions", "raw_predictions")

    println(df_WithPredictions.show())

    println(" ----- F-MEASURE -----")
    val accuracy = evaluator.evaluate(df_WithPredictions)
    println("Logistic Regression Classifier Accuracy (F1-Measure): " + accuracy)

    // PRINT PREDICTIONS
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // SAVE MODEL
    model.write.overwrite().save(output)

    println("########## END PROCESS ##########")
  }
}
