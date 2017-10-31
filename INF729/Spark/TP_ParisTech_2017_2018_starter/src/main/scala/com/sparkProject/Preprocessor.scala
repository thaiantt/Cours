package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.to_timestamp
import org.apache.spark.sql.functions.to_date
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.regexp_replace

object Preprocessor {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    /* Entry point for the whole Spark's SQL brick*/
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.sqlContext.implicits._

    /** *****************************************************************************
      *
      * TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("Hello World ! Let's start our preprocessor")

    /** 1 - CHARGEMENT DES DONNEES **/

    println("++++++++++ QUESTION 1.A : LOAD DATA ++++++++++")
    /* Load data into a DataFrame */
    /* Replace any false value with null */
    var df = spark.read.format("org.apache.spark.csv")
      .option("header", true)
      .option("inferSchema", true)
      .option("nullValue", "false")
      .csv("train.csv")

    // Deal with quotes characters in string data
    val df0 = df.withColumn("desc", regexp_replace(df("desc"),"\"{2,}", " "))
      .withColumn("name", regexp_replace(df("name"),"\"{2,}", " "))


    /* OLD : Replace any false value with null */
    // Define udf:
    /*def udf_replace_false = udf { (value: String) =>
      if (value == "false")
        null
      else
        value
    }

    df = df.withColumn("goal", udf_replace_false($"goal")).withColumn("country", udf_replace_false($"country"))*/

    println("++++++++++ QUESTION 1.B : DATAFRAME DIMENSIONS ++++++++++")
    /* Number of lines */
    val count_rows = df.count()
    println("Number of rows : " + count_rows)

    /* Number of columns */
    val count_cols = df.columns.size
    println("Number of columns : " + count_cols)

    println("++++++++++ QUESTION 1.C : SEE DATAFRAME ++++++++++")
    /* See DataFrame */
    println(df.show())

    println("++++++++++ QUESTION 1.D : SCHEMA ++++++++++")
    /* Schema */
    println("df schema : " + df.printSchema())

    println("++++++++++ QUESTION 1.E : CONVERT TYPES ++++++++++")
    /* Assign new type if necessary */
    val df2 = df.withColumn("final_status", df("final_status").cast("int"))
      .withColumn("backers_count", df("backers_count").cast("int"))
      .withColumn("created_at", df("created_at").cast("int"))
      .withColumn("state_changed_at", df("state_changed_at").cast("int"))
      .withColumn("goal", df("goal").cast("int"))
      .withColumn("deadline", df("deadline").cast("int"))
      .withColumn("launched_at", df("launched_at").cast("int"))

    println("df2 schema : " + df2.printSchema())

    /** 2 - CLEANING **/

    /* Number of elements in each class (final_status) */
    val groups = df2.groupBy("final_status").count()

    println("++++++++++ (OLD) QUESTION 2.F : NUMBER OF ELEMENTS IN EACH CLASS ++++++++++")
    println(groups.sort(groups("count").desc).show()) // print in a descending order for count

    println("++++++++++ (OLD) QUESTION 2.G : KEEP FINAL STATUS 0 AND 1 ++++++++++")
    /* Keep Success (=1) and Fail (=0) only */
    val df3 = df2.filter("final_status = 0 OR final_status = 1")

    // println(df3.select("final_status").show())
    // println(df3.show())

    /* Remove observations with goal < 0 */
    //val df4 = df3.filter("goal >= 0")
    val df4 = df3

    println("++++++++++ QUESTION 2.A : STATISTICAL DESCRIPTION ++++++++++")
    /* Statistical description */
    val int_var = Seq("final_status", "backers_count")
    println(df4.select(int_var.head, int_var.tail: _*)
      .describe()
      .show(50))

    println("++++++++++ QUESTION 2.B : DROP DUPLICATES ++++++++++")
    /* Drop duplicates */
    println("Number of lines in DataFrame : " + df4.count())
    val df5 = df4.dropDuplicates()
    println("Number of lines in DataFrame after dropping duplicates : " + df5.count())

    println("++++++++++ QUESTION 2.C : REMOVE COLUMN ++++++++++")
    /* Remove disable_communication */
    val df6 = df5.drop($"disable_communication")
    println(df6.printSchema())

    /* Remove backer_count and state_changed_at */
    println("++++++++++ QUESTION 2.D : DROP COLUMNS ++++++++++")
    val df7 = df6.drop($"backers_count").drop($"state_changed_at")
    println(df7.printSchema())
    println(df7.show(50))

    println("++++++++++ QUESTION 2.E : UDF COUNTRY AND CURRENCY ++++++++++")
    /* See that the country sometimes appears in currency */
    println(df7.filter("country is null")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50))

    /* udfs */
    def udf_country = udf { (country: String, currency: String) =>
      if (country == null)
        currency
      else
        country
    }

    def udf_currency = udf { (currency: String) =>
      if (currency != null)
        if (currency.length() != 3)
          null
        else
          currency
      else
        null
    }

    // Apply UDF
    val df7_modified = df7.withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))

    println(df7_modified.show(20))

    /* Check country2 now */
    println("AFTER CLEANING COUNTRY AND CURRENCY")
    println(df7_modified.filter("country is null")
      .groupBy("currency2")
      .count
      .orderBy($"count".desc)
      .show(50))

    println("++++++++++ QUESTION 2.F : NUMBER OF ELEMENTS IN EACH CLASS ++++++++++")
    val groups_modified = df7_modified.groupBy("final_status").count()
    println(groups_modified.sort(groups_modified("count").desc).show()) // print in a descending order for count


    /** 3 - FEATURE ENGINEERING **/
    println("++++++++++ QUESTION 3.A : CREATE DAYS CAMPAIGN ++++++++++")
    /* Create day_campaign column : number of DAYS */
    def udf_day_campaign = udf { (deadline: Integer, launched_at: Integer) =>
      if (deadline != null)
        if (launched_at != null)
          (deadline - launched_at) / (60 * 60 * 24)
        else
          -1
      else
        -1
    }

    val df_days_campaign = df7_modified.withColumn("days_campaign", udf_day_campaign($"deadline", $"launched_at") )
    // println(df_ts.show())

    println("++++++++++ QUESTION 3.B : CREATE HOUR PREPA ++++++++++")
    /* Create hours_prepa column : number of HOURS */
    def udf_hours_prepa = udf { (created_at: Integer, launched_at: Integer) =>
      if (created_at != null)
        if (launched_at != null)
          (launched_at - created_at) / (60 * 60)
        else
          -1.0
      else
        -1.0
    }

    val df_hours_prepa = df_days_campaign.withColumn("hours_prepa", udf_hours_prepa($"created_at", $"launched_at") )
    // println(df_hours_prepa.show())

    println("++++++++++ QUESTION 3.C : DELETE COLUMNS ++++++++++")
    /* Delete launched_at, deadline, created_at columns */
    val df_dropped_ts = df_hours_prepa.drop("launched_at").drop("deadline").drop("created_at")

    println("++++++++++ QUESTION 3.D : ADD TEXT COLUMN ++++++++++")
    /* Add column named text. Concatenate columns : name, desc, keywords */
    def udf_text = udf { (name : String, desc : String, keywords : String) =>
      name + " " + desc + " " + keywords
    }

    val df_text = df_dropped_ts.withColumn("text", udf_text($"name", $"desc", $"keywords") )

    println(df_text.printSchema())

    /** VALEURS NULLES **/
    println("++++++++++ QUESTION 4.A : NULL VALUES ++++++++++")
    /*def udf_null_val = udf { (myValue : Float) =>
      if (myValue != null)
        if (myValue != 0)
            myValue
        else
            -1.0
      else
        -1.0
    }

    val df_cleaned_null = df_text.withColumn("days_campaign", udf_null_val($"days_campaign"))
      .withColumn("hours_prepa", udf_null_val($"hours_prepa"))
      .withColumn("goal", udf_null_val($"goal"))*/

    val df_cleaned_null = df_text.filter("goal > 0")
      .na
      .fill(Map("days_campaign" -> -1,
      "hours_prepa" -> -1,
      "goal" -> -1))


    /** SAUVEGARDER UN DATAFRAME **/
    println("++++++++++ QUESTION 5 : SAVE DATAFRAME ++++++++++")
    println(df_cleaned_null.show())
    /*df_cleaned_null.write.format("com.databricks.spark.csv")
        .option("delimiter", ";")
        .save("train_cleaned_bis.csv")*/

    /*df0.write.format("com.databricks.spark.csv")
      .option("delimiter", ";")
      .save("train0.csv")*/
    df_cleaned_null.write.parquet("data_cleaned.parquet")

  }

}
