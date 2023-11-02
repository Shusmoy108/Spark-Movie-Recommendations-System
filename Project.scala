// Databricks notebook source
""""
Module Name: CSC 735 HW3
Author : Shusmoy Chowdhury
Date of Creation: 10/15/2023
Purpose: Working with Sark with Join and Window
"""
//importing necessary libraries
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.asc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
//loading  the dataset into databricks spark cluster
// val links = spark.read.option("header", "true") 
//  .option("inferSchema","true")
//  .csv("/FileStore/tables/links-1.csv")

val ratings = spark.read.option("header", "true") 
 .option("inferSchema","true")
.csv("/FileStore/tables/ratings-1.csv")
val tags = spark.read.option("header", "true") 
 .option("inferSchema","true")
 .csv("/FileStore/tables/tags-1.csv")

val movies = spark.read.option("header", "true") 
 .option("inferSchema","true")
.csv("/FileStore/tables/movies-4.csv")
// Creating table for the SQL
//df1.createOrReplaceTempView("movies_table") 
//df2.createOrReplaceTempView("movie_reviews_table")
//movies.show(false)
ratings.show(false)

// COMMAND ----------

//lnks.show(false)

// COMMAND ----------

movies.show(false)

// COMMAND ----------

tags.show(false)

// COMMAND ----------

var tagsdf= tags.groupBy("movieId").agg(collect_set("tag").alias("Tags")).orderBy("movieId")

// COMMAND ----------

var ratingDF=ratings.groupBy("movieId").agg(count("userId").alias("UserCount"),mean("rating").alias("AverageRating")).orderBy("movieId")
var joinmovierating= movies.join(ratingDF,"movieId").orderBy("movieId")
joinmovierating.show()



// COMMAND ----------

var maindf= joinmovierating.join(tagsdf,"movieId").orderBy("movieId")

// COMMAND ----------



// COMMAND ----------

import org.apache.spark.ml.feature.RegexTokenizer
val regexTokenizer = new RegexTokenizer()
  .setInputCol("genres")
  .setOutputCol("words")
  .setPattern("\\W")
//val tkn= new Tokenizer().setInputCol("genres")
val wordsData = regexTokenizer.transform(maindf.select("genres", "movieId"))

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

val hashingTF = new HashingTF()
  .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

val featurizedData = hashingTF.transform(wordsData)
// alternatively, CountVectorizer can also be used to get term frequency vectors

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("genreFeatures")
val idfModel = idf.fit(featurizedData)

val rescaledData = idfModel.transform(featurizedData).drop("genres","words","rawFeatures")

rescaledData.show(false)

// COMMAND ----------

var newdf= maindf.join(rescaledData,"movieId").orderBy("movieId")
//newdf.drop("words", "rawfeatures")
newdf.show()

// COMMAND ----------

val hashingTF2 = new HashingTF()
  .setInputCol("Tags").setOutputCol("rawFeatures").setNumFeatures(10)

val featurizedData2 = hashingTF2.transform(newdf.select("Tags", "movieId"))
// alternatively, CountVectorizer can also be used to get term frequency vectors

val idf2 = new IDF().setInputCol("rawFeatures").setOutputCol("tagFeatures")
val idfModel2 = idf2.fit(featurizedData2)

val rescaledData2 = idfModel2.transform(featurizedData2).drop("Tags","rawFeatures")

rescaledData2.show(false)

// COMMAND ----------

val finalDf= newdf.join(rescaledData2,"movieId").drop("Tags","genres")
val x=finalDf.withColumnRenamed("UserCount", "features")
finalDf.show(false)

// COMMAND ----------

// import org.apache.spark.ml.clustering.KMeans
// import org.apache.spark.ml.evaluation.ClusteringEvaluator

// // Loads data.
// //val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

// // Trains a k-means model.
// val kmeans = new KMeans().setK(2).setSeed(1L)
// val model = kmeans.fit(x)

// // Make predictions
// val predictions = model.transform(x)

// // Evaluate clustering by computing Silhouette score
// val evaluator = new ClusteringEvaluator()

// val silhouette = evaluator.evaluate(predictions)
// println(s"Silhouette with squared euclidean distance = $silhouette")

// // Shows the result.
println("Cluster Centers: ")

// COMMAND ----------

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
var assembler= new VectorAssembler().setInputCols(Array("UserCount","genreFeatures", "tagFeatures","AverageRating")).setOutputCol("features")
val featureDf=assembler.transform(finalDf)
featureDf.select("features").show(false)
//val km = new KMeans().setK(5)
//println(km.explainParams())
//val kmModel = km.fit(finalDf)


// COMMAND ----------

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

// Loads data.
//val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

// Trains a k-means model.
val kmeans = new KMeans().setK(20)
val model = kmeans.fit(featureDf)

// Make predictions
val predictions = model.transform(featureDf)

// COMMAND ----------

val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

// Shows the result.
println("Cluster Centers: ")

// COMMAND ----------

import org.apache.spark.ml.clustering.BisectingKMeans
val bkm = new BisectingKMeans().setK(10)
println(bkm.explainParams())
val bkmModel = bkm.fit(featureDf)
val summary = bkmModel.summary
summary.clusterSizes // number of points
bkmModel.computeCost(featureDf)

println("Cluster Centers: ")
bkmModel.clusterCenters.foreach(println)

// COMMAND ----------

predictions.show()

// COMMAND ----------

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

// Loads data.
//val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

// Trains a k-means model.

val evaluator = new ClusteringEvaluator()
        .setPredictionCol("prediction")
        .setMetricName("silhouette")
for(k <- 2 to 10 by 1 ){
    //clusteringScore0(numericOnly,k)
    val kmeansModel = new KMeans().setK(k).fit(featureDf)
    val transformedDF = kmeansModel.transform(featureDf)

    val score = evaluator.evaluate(transformedDF)

    println(k)
    println(score)
}

// COMMAND ----------

val id = predictions.select("prediction").where(col("title").equalTo("Copycat (1995)"))
var movi= predictions.join(id,"prediction")
movi.select("title", "AverageRating", "UserCount").where(col("title").notEqual("Copycat (1995)")).sort(desc("AverageRating"), desc("UserCount")).show()
//movi.limit(10).show()

// COMMAND ----------

predictions.groupBy("prediction").count().orderBy("prediction").show()

// COMMAND ----------

val summary = model.summary
summary.clusterSizes // number of points
//model.computeCost(featureDf)
//println("Cluster Centers: ")
//model.clusterCenters.foreach(println)

// COMMAND ----------

val summary = kmModel.summary
summary.clusterSizes // number of points
kmModel.computeCost(featureDf)
println("Cluster Centers: ")
kmModel.clusterCenters.foreach(println)

// COMMAND ----------

import org.apache.spark.ml.clustering.GaussianMixture
val gmm = new GaussianMixture().setK(10)
println(gmm.explainParams())
val gmmmodel = gmm.fit(featureDf)
val gmmsummary = gmmmodel.summary
gmmmodel.weights


// COMMAND ----------


import org.apache.spark.ml.clustering.GaussianMixture
val gmm = new GaussianMixture().setK(10)
val evaluator = new ClusteringEvaluator()
        .setPredictionCol("prediction")
        .setMetricName("silhouette")
for(k <- 5 to 30 by 5 ){
    //clusteringScore0(numericOnly,k)
    val kmeansModel =   new GaussianMixture().setK(k).fit(featureDf)
    val transformedDF = kmeansModel.transform(featureDf)

    val score = evaluator.evaluate(transformedDF)

    println(k)
    println(score)
}

// COMMAND ----------

gmmmodel.weights

// COMMAND ----------

gmmmodel.gaussiansDF.show()

// COMMAND ----------

gmmsummary.cluster.show()

// COMMAND ----------

gmmsummary.clusterSizes

// COMMAND ----------

gmmsummary.probability.show()

// COMMAND ----------

var gmmGaus= gmmmodel.transform(featureDf)
gmmGaus.show()

// COMMAND ----------

val idg = gmmGaus.select("prediction").where(col("title").equalTo("Copycat (1995)"))
var movig= gmmGaus.join(idg,"prediction")
movig.select("title","prediction").where(col("title").notEqual("Copycat (1995)")).sort(desc("AverageRating"), desc("UserCount")).show()

// COMMAND ----------

import org.apache.spark.ml.clustering.LDA
val lda = new LDA().setK(10).setMaxIter(5)
println(lda.explainParams())
val ldamodel = lda.fit(featureDf)

// COMMAND ----------

ldamodel.describeTopics(10).show()
//cvFitted.vocabulary

// COMMAND ----------

var ldapred= ldamodel.transform(featureDf)
ldapred.select("topicDistribution").show(false)

// COMMAND ----------

import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector
val evaluator = new ClusteringEvaluator()
        .setPredictionCol("prediction")
        .setMetricName("silhouette")
for(k <- 5 to 30 by 5 ){
    //clusteringScore0(numericOnly,k)
    val kmeansModel = new LDA().setK(k).fit(featureDf)
    val transformedDF = kmeansModel.transform(featureDf)
val func = udf( (x:Vector) => x.toDense.values.toSeq.indices.maxBy(x.toDense.values) )
val ldaprediction=transformedDF.withColumn("prediction",func($"topicDistribution"))
val score = evaluator.evaluate(ldaprediction)

    println(k)
    println(score)
}

// COMMAND ----------

ldapred.printSchema

// COMMAND ----------

import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector
val func = udf( (x:Vector) => x.toDense.values.toSeq.indices.maxBy(x.toDense.values) )

// val pos = ldapred.select($"*", posexplode($"topicDistribution"))
// val max_cols = pos
//   .groupBy("topicDistribution")
//   .agg(max("topicDistribution") as "max_col")
// val solution = pos
//   .join(max_cols, "topicDistribution")
//   .filter($"topicDistribution" === $"max_col")
//   .select("topicDistribution", "pos")

val ldaprediction=ldapred.withColumn("prediction",func($"topicDistribution"))
ldaprediction.groupBy("prediction").count().orderBy("prediction").show()

// COMMAND ----------

val idl = ldaprediction.select("prediction").where(col("title").equalTo("Copycat (1995)"))
var movil= ldaprediction.join(idl,"prediction")
movil.select("title","prediction").where(col("title").notEqual("Copycat (1995)")).show()

// COMMAND ----------

import org.apache.spark.ml.linalg.Vector
val ds =ldapred.select("topicDistribution")
val arrays = ds
  .map { r => r.getAs[Vector](0).toArray }
  .withColumnRenamed("value", "distribution")
//arrays.show()
val pos = arrays.select($"*", posexplode($"distribution"))
val max_cols = pos
  .groupBy("distribution")
  .agg(max("col") as "max_col")
val solution = pos
  .join(max_cols, "distribution")
  .filter($"col" === $"max_col")
  .select("distribution", "pos")
  solution.groupBy("pos").count().orderBy("pos").show()

// COMMAND ----------

val idl = ldaprediction.select("prediction").where(col("title").equalTo("Copycat (1995)"))
var movil= ldaprediction.join(idl,"prediction")
movil.select("title","prediction").where(col("title").notEqual("Copycat (1995)")).sort(desc("AverageRating"), desc("UserCount")).show()

// COMMAND ----------

// import org.apache.spark.ml.recommendation.ALS
// val als = new ALS()
// val alsModel = als.fit(featureDf)
// val predictions = alsModel.transform(featureDf)


// COMMAND ----------

// import org.apache.spark.ml.feature.MinHashLSH
// import org.apache.spark.mllib.linalg.Vectors
// val idl = ldaprediction.select("features","prediction").where(col("title").equalTo("Copycat (1995)"))
// var movil= ldaprediction.join(idl,"prediction")
// //movil.select("title","prediction").where(col("title").notEqual("Copycat (1995)")).show()
// val mh = new MinHashLSH()
//   .setNumHashTables(5)
//   .setInputCol("features")
//   .setOutputCol("hashes")
// val model = mh.fit(movil)
// //model.transform(movil).show()
// var values =idl.select("features").take(1)
// val doubVals = values.map{ row =>   row.getDouble(0) }
// print(doubVals)
// // val vector = Vectors.dense{ doubVals.collect}
// // model.approxNearestNeighbors(movil, vector, 2).show()
