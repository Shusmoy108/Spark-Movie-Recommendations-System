// Databricks notebook source
//importing necessary libraries
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.asc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

//loading the movies.csv, ratings.csv, tags.csv from movilens dataset url: https://grouplens.org/datasets/movielens/latest/
val movies = spark.read.option("header", "true") 
 .option("inferSchema","true") 
.csv("/FileStore/tables/movies-2.csv")
val ratings = spark.read.option("header", "true") 
 .option("inferSchema","true")
.csv("/FileStore/tables/ratings.csv")

val tags = spark.read.option("header", "true") 
 .option("inferSchema","true")
 .csv("/FileStore/tables/tags.csv")

// COMMAND ----------

//showing the data from movies.csv
movies.show(false)

// COMMAND ----------

//showing the data from ratings.csv
ratings.show(false)

// COMMAND ----------

//showing the data from tags.csv
tags.show(false)

// COMMAND ----------

//making a set of tags for each movies from the tags dataset
var tagsdf= tags.groupBy("movieId").agg(collect_set("tag").alias("Tags")).orderBy("movieId")
tagsdf.show()

// COMMAND ----------

//inner joining the movies and ratings dataset based on the movieId
var ratingDF=ratings.groupBy("movieId").agg(count("userId").alias("UserCount"),mean("rating").alias("AverageRating")).orderBy("movieId")
var joinmovierating= movies.join(ratingDF,"movieId").orderBy("movieId")
joinmovierating.show()

// COMMAND ----------

//inner joining the movie tags set dataset with joinmovierating table  based on the movieId
var maindf= joinmovierating.join(tagsdf,"movieId").orderBy("movieId")
maindf.show()

// COMMAND ----------

//using regex tokenizer for splitting the movie genres
import org.apache.spark.ml.feature.RegexTokenizer
val regexTokenizer = new RegexTokenizer()
  .setInputCol("genres")
  .setOutputCol("words")
  .setPattern("\\W")
//val tkn= new Tokenizer().setInputCol("genres")
val wordsData = regexTokenizer.transform(maindf.select("genres", "movieId"))
wordsData.show(false)

// COMMAND ----------

//making each genres to double values using hashingTF for feeding them into the clustering algorithm

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

//inner joiining the formatted genre features with main dataset
var newdf= maindf.join(rescaledData,"movieId").orderBy("movieId")
//newdf.drop("words", "rawfeatures")
newdf.show()

// COMMAND ----------

//making each tags to double values using hashingTF for feeding them into the clustering algorithm

val hashingTF2 = new HashingTF()
  .setInputCol("Tags").setOutputCol("rawFeatures").setNumFeatures(10)
val featurizedData2 = hashingTF2.transform(newdf.select("Tags", "movieId"))
val idf2 = new IDF().setInputCol("rawFeatures").setOutputCol("tagFeatures")
val idfModel2 = idf2.fit(featurizedData2)
val rescaledData2 = idfModel2.transform(featurizedData2).drop("Tags","rawFeatures")
rescaledData2.show()

// COMMAND ----------

//inner joiining the formatted tag features with main dataset
val finalDf= newdf.join(rescaledData2,"movieId").drop("Tags","genres")
finalDf.show()

// COMMAND ----------

// feature extraction using vector assembler with "UserCount","genreFeatures", "tagFeatures","AverageRating"
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
var assembler= new VectorAssembler().setInputCols(Array("UserCount","genreFeatures", "tagFeatures","AverageRating")).setOutputCol("features")
val featureDf=assembler.transform(finalDf)
featureDf.select("movieId","features").show()

// COMMAND ----------

// showing features after feature extraction
featureDf.select("features").show(false)

// COMMAND ----------

// finidng the training cost(within-cluster sum of squares (WCSS) ) for different cluster size and then used elbow method to find the op

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
for(k <- 5 to 30 by 5 ){
    val kmeansModel = new KMeans().setK(k).fit(featureDf)
    val transformedDF = kmeansModel.transform(featureDf)
    println(f"[$k, ${kmeansModel.summary.trainingCost}%1.14f]")
}

// COMMAND ----------

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator


// Trains the k-means model. We have choosen k=10 from the elbow method.
val kmeans = new KMeans().setK(10)
val model = kmeans.fit(featureDf)

// Make clusters with K Means Clustering
val predictions = model.transform(featureDf)
predictions.show()

// COMMAND ----------

//Showing the cluster sizes 
predictions.groupBy("prediction").count().orderBy("prediction").show()

// COMMAND ----------

//finding the recent highest rated movie for userId 1 
val recent = ratings.where(col("userId").equalTo(1)).sort(desc("timestamp")).limit(10)
val top = recent.sort(desc("rating")).limit(1)
top.show()

// COMMAND ----------

//finding the movie title and predicted cluster for the recent highest rated movie for userId 1 
val joinPred = predictions.join(top, "movieId").select("title","prediction")
joinPred.show()

// COMMAND ----------

// showing top ten movies from our movie recomendation system based on k means clustering
val id = predictions.select("prediction").where(col("title").equalTo("Gladiator (1992)"))
var movi= predictions.join(id,"prediction")
movi.select("title", "AverageRating", "UserCount").where(col("title").notEqual("Gladiator (1992)")).sort(desc("AverageRating"), desc("UserCount")).limit(10).show(false)

// COMMAND ----------

//calculating silhouette co-efficient for K means clustering 
val evaluator = new ClusteringEvaluator()
        .setPredictionCol("prediction")
        .setMetricName("silhouette")

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

// COMMAND ----------

import org.apache.spark.ml.clustering.GaussianMixture
// Trains the gmm model. We have choosen k=10 which we used in K Means clustering
val gmm = new GaussianMixture().setK(10)
val gmmmodel = gmm.fit(featureDf)
// Make clusters with gmm Clustering and showing them
var gmmGaus= gmmmodel.transform(featureDf)
gmmGaus.show()


// COMMAND ----------

//Showing the cluster sizes 
gmmGaus.groupBy("prediction").count().orderBy("prediction").show()

// COMMAND ----------

//calculating silhouette co-efficient for gmm clustering 
import org.apache.spark.ml.evaluation.ClusteringEvaluator
val evaluator = new ClusteringEvaluator()
        .setPredictionCol("prediction")
        .setMetricName("silhouette")

val silhouette = evaluator.evaluate(gmmGaus)
println(s"Silhouette with squared euclidean distance for gmm = $silhouette")

// COMMAND ----------

// showing top ten movies from our movie recomendation system based on gmm clustering
val idg = gmmGaus.select("prediction").where(col("title").equalTo("Gladiator (1992)"))
var movig= gmmGaus.join(idg,"prediction")
movig.select("title", "AverageRating", "UserCount").where(col("title").notEqual("Gladiator (1992)")).sort(desc("AverageRating"), desc("UserCount")).limit(10).show(false)

// COMMAND ----------

// Trains the lda model. We have choosen k=10 which we used in K Means clustering
import org.apache.spark.ml.clustering.LDA
val lda = new LDA().setK(10).setMaxIter(5)
val ldamodel = lda.fit(featureDf)

// COMMAND ----------

// generating the topic distribution with lda
var ldapred= ldamodel.transform(featureDf)
ldapred.select("topicDistribution").show(false)

// COMMAND ----------

//finding the clusters with maximum probability of topic distribution and Showing the cluster sizes based on that
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector
val func = udf( (x:Vector) => x.toDense.values.toSeq.indices.maxBy(x.toDense.values) )
val ldaprediction=ldapred.withColumn("prediction",func($"topicDistribution"))
ldaprediction.groupBy("prediction").count().orderBy("prediction").show()

// COMMAND ----------

//calculating silhouette co-efficient for lda clustering 
val silhouette = evaluator.evaluate(ldaprediction)
println(s"Silhouette with squared euclidean distance lda = $silhouette")

// COMMAND ----------

// showing top ten movies from our movie recomendation system based on lda clustering
val idl = ldaprediction.select("prediction").where(col("title").equalTo("Gladiator (1992)"))
var movil= ldaprediction.join(idl,"prediction")
movil.select("title", "AverageRating", "UserCount").where(col("title").notEqual("Gladiator (1992)")).sort(desc("AverageRating"), desc("UserCount")).limit(10).show(false)
