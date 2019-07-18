// Databricks notebook source
// MAGIC %md
// MAGIC #### Q2 - Skeleton Scala Notebook
// MAGIC This template Scala Notebook is provided to provide a basic setup for reading in / writing out the graph file and help you get started with Scala.  Clicking 'Run All' above will execute all commands in the notebook and output a file 'examplegraph.csv'.  See assignment instructions on how to to retrieve this file. You may modify the notebook below the 'Cmd2' block as necessary.
// MAGIC 
// MAGIC #### Precedence of Instruction
// MAGIC The examples provided herein are intended to be more didactic in nature to get you up to speed w/ Scala.  However, should the HW assignment instructions diverge from the content in this notebook, by incident of revision or otherwise, the HW assignment instructions shall always take precedence.  Do not rely solely on the instructions within this notebook as the final authority of the requisite deliverables prior to submitting this assignment.  Usage of this notebook implicitly guarantees that you understand the risks of using this template code. 

// COMMAND ----------

/*
DO NOT MODIFY THIS BLOCK
This assignment can be completely accomplished with the following includes and case class.
Do not modify the %language prefixes, only use Scala code within this notebook.  The auto-grader will check for instances of <%some-other-lang>, e.g., %python
*/
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._
case class edges(Source: String, Target: String, Weight: Int)
import spark.implicits._

// COMMAND ----------

/* 
Create an RDD of graph objects from our toygraph.csv file, convert it to a Dataframe
Replace the 'examplegraph.csv' below with the name of Q2 graph file.
*/

val df = spark.read.textFile("/FileStore/tables/bitcoinotc.csv") 
  .map(_.split(","))
  .map(columns => edges(columns(0), columns(1), columns(2).toInt)).toDF()

// COMMAND ----------

// View the imported dataset
df.show()

// COMMAND ----------

// 1. eliminate duplicate edges
val df_dedupe = df.dropDuplicates()
df_dedupe.show()

// COMMAND ----------

// 2. filter nodes by edge weight >= 5
val df_filtered = df_dedupe.filter(col("Weight") >= 5)
df_filtered.show()

// COMMAND ----------

// find node with highest weighted-in-degree, if two or more nodes have the same weighted-in-degree, report the one with the lowest node id
// find node with highest weighted-out-degree, if two or more nodes have the same weighted-out-degree, report the one with the lowest node id
// find node with highest weighted-total degree, if two or more nodes have the same weighted-total-degree, report the one with the lowest node id
//toDF, join, select, groupBy, orderBy, agg

// Aggregate and store weighted-out and weight-in degrees for each node then join
val weight_out = df_filtered.groupBy("Source").sum("Weight").toDF("node","weighted-out-degree")
val weight_in = df_filtered.groupBy("Target").sum("Weight").toDF("node2","weighted-in-degree")

val node_weights = weight_out.join(
    weight_in
,  weight_out("node") <=> weight_in("node2")
, "inner")
.select("node", "weighted-out-degree", "weighted-in-degree")

// Add weighted-in and weighted-out into a final total column
val columnsToSum = List(col("weighted-out-degree"), col("weighted-in-degree"))
val summed_nodes = node_weights.withColumn("weighted-total-degree", columnsToSum.reduce(_ + _))

// Store the max value for all weight columns
val maxes = summed_nodes.agg(max("weighted-out-degree"), max("weighted-in-degree"), max("weighted-total-degree"))
val out_max = maxes.first().getAs[Long](0)
val in_max = maxes.first().getAs[Long](1)
val total_max = maxes.first().getAs[Long](2)

// Use the max values stored above to filter the dataframes, get min node of each data frame
val out_df = summed_nodes.filter(col("weighted-out-degree") >= out_max)
                         .withColumn("node_int", expr("CAST(node as Int)"))
                         .orderBy($"node_int")
                         .select("node_int", "weighted-out-degree")
                         .groupBy($"weighted-out-degree")
                         .agg(min($"node_int") as "node")
                         .orderBy(asc("node"))
                         .select("node", "weighted-out-degree")
                         .withColumn("c", lit("o"))

val in_df = summed_nodes.filter(col("weighted-in-degree") >= in_max)
                         .withColumn("node_int", expr("CAST(node as Int)"))
                         .orderBy($"node_int")
                         .select("node_int", "weighted-in-degree")
                         .groupBy($"weighted-in-degree")
                         .agg(min($"node_int") as "node")
                         .orderBy(asc("node"))
                         .select("node", "weighted-in-degree")
                         .withColumn("c", lit("i"))

val total_df = summed_nodes.filter(col("weighted-total-degree") >= total_max)
                         .withColumn("node_int", expr("CAST(node as Int)"))
                         .orderBy($"node_int")
                         .select("node_int", "weighted-total-degree")
                         .groupBy($"weighted-total-degree")
                         .agg(min($"node_int") as "node")
                         .orderBy(asc("node"))
                         .select("node", "weighted-total-degree")
                         .withColumn("c", lit("t"))


// COMMAND ----------

/*
Create a dataframe to store your results
Schema: 3 columns, named: 'v', 'd', 'c' where:
'v' : vertex id
'd' : degree calculation (an integer value.  one row with highest weighted-in-degree, a row w/ highest weighted-out-degree, a row w/ highest weighted-total-degree )
'c' : category of degree, containing one of three string values:
                                                'i' : weighted-in-degree
                                                'o' : weighted-out-degree                                                
                                                't' : weighted-total-degree
- Your output should contain exactly three rows.  
- Your output should contain exactly the column order specified.
- The order of rows does not matter.
                                                
A correct output would be:

v,d,c
4,15,i
2,20,o
2,30,t

whereas:
- Node 2 has highest weighted-out-degree with a value of 20
- Node 4 has highest weighted-in-degree with a value of 15
- Node 2 has highest weighted-total-degree with a value of 30

*/

val final_df = in_df.union(out_df).union(total_df)
.toDF("v","d","c")

// COMMAND ----------

display(final_df)
