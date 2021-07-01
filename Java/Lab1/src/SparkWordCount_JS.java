/**
 * Illustrates a wordcount using Spark in Java
 */

import java.util.Arrays;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class SparkWordCount_JS {

  /* 
   *  Specify three arguments:
   *    master - specify Spark master ("local" or "yarn")
   *    input - specify input file
   *    output - specify output file
   */
  public static void main(String[] args) throws Exception {
    // Check arguments
    if (args.length != 3) {
      System.err.println("usage: SparkWordCount_JS master input output");
      System.exit(1);
    }

    // Create a Java Spark session and context
    SparkSession spark = SparkSession
      .builder()
      .master(args[0])
      .appName("SparkWordCount_JS")
      .getOrCreate();
    JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

    // Load our input data
    JavaRDD<String> input = sc.textFile(args[1]);

    // Split up into words
    JavaPairRDD<String,Long> words = 
      input.flatMapToPair( 
                          x -> 
                            Arrays.asList(x.split("\\W+"))
                              .stream()
                              .map( w -> new Tuple2<>(w, 2L) )
                              .iterator()
                           );

    // Transform into pairs and count
    JavaPairRDD<String,Long> counts = words.reduceByKey( (x, y) -> x + y );

    // Save the word count back out to a text file, causing evaluation
    counts.saveAsTextFile(args[2]);
  }
}
