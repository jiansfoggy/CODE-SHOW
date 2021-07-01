/**
 * WebRefs - find all pages which refer to a given URL
 */

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.lang.Iterable;
import java.util.Iterator;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;


public class WebRefs {

  /* 
   *  Specify three arguments:
   *    master - specify Spark master ("local" or "yarn")
   *    input - specify input file
   *    output - specify output file
   */
  public static void main(String[] args) throws Exception {
    // Check arguments
    if (args.length != 3) {
      System.err.println("usage: SparkWordCount master input output");
      System.exit(1);
    }

    // Create a Java Spark session and context
    SparkSession spark = SparkSession
      .builder()
      .master(args[0])
      .appName("SparkWordCount")
      .getOrCreate();
    JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

    // Load our input data.
    JavaRDD<String> input = sc.textFile(args[1]);

    // Split up into URLs, delimited by white space. Output key/value pairs
    JavaPairRDD<String, String> urlRefs = input.flatMapToPair(
      x -> {
        // Create empty list of URL pairs for output
        ArrayList<Tuple2<String,String>> refs = new ArrayList<>();

        // Split input into an array of URLs (split on whitespace)
        String urls[] = x.split("\\s+");

        // For each URL after the first, generate a pair for output where the key is
        // the referred-to URL, and the value is the URL referring to it
        for (int i = 1; i < urls.length; ++i) {
          refs.add(new Tuple2<>(urls[i], urls[0]));
        }

        return refs.iterator();
      }
    );

    // Gather all references to each URL
    JavaPairRDD<String,String> allRefs =
      urlRefs.reduceByKey( (r1,r2) -> r1 + " " + r2 );

    // Sort the references alphabetically
    JavaPairRDD<String,String> allRefsSorted =
      allRefs.mapValues( refs -> {
          // Split refs string into separate URLs
          String s[] = refs.split("\\s+");

          // Sort the array of URLs
          Arrays.sort(s);

          // Output a string of the sorted URLs
          return String.join(" ", s);
        });

    // Save the references back to the output file
    allRefsSorted.saveAsTextFile(args[2]);
  }
}
