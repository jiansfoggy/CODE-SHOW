/*
 * Illustrates a wordcount using Spark in Java
 */

import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Collection;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
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
      System.err.println("usage: WebRef master input output");
      System.exit(1);
    }

    // Create a Java Spark session and context
    SparkSession spark = SparkSession
      .builder()
      .master(args[0])
      .appName("WebRef")
      .getOrCreate();
    JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

    // Load our input data
    JavaRDD<String> inputs = sc.textFile(args[1]);
      
    JavaPairRDD<String, String> address = 
      inputs.flatMapToPair(
                           x -> {
                                 // split the string line to a list
                                 String[] adrs_array = x.split("\\s+");
                                 // create an empty list and wait for adding new key_value pair
                                 ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                                 // only deal with the array longer than one
                                 if (adrs_array.length > 1) {
                                   // use a for loop to make k_v pair
                                   for(int i=1; i< adrs_array.length; i++){
                                     pairs.add(new Tuple2(adrs_array[i],adrs_array[0]));
                                   }
                                 }
                                 return pairs.iterator();                            
                                }                                                         
      );
      
      // Transform into pairs and count
      JavaPairRDD<String,String> refs_pair = address.reduceByKey( (x, y) -> x.concat(" ".concat(y)));
      
      /*
      * The answer of task 2
      * Please commit this JavaPairRDD<String,String> refs_Pair 
      * = refs_pair.mapValues(x -> {...}) part and change 
      * refs_Pair.saveAsTextFile to refs_pair.saveAsTextFile
      * to check the purely result of task 1.
      */
      JavaPairRDD<String,String> refs_Pair= refs_pair.mapValues(x -> { 
                                                                  String[] split_again = Arrays.asList(x.split("\\s+")).stream()
                                                                                                        .sorted()
                                                                                                        .toArray(String[]::new); 
                                                                  String reorder_value = String.join(" ", split_again);
                                                                  return reorder_value;
                                                                  });

      // Save the word count back out to a text file, causing evaluation
      refs_Pair.saveAsTextFile(args[2]);
  }    
}

// /u/home/mikegoss/PDCPublic/Labs/Lab2Check/checkLab2.sh Short /u/home/jiasun/COMP4333/Lab2/out_yarn_short
// /u/home/mikegoss/PDCPublic/Labs/Lab2Check/checkLab2.sh Full /u/home/jiasun/COMP4333/Lab2/out_yarn_full
// /u/home/mikegoss/PDCPublic/Labs/Lab2Check/checkLab2.sh Short /u/home/jiasun/COMP4333/Lab2/output_short_sort
// /u/home/mikegoss/PDCPublic/Labs/Lab2Check/checkLab2.sh Full /u/home/jiasun/COMP4333/Lab2/output_full_sort

