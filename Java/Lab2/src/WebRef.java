/**
 * Illustrates a wordcount using Spark in Java
 */

import java.util.Arrays;
import java.util.ArrayList;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class WebRef {

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

    for(String input:inputs.collect()){
        JavaPairRDD<String,String> words = 
            input.flatMapToPair(
              new PairFlatMapFunction<String, String, String>(){
                public Iterator<Tuple2<String,String>> call(String x) {
                  String[] words_array = x.split("\\s+");
                  ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                  for(int i=1; i< words_array.length; i++){
                    pairs.add(new Tuple2(words_array[i],words_array[0]));
                  }
                  return pairs.iterator();
                }
        });

        // Transform into pairs and count
        // JavaPairRDD<String,Long> counts = words.reduceByKey( (x, y) -> x + y );
        JavaPairRDD<String,String> counts = words.reduceByKey( (x, y) -> x.concat("\\s+".concat(y)));

        // Save the word count back out to a text file, causing evaluation
        counts.saveAsTextFile(args[2]);


        JavaPairRDD<String, String> words = 
            input.flatMapToPair(x -> {
                                  String[] words_array = x.split("\\s+");
                                  ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                                  for(int i=1; i< words_array.length; i++){
                                      pairs.add(new Tuple2(words_array[i],words_array[0]));
                                  }
                                  pairs.iterator();

            });
        // Transform into pairs and count
        // JavaPairRDD<String,Long> counts = words.reduceByKey( (x, y) -> x + y );
        JavaPairRDD<String,String> counts = words.reduceByKey( (x, y) -> x.concat("\\s+".concat(y)));

        // Save the word count back out to a text file, causing evaluation
        counts.saveAsTextFile(args[2]);
    }    
  }
}

JavaPairRDD<String,String> refs_pair_in_order = refs_pair.mapValues(x -> { 
                                                                      String[] split_again = Arrays.asList(x.split("\\s+").stream()
                                                                                              .sorted()
                                                                                              .toArray(String[]::new)); 
                                                                      String reorder_value = String.join(" ", split_again);
                                                                    });


JavaPairRDD<String,String> new_order = counts.mapValues(x -> Arrays.asList(x.split("\\s+"))
                                                            .stream().sorted().toArray(String[]::new)
                                                            .map((i,j) -> i.concat(" ".concat(j)))
                                                        ); 
String.join("\\s+", strArray)
// Java 8 Streams are used here to do the sorting and iterations
String[] sortedPlanets = Arrays.stream(planets) // Create the stream
                               .sorted() // Sort the stream
                               .toArray(String[]::new); // Convert back to String[]


JavaPairRDD<String,String> words = 
      input.flatMapToPair(
        new PairFlatMapFunction<String, String, String>(){
          public Iterator<Tuple2<String,String>> call(String x) {
            String[] words_array = x.split("\\s+");
            ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
            for(int i=1; i< words_array.length; i++){
              pairs.add(new Tuple2(words_array[i],words_array[0]));

            }
            return pairs.iterator();
          }
        });

    JavaPairRDD<String,String> words = 
      input.flatMapToPair( 
                          x -> 
                            Arrays.asList(x.split("\\s+"))
                              .stream()
                              .filter(y -> y != Arrays.asList(x.split("\\s+"))[0])
                              .map( w -> new Tuple2<>(w, Arrays.asList(x.split("\\s+"))[0]) )
                              .iterator()
                          );
    
    JavaPairRDD<String,String> words = 
      input.flatMapToPair( 
                          x -> { web_list = Arrays.asList(x.split("\\s+"))
                                              .stream();
                                 web_kv = web_list.filter(y -> y != web_list[0])
                                                  .map(w -> new Tuple2<>(w, web_list[0]))
                                                  .iterator();
                          }                         
                          );

    

    for(String input:inputs.collect()){
        JavaPairRDD<String,String> words = 
            input.flatMapToPair( 
                                x -> {web_list = Arrays.asList(x.split("\\s+"))
                                                .stream();
                                      web_kv = web_list.filter(y -> y != web_list[0])
                                                .map(w -> new Tuple2<>(w, web_list[0]))
                                                .iterator();
                                      }                         
                               );

        }


  for(int k = 0; k <   String input:inputs.collect()){
        JavaPairRDD<String, String> words = 
            input.flatMapToPair(x -> {
                                  String[] words_array = x.split("\\s+");
                                  ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                                  for(int i=1; i< words_array.length; i++){
                                      pairs.add(new Tuple2(words_array[i],words_array[0]));
                                  }
                                  pairs.iterator();

            })
        // Transform into pairs and count
        // JavaPairRDD<String,Long> counts = words.reduceByKey( (x, y) -> x + y );
        JavaPairRDD<String,String> counts = words.reduceByKey( (x, y) -> x.concat("\\s+".concat(y)));

        // Save the word count back out to a text file, causing evaluation
        counts.saveAsTextFile(args[2]);
    }
  JavaPairRDD<String,String> words = {
       input.flatMapToPair({String[] words_array = x -> Arrays.asList(x.split("\\s+")).stream();
                            ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                            if (words_array.length > 1) {
                                   for(int i=1; i< words_array.length; i++){
                                     pairs.add(new Tuple2(words_array[i],words_array[0]));
                                   }
                                   pairs.iterator();
                            } else {continue;}        


       });
     }
                           x -> {
                                // split the string line to a list
                             String[] words_array = x.split("\\s+");
                             // words_array = Arrays.asList(x.split("\\s+")).stream();
                             // create an empty list and wait for adding new key_value pair                                  ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                             // use a for loop to make k_v pair
                             // create an empty list and wait for adding new key_value pair
                             ArrayList<Tuple2<String, String>> pairs = new ArrayList<>();
                             // use a for loop to make k_v pair
                             for(int i=1; i< words_array.length; i++){
                                   pairs.add(new Tuple2(words_array[i],words_array[0]));
                             }
                             pairs.iterator();
       })
    }
  };
      
        




