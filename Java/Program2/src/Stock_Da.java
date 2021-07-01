import scala.Tuple2;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.NavigableMap;
import java.util.DoubleSummaryStatistics;
import static java.lang.Double.parseDouble;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.LongAccumulator;


public class Stock_Data {

   /**
   	*  Specify five arguments:
    *  master - specify Spark master ("local" or "spark-yarn")
    *  input1 - specify input stock price data file(s)
    *  input2 - specify input company fundamental data file(s)
    *  output - specify output directory
    */
	public static void main(String[] args) {
		// Check arguments
    	if (args.length != 4) {
      	System.err.println(
              	"usage: Stock_Data master input1 input2 output");
      	System.exit(1);
    	}

    	// Create a Java Spark Context
    	SparkConf conf = new SparkConf().setMaster(args[0]).setAppName("Stock_Data");
    	JavaSparkContext sc = new JavaSparkContext(conf);

    	// Load our stock price data.
    	JavaRDD<String> input_p = sc.textFile(args[1]);
    	final LongAccumulator stkRcrdAcc = sc.sc().longAccumulator("StckCount");

    	// Load our fundamental data.
    	JavaRDD<String> input_f = sc.textFile(args[2]);
    	final LongAccumulator cpyRcrdAcc = sc.sc().longAccumulator("CmpyCount");

    	// Map each stock data record into key value pair
    	// JavaPairRDD<String, Double> stockPrice = input.mapPartitionsToPair(
    	JavaPairRDD<String, Double> stock1rnd = input_p.flatMapToPair(
    		x->{
    			// Create empty list of URL pairs for output
    			ArrayList<Tuple2<String, Double>> new_gen1 = new ArrayList<>();
    			// Split input into an array of URLs (split on comma)       
    			String price[] = x.split("\\,"); 
    			// Increment global price count
      			stkRcrdAcc.add(1);
    			new_gen1.add(new Tuple2<>(price[1], parseDouble(price[3])));
    			return new_gen1.iterator();
    		});

    	JavaPairRDD<String, Iterable<Double>> stock2rnd = stock1rnd.groupByKey();
    	stock2rnd.persist(StorageLevel.MEMORY_ONLY());
    	JavaPairRDD<String, ArrayList<Double>> stockPrice = stock2rnd.mapValues(
    		grps->{
    			ArrayList<String> valList1 = new ArrayList<>();
    			for (String s : grps) valList1.add(s);
    			Collections.sort(valList1);
    			return valList1;
    		});
    	Long stckNum = stockPrice.count();

    	JavaPairRDD<String, Double> cmpny1rnd = input_f.flatMapToPair(
    		y->{
    			// Create empty list of URL pairs for output
    			ArrayList<Tuple2<String, Double>> new_gen2 = new ArrayList<>();
    			// Split input into an array of URLs (split on comma)       
    			String fndmt[] = y.split("\\,");
    			// Increment global company count
      			cpyRcrdAcc.add(1); 
      			if(fndmt.length==79 && parseDouble(fndmt[78]) > 0){
      				new_gen2.add(new Tuple2<>(fndmt[1], 
      					parseDouble(fndmt[78])));
      				return new_gen2.iterator();
      			}
      		});

    	JavaPairRDD<String, Iterable<Double>> cmpny2rnd = cmpny1rnd.groupByKey();
    	cmpny2rnd.persist(StorageLevel.MEMORY_ONLY());
    	JavaPairRDD<String, Double> cmpnyData = cmpny2rnd.mapValues(
    		grps->{
    			ArrayList<String> valList2 = new ArrayList<>();
    			for (String s : grps) valList2.add(s);
    			Collections.sort(valList2);
    			Double estmtdShareAve = valList2.stream().mapToDouble(x->(double)x).average();
    			return estmtdShareAve;
    		});
    	Long cmpyNum = cmpnyData.count();

    	JavaPairRDD<String, Tuple2<ArrayList<Double>, Double>> joinList = stockPrice.join(cmpnyData);
        Long valid_join = joinList.count();

        Long n = cmpyNum - valid_join;
        Long m = stckNum - valid_join;

        System.out.println(">> Missing price tickers: "+n);
        System.out.println(">> Missing fundamentals tickers: "+m);
        System.out.println(">> Stock price records read: "+stkRcrdAcc.value());
        System.out.println(">> Fundamentals records read: "+cpyRcrdAcc.value());

        JavaPairRDD<String, String> output1 = joinList.mapValues(
        	z->{
        		ArrayList<Double> closeList = z._1;
        		ArrayList<Double> valuation = closeList.stream().map(i->i*z._2)
        											   .toArray(Double[]::new);

        		float pMean = closeList.stream().mapToDouble(x->(double)x).mean().floatValue();
        		float pMin  = closeList.stream().mapToDouble(x->(double)x).min().floatValue();
        		float pMax  = closeList.stream().mapToDouble(x->(double)x).max().floatValue();
        		float pVar  = closeList.stream().mapToDouble(x->(double)x).variance().floatValue();
        		float vMean = valuation.stream().mapToDouble(x->(double)x).mean().floatValue();
        		float vMin  = valuation.stream().mapToDouble(x->(double)x).min().floatValue();
        		float vMax  = valuation.stream().mapToDouble(x->(double)x).max().floatValue();
        		float vVar  = valuation.stream().mapToDouble(x->(double)x).variance().floatValue();
 
 				String concate = pMean+" "+pMin+" "+pMax+" "+pVar+" "+vMean+" "+vMin+" "+vMax+" "+vVar;
 				return concate;
        	});

        // Sort the entire list by stock ticker and output to file(s)
    	output1.sortByKey().saveAsTextFile(args[2]);

        JavaPairRDD<String, MaxMinVar> output2 = joinList.mapValues(
        	x->{
        		MaxMinVar extra = new MaxMinVar();
        		ArrayList<Double> closeList = z._1;
        		ArrayList<Double> valuation = closeList.stream().map(i->i*z._2)
        											   .toArray(Double[]::new);
        		Double pVar  = closeList.stream().mapToDouble(x->(double)x).variance();
        		Double vVar  = valuation.stream().mapToDouble(x->(double)x).variance();
                
                extra.addVar(pVar, vVar);
            });
    	// Compute statistics
    	MaxMinVar stats = output2.map(nc->nc._2)
    							 .reduce((stats1, stats2) -> {
    							 	 stats1.mergeOther(stats2);
    							 	 return stats1;
    							 	});
        
    	System.out.println(">> Stock with maximum price variance: " + 
    		stats.maxpVal.floatValue());
		System.out.println(">> Stock with minimum price variance: " + 
			stats.minpVal.floatValue());
		System.out.println(">> Stock with maximum valuation variance: " + 
			stats.maxvVal.floatValue());
		System.out.println(">> Stock with minimum valuation variance: " + 
			stats.minvVal.floatValue());
	}

	static class MaxMinVar implements Serializable {
		final Double minpVal = Double.POSITIVE_INFINITY;
    	final Double maxpVal = Double.NEGATIVE_INFINITY;
    	final Double minvVal = Double.POSITIVE_INFINITY;
    	final Double maxvVal = Double.NEGATIVE_INFINITY;

    	public void addVar(Double pVal, Double vVal) {
    		minpVal = Math.min(minpVal, pVal);
      		maxpVal = Math.max(maxpVal, pVal);
      		minvVal = Math.min(minvVal, vVal);
      		maxvVal = Math.max(maxvVal, vVal);
      	}

      	public void mergeOther(MaxMinVar other) {
      		minpVal = Math.min(minpVal, other.minpVal);
      		maxpVal = Math.max(maxpVal, other.maxpVal);
      		minvVal = Math.min(minvVal, other.minvVal);
      		maxvVal = Math.max(maxvVal, other.maxvVal);
      	}
      }
}