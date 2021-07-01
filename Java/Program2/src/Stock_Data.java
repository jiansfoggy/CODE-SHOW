/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import scala.Tuple2;
import java.util.Arrays;
import java.util.ArrayList;
import java.io.Serializable;
import java.util.Collections;
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

        /*
         * Stock Price Data Part
         */
    	// Map each stock data record into key value pair
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
    			ArrayList<Double> valList1 = new ArrayList<>();
    			for (Double s : grps) valList1.add(s);
    			Collections.sort(valList1);
    			return valList1;
            });
    	Long stckNum = stockPrice.count();

        /*
         * Company Fundamental Data Part
         */
         // Map each company data record into key value pair
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
                } 
                return new_gen2.iterator();
      		});

    	JavaPairRDD<String, Iterable<Double>> cmpny2rnd = cmpny1rnd.groupByKey();
    	cmpny2rnd.persist(StorageLevel.MEMORY_ONLY());
    	JavaPairRDD<String, Double> cmpnyData = cmpny2rnd.mapValues(
    		grps->{
                ArrayList<Double> valList2 = new ArrayList<>();
                for (Double s : grps) valList2.add(s);
                Collections.sort(valList2);
                Double estmtdShareAve = valList2.stream().mapToDouble(x->(double)x).average().orElse(0.0);
                return estmtdShareAve;
            });
    	Long cmpyNum = cmpnyData.count();

        /*
         * Part After Join
         */
    	JavaPairRDD<String, Tuple2<ArrayList<Double>, Double>> joinList = stockPrice.join(cmpnyData);
        Long valid_join = joinList.count();

        Long n = cmpyNum - valid_join;
        Long m = stckNum - valid_join;

        System.out.println(">> Missing price tickers: "+n);
        System.out.println(">> Missing fundamentals tickers: "+m);
        System.out.println(">> Stock price records read: "+stkRcrdAcc.value());
        System.out.println(">> Fundamentals records read: "+cpyRcrdAcc.value());

        JavaPairRDD<String, String> output1 = joinList.mapValues(
        	z->{Double[] closeList = z._1.stream().toArray(Double[]::new);
                Double[] valuation = z._1.stream().map(i->i*z._2).toArray(Double[]::new);
                //ArrayList<Double> closeList = z._1;
                // Double[] valuation = closeList.stream().map(i->i*z._2).toArray(Double[]::new);
                // sorting array
                Arrays.sort(closeList);
                Arrays.sort(valuation);
                String pStat = calList(closeList);
                String vStat = calList(valuation);
                
                return pStat+" "+vStat;
        	});

        // Sort the entire list by stock ticker and output to file(s)
    	output1.sortByKey().saveAsTextFile(args[3]);

        JavaPairRDD<String, MaxMinVar> output2 = joinList.flatMapToPair(
            z->{// create an empty tuple list for return value
                ArrayList<Tuple2<String, MaxMinVar>> grad_part = new ArrayList<>();
                // separate the key value and value value
                String keyPart = z._1;
                Tuple2<ArrayList<Double>, Double> valPart = z._2;
                // create two array to hold price list and valuation list
                Double[] closeList = valPart._1.stream().toArray(Double[]::new);
                Double[] valuation = valPart._1.stream().map(i->i*valPart._2).toArray(Double[]::new);
                // calculate var for price and valuation
                Double pVar = extctVar(closeList);
                Double vVar = extctVar(valuation);
                // define a new MaxMinVar class and load value to compare
                MaxMinVar extra = new MaxMinVar();  
                extra.addVar(keyPart, pVar, vVar);
                grad_part.add(new Tuple2<>(keyPart, extra));
                return grad_part.iterator();
            });
    	// Compute statistics
    	MaxMinVar stats = output2.map(nc->nc._2)
                                .reduce((stats1, stats2) -> {
                                    stats1.mergeOther(stats2);
                                    return stats1;
                                });
        
    	System.out.println(">> Stock with maximum price variance: " + 
            stats.maxpTkr + " " + stats.maxpVal.floatValue());
		System.out.println(">> Stock with minimum price variance: " + 
			stats.minpTkr+ " " + stats.minpVal.floatValue());
		System.out.println(">> Stock with maximum valuation variance: " + 
			stats.maxvTkr+ " " + stats.maxvVal.floatValue());
		System.out.println(">> Stock with minimum valuation variance: " + 
			stats.minvTkr+ " " + stats.minvVal.floatValue());
	}

	static class MaxMinVar implements Serializable {
            String minpTkr;
            String maxpTkr;
            String minvTkr;
            String maxvTkr;
            Double minpVal = Double.POSITIVE_INFINITY;
    	    Double maxpVal = Double.NEGATIVE_INFINITY;
            Double minvVal = Double.POSITIVE_INFINITY;
            Double maxvVal = Double.NEGATIVE_INFINITY;

            public void addVar(String ticker, Double pVal, Double vVal) {
                if (pVal < minpVal){
                    minpVal = pVal;
                    minpTkr = ticker;
                }
                if (pVal >= maxpVal){
                    maxpVal = pVal;
                    maxpTkr = ticker;
                }
                if (vVal < minvVal){
                    minvVal = vVal;
                    minvTkr = ticker;
                } 
                if (vVal >= maxvVal){
                    maxvVal = vVal;
                    maxvTkr = ticker;
                } 
                
                //minpVal = Math.min(minpVal, pVal);
                //maxpVal = Math.max(maxpVal, pVal);
                //minvVal = Math.min(minvVal, vVal);
                //maxvVal = Math.max(maxvVal, vVal);
            }

            public void mergeOther(MaxMinVar other) {
                if (other.minpVal < minpVal){
                    minpVal = other.minpVal;
                    minpTkr = other.minpTkr;
                }
                if (other.maxpVal >= maxpVal){
                    maxpVal = other.maxpVal;
                    maxpTkr = other.maxpTkr;
                }
                if (other.minvVal < minvVal){
                    minvVal = other.minvVal;
                    minvTkr = other.minvTkr;
                }
                if (other.maxvVal >= maxvVal){
                    maxvVal = other.maxvVal;
                    maxvTkr = other.maxvTkr;
                }
                //minpVal = Math.min(minpVal, other.minpVal);
                //maxpVal = Math.max(maxpVal, other.maxpVal);
                //minvVal = Math.min(minvVal, other.minvVal);
                //maxvVal = Math.max(maxvVal, other.maxvVal);
            }
        }
        
        public static String calList (Double[] xs){
            float minValue = xs[0].floatValue();
            float maxValue = xs[xs.length-1].floatValue();
            Double total = 0.0;
            for (Double x : xs) {
                total = total + x;
            }
            Double averVal = total/xs.length;
            float aveValue = averVal.floatValue();
            Double difValue = 0.0;
            for (Double x : xs) {
                difValue = difValue + Math.pow(x-averVal,2);
            }
            Double varVal1 = difValue/xs.length;
            float varValue = varVal1.floatValue();
            String shuchu = aveValue+" "+minValue+" "+maxValue+" "+varValue;
            return shuchu;            
	}
        
        public static Double extctVar (Double[] xs){
            Double total = 0.0;
            Double difValue = 0.0;
            for (Double x : xs) {
                total = total + x;
            }
            Double averVal = total/xs.length;
            for (Double x : xs) {
                difValue = difValue + Math.pow(x-averVal,2);
            }
            Double varValue = difValue/xs.length;
            return varValue;           
	}
}

/* 
Short Version Dataset
>> Missing price tickers: 0
>> Missing fundamentals tickers: 0
>> Stock price records read: 5286
>> Fundamentals records read: 12
>> Stock with maximum price variance: AAPL 802.17834
>> Stock with minimum price variance: MSFT 116.804794
>> Stock with maximum valuation variance: AAPL 1.667692E22
>> Stock with minimum valuation variance: IBM 6.712606E20

./run_spark_yarn_s.sh
sort out_yarn_short/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/short_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0094 > logs_short.tmp
grep '^>' logs_short.tmp
less logs_short.tmp

Full Version Dataset
>> Missing price tickers: 1
>> Missing fundamentals tickers: 70
>> Stock price records read: 851264
>> Fundamentals records read: 1781
>> Stock with maximum price variance: PCLN 154120.27
>> Stock with minimum price variance: PBCT 2.4781508
>> Stock with maximum valuation variance: AAPL 1.667692E22
>> Stock with minimum valuation variance: CSRA 2.36561284E17

./run_spark_yarn_f.sh
sort out_yarn_full/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/full_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0095 > logs_full.tmp
grep '^>' logs_full.tmp
less logs_full.tmp

*/
