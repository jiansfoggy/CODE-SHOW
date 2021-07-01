import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.NavigableMap;
import java.util.Optional;

import org.apache.hadoop.conf.Configuration;
//import org.apache.hadoop.fs.shell.CopyCommands.Put;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat;
import org.apache.hadoop.mapreduce.Job;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.graphx.Edge;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;
//import shapeless.ops.zipper.Put;

class ParseCSV {
	/**
 	* ParseCSV - parse a line from a CSV file, with optional double quotes
	*
   	* parseLine - separate input line into fields at commas, respecting
   	*             quoted strings.  We remove quotes around strings.
   	*/
  	static String[] parseLine(String text) {
    	String[] split = text.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
    
    	for (int i = 0; i < split.length; ++i) {
      		String s = split[i];
      	    if (s.startsWith("\"") && s.endsWith("\"")) {
        		split[i] = s.substring(1, s.length() - 1);
      		}
    	}
    	return split;
  	}
}

public class BuildMovieTable{
	public static void main(String[] args) {
		if (args.length!=3){
			System.err.println("usage: BuildMovieTable master inputTable outputTable");
		}
		String outputTable = args[2];

		// Create a java spark session and context
		SparkSession spark = SparkSession.builder().master(args[0])
							 .appName("SparkHBase").config("spark.hadoop.validateOutputSpecs", false)
							 .getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

		JavaRDD<String> input_mv = sc.textFile(args[1]+"/movie.csv");
		System.out.println(">> movie number: "+input_mv.count());
		JavaRDD<String> input_tg = sc.textFile(args[1]+"/tag.csv");
		System.out.println(">> tag number: "+input_tg.count());
		//JavaRDD<String> input_rt = sc.textFile(args[1]+"/rating/rating.csv");
    JavaRDD<String> input_rt = sc.textFile(args[1]+"/rating");
    System.out.println(">> rate number: "+input_rt.count());        
    // For movie file
		JavaPairRDD<String, ValMTR>moviePart = input_mv.mapPartitionsToPair(
			mv ->{
				  // Create empty arraylist to hold values
				  ArrayList<Tuple2<String, ValMTR>> m_kv = new ArrayList<>();
				  while (mv.hasNext()) {
					// Split the line as string list
					String mv1 = mv.next();
					String mrowList[] = ParseCSV.parseLine(mv1.toString());
					System.out.println("> mv mapPartitionsToPair: "+mrowList.length);
					// Assign values
					String mvKey = mrowList[0];
					String titleVal = mrowList[1];
					String genresVal = mrowList[2];
					System.out.println("> genre: "+genresVal);

					ValPac mvTpl = new ValPac(titleVal,genresVal);
					ValMTR mm = new ValMTR(mvTpl, null, null);

					// Save values back
					m_kv.add(new Tuple2<>(mvKey, mm));
				}
				return m_kv.iterator();
			});

		// For tag file
		JavaPairRDD<String, ValPac>tagPart = input_tg.mapPartitionsToPair(
			tg ->{
				  // Create empty arraylist to hold value
				  ArrayList<Tuple2<String, ValPac>> t_kv = new ArrayList<>();
				  while (tg.hasNext()) {
					// Split the line as string list
					String tg1 = tg.next();
					String trowList[] = ParseCSV.parseLine(tg1.toString());
					// Assign values
					String tgKey = trowList[1];
					String tusrVal = trowList[0];
					// tag | timestamp
					String ttVal = trowList[2]+"|"+trowList[3];
					ValPac tgTpl = new ValPac(tusrVal,ttVal);
                	// Save values back
					t_kv.add(new Tuple2<>(tgKey, tgTpl));
				}
				return t_kv.iterator();
			});

	  JavaPairRDD<String, Iterable<ValPac>> taggrp = tagPart.groupByKey();
    JavaPairRDD<String, ValMTR> tagGrp = taggrp.mapValues(
        tg ->{
            ArrayList<ValPac> retnVal = new ArrayList<>();
            for (ValPac v : tg) retnVal.add(v);
            ValMTR tt = new ValMTR(null, retnVal, null);
          	return tt;
          });

		// For rate file
		JavaPairRDD<String, ValPac>ratePart = input_rt.mapPartitionsToPair(
			  rt ->{
				    // Create empty arraylist to hold value
				    ArrayList<Tuple2<String, ValPac>> r_kv = new ArrayList<>();
				    while (rt.hasNext()) {
					  // Split the line as string list
					  String rt1 = rt.next();
					  String rrowList[] = ParseCSV.parseLine(rt1.toString());
					  // Assign values
					  String rtKey = rrowList[1];
					  String rusrVal = rrowList[0];
					  // rate | timestamp
					  String rtVal = rrowList[2]+"|"+rrowList[3];
					  ValPac rtTpl = new ValPac(rusrVal,rtVal);
					  // Save values back
					  r_kv.add(new Tuple2<>(rtKey, rtTpl));
          }
          return r_kv.iterator();
        });

	  JavaPairRDD<String, Iterable<ValPac>> rategrp = ratePart.groupByKey();
    JavaPairRDD<String, ValMTR> rateGrp = rategrp.mapValues(
        rt ->{
            ArrayList<ValPac> rtrnVal = new ArrayList<>();
        		for (ValPac v : rt) rtrnVal.add(v);
        		ValMTR rr = new ValMTR(null, null, rtrnVal);
        		return rr;
          });

		// Join 3 Key-Value pairs here.
    JavaPairRDD<String, ValMTR> joinList1 = moviePart.union(tagGrp);
    JavaPairRDD<String, ValMTR> joinList2 = joinList1.union(rateGrp);
                
    // Fuse Movie, Tag, Rate part: Combine by key 
		JavaPairRDD<String, ArrayList<ValMTR>> mtrCBK = joinList2.combineByKey(
			  grps -> {
            ArrayList<ValMTR> valList = new ArrayList<>();
				    valList.add(grps);
				    return valList;
          },
        (valList, grps) -> {valList.add(grps); return valList;},
        (valList1, valList2) -> {valList1.addAll(valList2); return valList1;}
        );
		JavaPairRDD<String, ArrayList<ValMTR>> mtrMapVal = mtrCBK.mapValues(mtrs -> mtrs);
                
    // Define output column family and column names
		byte[] infoBt = "info".getBytes();
		byte[] mveBt  = "movie".getBytes();
		byte[] mvIDBt = "movieID".getBytes();
		byte[] ttlBt  = "title".getBytes();
		byte[] gnrsBt = "genres".getBytes();

		byte[] tagBt  = "tag".getBytes();

		byte[] rateBt = "rating".getBytes();

		// Try to insert data into HBase row by row 
		JavaPairRDD<ImmutableBytesWritable, Put> changesRDD = mtrMapVal.mapPartitionsToPair(
			rowIn ->{
				ArrayList<Tuple2<ImmutableBytesWritable,Put>> puts = new ArrayList<>();
				while (rowIn.hasNext()) {
					Tuple2<String, ArrayList<ValMTR>> rowKV = rowIn.next();
                    // Get row number
                    String rowID = rowKV._1;
                    Put put = new Put(rowID.getBytes());

                    ArrayList<ValMTR> valFromFst = rowKV._2;

                    for (int i=0; i<valFromFst.size(); i++){
                    	if(valFromFst.get(i).val1!=null & valFromFst.get(i).val2==null & valFromFst.get(i).val3==null){
                    		// Separate values from movie family
                            ValPac findMV = valFromFst.get(i).val1;
                            String mTitle = findMV.val1;
                            String mGenre = findMV.val2;
                            put.addColumn(infoBt, ttlBt, mTitle.getBytes());
                            put.addColumn(infoBt, gnrsBt, mGenre.getBytes());
                        } else if (valFromFst.get(i).val1==null & valFromFst.get(i).val2!=null & valFromFst.get(i).val3==null){
                        	// Separate values from tag family
                        	ArrayList<ValPac> findTG = valFromFst.get(i).val2;
                        	// Insert data by row
                        	for (ValPac tgs : findTG){
                        		String tusrID = tgs.val1;
                        		String tgTxt  = tgs.val2;
                        		put.addColumn(tagBt, tusrID.getBytes(), tgTxt.getBytes());
                        	}
                        } else if (valFromFst.get(i).val1==null & valFromFst.get(i).val2==null & valFromFst.get(i).val3!=null){
                            // Separate values from rating family
                            ArrayList<ValPac> findRT = valFromFst.get(i).val3;
                            // Insert data by row
                            for (ValPac rts : findRT){
                            	String rusrID = rts.val1;
                            	String rtPnt  = rts.val2;
                            	put.addColumn(rateBt, rusrID.getBytes(), rtPnt.getBytes());

                            }
                        }
                    }
                    puts.add(new Tuple2<>(new ImmutableBytesWritable(rowID.getBytes()),put));
                }
                if (puts.isEmpty()){
                  System.out.println("# Empty Insert: " + puts);
                	return puts.iterator();
                } else {
                	System.out.println("# Do Insert: " + puts);
                	return puts.iterator();
                }
            });

                // Create the HBase configuration for input
		Configuration hConf = HBaseConfiguration.create();
		hConf.set(TableInputFormat.INPUT_TABLE, outputTable);
		// Create Hadoop API config to write back to HBase table
		Job hJob =null;
		try {
        hJob = Job.getInstance(hConf);
		}
		catch (IOException e) {
			  System.err.println("ERROR: Job.getInstance exception: "+e.getMessage());
		}
		hJob.getConfiguration().set(TableOutputFormat.OUTPUT_TABLE, outputTable);
		hJob.setOutputFormatClass(TableOutputFormat.class);

		changesRDD.saveAsNewAPIHadoopDataset(hJob.getConfiguration());
	}

	static class ValPac implements Serializable {
      String val1;
      String val2;
      ValPac(String a, String b) {
        	val1 = a;
        	val2 = b;
        }
      } 
  static class ValMTR implements Serializable {
    	ValPac val1;
    	ArrayList<ValPac> val2;
    	ArrayList<ValPac> val3;
    	ValMTR(ValPac a, ArrayList<ValPac> b, ArrayList<ValPac> c) {
    		 val1 = a;
    		 val2 = b;
    		 val3 = c;
        }
      }
}

/*
./run_yarn_100K.sh

/u/home/mikegoss/PDCPublic/compare_tables.sh moviesSample100K "${USER}:movies" | head -50
yarn logs -applicationId application_1581706961583_0239 > logs_100K.tmp
less logs_100K.tmp

grep '^>' logs_100K.tmp
grep '^#' logs_100K.tmp
grep '^>>' logs_100K.tmp

hbase shell
disable "jiasun:movies"
drop "jiasun:movies"
create "jiasun:movies","info","rating","tag"
---------------------------------------------------

./run_yarn_20M.sh

/u/home/mikegoss/PDCPublic/compare_tables.sh moviesSample20M "${USER}:movies20M" | head -50
yarn logs -applicationId application_1581706961583_0261 > logs_20M.tmp
less logs_20M.tmp
grep '^>' logs_20M.tmp
grep '^#' logs_20M.tmp
grep '^>>' logs_20M.tmp

disable "jiasun:movies20M"
drop "jiasun:movies20M"
create "jiasun:movies20M","info","rating","tag"
*/