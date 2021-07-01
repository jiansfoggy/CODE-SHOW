import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.NavigableMap;
import static java.lang.Double.parseDouble;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import com.google.common.primitives.Bytes; 

public class JoinHBase {
	/* Command Line arguments: master - specify Spark master
	 * input - specify input HBase table
	 * output = specify output HBase table
	 */
	public static void main(String[] args) {
		if (args.length != 3) { // check args
			System.err.println("> usage: Spark master inputTable.");
		}
		// Create a Java Spark session
		SparkSession spark = SparkSession.builder()
			.master(args[0]).appName("SparkHBase")
			.config("spark.hadoop.validateOutputSpecs", false)
			.getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

		// Create the HBase configuration for input table
		Configuration hConf = HBaseConfiguration.create();
		String tableName = args[1];
		hConf.set(TableInputFormat.INPUT_TABLE, tableName);

		// Create the HBase configuration for output table
                Configuration oConf = HBaseConfiguration.create();
		String outName = args[2];
		oConf.set(TableInputFormat.INPUT_TABLE, outName);

		// Get an RDD from the HBase table
		JavaPairRDD<ImmutableBytesWritable, Result> tableRDD = 
			sc.newAPIHadoopRDD(hConf, TableInputFormat.class,
							   ImmutableBytesWritable.class,
							   Result.class);
		// Persist the input RDD since we're using multiple operations
		// (count and mapValues)
		tableRDD.persist(StorageLevel.MEMORY_ONLY());

		System.out.println(">> Table size = " + tableRDD.count());

		// Definitions for new column "score:total"
		byte[] plyrFamily = "Players".getBytes();
		byte[] plyrID = "playerID".getBytes();
		byte[] nameFst = "nameFirst".getBytes();
		byte[] nameLst = "nameLast".getBytes();
		byte[] nameCol = "nameFull".getBytes();
		
		byte[] hallFamily = "HallOfFame".getBytes();
		byte[] indtCol = "inducted".getBytes();				
		byte[] yearCol = "yearid".getBytes();
		byte[] cateCol = "category".getBytes();
        
        byte[] hoFamily = "HOF".getBytes();

        // Generate an RDD of changes (Put objects) for the table
        JavaPairRDD<ImmutableBytesWritable, Put> addRDD = tableRDD.flatMapValues(
        	x -> {
        		// r is a row (Result) object to hold inserted values
        		ArrayList<Put> puts = new ArrayList<Put>();
        		Put put = new Put(x.getRow());

        		// Get map of all columns in score family in this row
        		NavigableMap<byte[],byte[]> hallMap = x.getFamilyMap(hallFamily);
        		// Iterate over all columns
        		if(hallMap.containsKey(yearCol) && (new String(hallMap.get(yearCol))!= null)){
        			String yearID = new String(hallMap.get(yearCol));
        			System.out.println("> yearID: "+ yearID);
        			if(hallMap.containsKey(indtCol) && (new String(hallMap.get(indtCol))).equals("Y")){
        				String indtVal = new String(hallMap.get(indtCol));
        				System.out.println("> inducted: "+ indtVal);
        				
        				NavigableMap<byte[],byte[]> plyrMap = x.getFamilyMap(plyrFamily);
        				
						/*
						 * full name part
						 */
						String fName = new String(plyrMap.get(nameFst));
						String lName = new String(plyrMap.get(nameLst));
						ConcatName fl = new ConcatName();
						fl.concatnt(fName, lName);
	        			System.out.println("> full name: "+fl);
	        			put.addColumn(hoFamily, nameCol, fl.toStringBytes());
	        			
	        			/*
						 * yearid part
						 */
	        			put.addColumn(hoFamily, yearCol, yearID.getBytes());
                        
                        /*
						 * category part
						 */
				        String cateVal = new String(hallMap.get(cateCol));
				        System.out.println("> category: "+ cateVal);
				        put.addColumn(hoFamily, cateCol, cateVal.getBytes());

				        /*
						 * save this row to the arraylist
						 */
				        puts.add(put);

				    } else {System.out.println(">> Nothing insert.");}
				} else {System.out.println(">> Nothing insert.");}
				
				if (puts.isEmpty()){
					System.out.println("> Empty Insert: " + puts);
					return puts;
				} else {
					System.out.println("> Do Insert: " + puts);
					return puts;}
        	});
		// Create Hadoop API config to write back to HBase table
		Job hOutputJob = null;
		try {
			hOutputJob = Job.getInstance(hConf);
		}
		catch (IOException e) {
			System.err.println("ERROR: Job.getInstance exception: " + e.getMessage());
			System.exit(2);
		}
		hOutputJob.getConfiguration().set(TableOutputFormat.OUTPUT_TABLE, outName);
		hOutputJob.setOutputFormatClass(TableOutputFormat.class);
		// Write changes to HBase table
		addRDD.saveAsNewAPIHadoopDataset(hOutputJob.getConfiguration());
	}

	static class ConcatName {
		String nameString = " ";
		public void concatnt(String a, String b) {
			nameString = a+" "+b;
		}
		// @Override
		// Return string representation as array of bytes
		public byte[] toStringBytes() {
			return new String(nameString).getBytes();
		};
	}
}

// hbase shell
// scan "jiasun:Lab5"
// disable "jiasun:Lab5"
// drop "jiasun:Lab5"
// create "jiasun:Lab5","HOF"
// ./run_spark_yarn.sh
// /u/home/mikegoss/PDCPublic/compare_tables.sh Lab5SampleOutput "${USER}:Lab5" | head -50
// yarn logs -applicationId application_1581706961583_0062 > logs_short.tmp
// less logs_short.tmp
// grep '^>' logs_short.tmp

