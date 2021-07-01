import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.NavigableMap;
import static java.lang.Double.parseDouble;
import static java.lang.Long.parseLong;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat;
import org.apache.hadoop.mapreduce.Job;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;

public class AnalyzeMoviesG{
	public static void main(String[] args) {
		if (args.length!=3){
                    System.err.println("usage: BuildMovieTable master inputTable outputTable");
		}

        // Create a java spark session and context
		SparkSession spark = SparkSession.builder().master(args[0])
                        .appName("SparkHBase").config("spark.hadoop.validateOutputSpecs", false)
                        .getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
		// Create the HBase configuration for input
		Configuration hConf = HBaseConfiguration.create();
                String inputTable = args[1];
		hConf.set(TableInputFormat.INPUT_TABLE, inputTable);

		// Get an RDD from the HBase table
		JavaPairRDD<ImmutableBytesWritable, Result> tableRDD = 
                        sc.newAPIHadoopRDD(hConf,TableInputFormat.class,
                                ImmutableBytesWritable.class,Result.class);

		tableRDD.persist(StorageLevel.MEMORY_ONLY());
		// Define output column family and column names
		byte[] infoBt = "info".getBytes();
		byte[] gnrsBt = "genres".getBytes();

		byte[] rateBt = "rating".getBytes();

		byte[] tagBt  = "tag".getBytes();
                
        // Deal with tag part and rate part together
        JavaPairRDD<String, ValLst> tgrtRDD = tableRDD.mapPartitionsToPair(
            x ->{
                // Looks like one input row can save as many output row
                ArrayList<Tuple2<String, ValLst>> transit1 = new ArrayList<>();
                while (x.hasNext()) {
                    Tuple2<ImmutableBytesWritable, Result> tagKV = x.next();
                    Result trow = tagKV._2;

                    // Check for info family
                    NavigableMap<byte[], byte[]>infoMap = trow.getFamilyMap(infoBt);
                    
                    // Check for tag family
                    NavigableMap<byte[], byte[]>tagMap  = trow.getFamilyMap(tagBt);

                    // Check for rate family
                    NavigableMap<byte[], byte[]>rateMap = trow.getFamilyMap(rateBt);

                    // Extract values from each info family
                    String in_genre = new String(infoMap.get(gnrsBt));
                    String[] genreLst = in_genre.split("\\|");

                    for(String genre:genreLst){
                        tagMap.forEach(
                            (k,v) -> {
                                String vTags      = new String(v);
                                String[] twoPart1 = vTags.split("\\|");
                                String gnrYear    = genre+"|"+twoPart1[twoPart1.length-1].substring(0, 4);
                                String tagTxt     = twoPart1[0];
                                ValLst forTags    = new ValLst(null,tagTxt,1.0);
                                transit1.add(new Tuple2<>(gnrYear, forTags));
                            });
                        rateMap.forEach(
                            (p,q)->{
                                String vRates     = new String(q);
                                String[] twoPart2 = vRates.split("\\|");
                                String gneYear    = genre+"|"+twoPart2[1].substring(0, 4);
                                Double ratVal     = parseDouble(twoPart2[0]);
                                ValLst forRate    = new ValLst(ratVal,null,1.0);
                                transit1.add(new Tuple2<>(gneYear, forRate));
                            });
                    }
                }
                //System.out.println(">> tgrtRDD: "+transit1);
                return transit1.iterator();
            });

        tgrtRDD.persist(StorageLevel.MEMORY_ONLY());

        // Tag count part
        // tagOut ValLst(null,X,X)
        JavaPairRDD<String, ValLst> tagFlt = tgrtRDD.filter(x -> x._2.val2!=null);
        JavaPairRDD<String, ValLst> tagOut = tagFlt.reduceByKey((x,y)->x.getCnt(y));

        // Rate average part
        // rateOut ValLst(X,null,X)
        JavaPairRDD<String, ValLst> rateFlt = tgrtRDD.filter(x -> x._2.val1!=null);
        JavaPairRDD<String, ValLst> rateOut = rateFlt.reduceByKey((x,y)->x.compAve(y));
        // Extract tag count value
        JavaPairRDD<String, ValPkg> tagFnl = tagOut.mapValues(tg->new ValPkg(null, tg.val3.intValue()));

        // Extract rate aver value
        JavaPairRDD<String, ValPkg> rateFnl = rateOut.mapValues(
            rt->{
                Double rtAve = rt.val1/rt.val3;
                ValPkg backVal = new ValPkg(rtAve, null);
                //System.out.println("# rateFnl1: "+rtAve);
                return backVal;
            });

        JavaPairRDD<String, ValPkg> join_TR = rateFnl.union(tagFnl);
        JavaPairRDD<String, ArrayList<ValPkg>> joinList = join_TR.combineByKey(
            grps -> {
                ArrayList<ValPkg> valList = new ArrayList<>();
                valList.add(grps);
                return valList;
            },
            (valList, grps) -> {valList.add(grps); return valList;},
            (valList1, valList2) -> {valList1.addAll(valList2); return valList1;}
            );

        JavaPairRDD<String, String> output = joinList.mapValues(
            arrlt->{
                ValPkg getVal = new ValPkg(null,null);
                if (arrlt.size()==2){
                    for (int i=0; i<arrlt.size(); i++){
                        if (arrlt.get(i).val1!=null){
                            getVal.val1 = arrlt.get(i).val1;
                            // Float aveGet = arrlt.get(i).val1;
                        } else {
                            getVal.val2 = arrlt.get(i).val2;
                            //Long cntGet = arrlt.get(i).val2;
                        }
                    }
                } else if (arrlt.size()==1){
                    if (arrlt.get(0).val1!=null){
                        getVal.val1 = arrlt.get(0).val1;
                        getVal.val2 = 0;
                        // Long cntGet = new Long(0);
                    } 
                    else {
                        getVal.val1 = new Double(0.0);
                        getVal.val2 = arrlt.get(0).val2;
                    }
                }
                String concat_str = getVal.val1+" "+getVal.val2;
                // String concat_str = aveGet.toString()+" "+cntGet.toString();
                return concat_str;
            });
        tgrtRDD.unpersist();
		// Save the references back to the output file
    	output.sortByKey().saveAsTextFile(args[2]);
	}
    // The class to calculate average and counting number
    static class ValLst implements Serializable {
        Double val1;
        String val2;
        Double val3;
        ValLst(Double a, String b, Double c) {
            val1 = a;
            val2 = b;
            val3 = c;
        }
        ValLst compAve(ValLst vl){
            val1 = val1 + vl.val1;
            val3 = val3 + vl.val3;
            return new ValLst(val1, null, val3);
        }
        ValLst getCnt(ValLst vl){
            val3 = val3 + vl.val3;
            return new ValLst(null, val2, val3);
        }
    }  
    // The class for union function
    static class ValPkg implements Serializable {
        Double val1;
        Integer  val2;
        ValPkg(Double a, Integer b) {
            val1 = a;
            val2 = b;
        }
    } 
}

/*
sort out_yarn_100K/part* | diff -b - /u/home/mikegoss/PDCPublic/data/Prog3SampleOutput/task2G_100K.txt
yarn logs -applicationId application_1581706961583_0343 > logs_100K.tmp
less logs_100K.tmp
grep '^>' logs_100K.tmp
grep '^# tagFnl' logs_100K.tmp
get ../logs_100K.tmp logs_100K.tmp

disable "jiasun:movies"
drop "jiasun:movies"
create "jiasun:movies","info","rating","tag"

sort out_yarn_20M/part* | diff -b - /u/home/mikegoss/PDCPublic/data/Prog3SampleOutput/task2G_20M.txt
yarn logs -applicationId application_1584046847876_0002 > logs_20M.tmp
less logs_20M.tmp
grep '^>' logs_20M.tmp
get ../logs_20M.tmp logs_20M.tmp

disable "jiasun:movies20M"
drop "jiasun:movies20M"
create "jiasun:movies20M","info","rating","tag"
*/