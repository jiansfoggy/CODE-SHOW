import java.util.ArrayList;
import java.io.Serializable;
import java.util.Collections;
import static java.lang.Double.parseDouble;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.StorageLevels;

import scala.Tuple2;

// class for concatinate point and its location
class PtXY implements Serializable {
    public String pnt_name;
    public Double x;
    public Double y;
    public String set_PtXY(String pnt_name, Double x, Double y){
    	this.pnt_name = pnt_name;
    	this.x = x;
    	this.y = y;
        String concat_them = this.pnt_name+" "+this.x+" "+this.y;
        // think about create tuplet
    	return concat_them;
    }
}

// location pair class
class LtPair implements Serializable {
    public Double x1;
    public Double y1;
    public Double x2;
    public Double y2;
    public ArrayList<Double> set_LPair(String[] loc1, String[] loc2){
        this.x1 = parseDouble(loc1[1]);
        this.y1 = parseDouble(loc1[2]);
        this.x2 = parseDouble(loc2[1]);
        this.y2 = parseDouble(loc2[2]); 
        ArrayList<Double> dl = new ArrayList<>();
        dl.add(this.x1); 
        dl.add(this.y1);
        dl.add(this.x2);
        dl.add(this.y2);
        // Double[] list = Arrays.asList(this.x1,this.y1,this.x1,this.y1);
        return dl;
    }
}

class MaxMinAve implements Serializable {
    public Double sum;
    public Double cnt;   
    public Double maxVal;
    public Double minVal;
    
    MaxMinAve (Double sum, Double cnt, Double maxVal, Double minVal) {
        this.sum = sum;
        this.cnt = cnt;
        this.maxVal = maxVal;
        this.minVal = minVal;
    }
    public MaxMinAve compVal(MaxMinAve mma){
        this.cnt = this.cnt + mma.cnt;
        this.sum = this.sum + mma.sum;
        this.minVal = (this.minVal <= mma.minVal) ? this.minVal : mma.minVal;
        this.maxVal = (mma.maxVal <= this.maxVal) ? this.maxVal : mma.maxVal;
        return new MaxMinAve(this.sum, this.cnt, this.maxVal, this.minVal);
    }
}




public class Split_Problems {
	/* 
  	 *  Specify four arguments:
  	 *    master - specify Spark master ("local" or "yarn")
  	 *    input - specify input file
	 *    output - specify output file
	 *    cellSize - specify a real number specifying the cell size (must be > 0).
     *    maxDist - 
    */   
	public static Integer cal_cell (Double x, Double cellSize){
		return (int)Math.floor(x / cellSize);
	}

    public static Double cal_dist (String[] loc1, String[] loc2){
        Double x1 = parseDouble(loc1[1]);
        Double y1 = parseDouble(loc1[2]);
        Double x2 = parseDouble(loc2[1]);
        Double y2 = parseDouble(loc2[2]); 
        return Math.abs(Math.sqrt(Math.pow(x1-x2,2)+Math.pow(y1-y2,2)));
    }

    public static void main(String[] args) throws Exception {
    // Check arguments
		if (args.length != 5) {
      		System.err.println("usage: Map_Sub_Problems master input output cellSize");
      		System.exit(1);
        }

    	// Create a Java Spark session and context
    	SparkSession spark = SparkSession
    		.builder().master(args[0])
      		.appName("Map_Sub_Problems").getOrCreate();
    	JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
    	// Load our input data.
    	JavaRDD<String> input = sc.textFile(args[1]);

    	// Split up into URLs, delimited by white space. Output key/value pairs
    	JavaPairRDD<String, String> point_KV = input.flatMapToPair(
    		x -> {
                    String p_ls[] = x.split("\\,");
                    // Create empty list of URL pairs for output
                    ArrayList<Tuple2<String,String>> p_kv = new ArrayList<>();
                    // Split input into an array of URLs (split on whitespace)             

                    int cell_x = cal_cell(parseDouble(p_ls[1]), 
                        parseDouble(args[3]));
                    int cell_y = cal_cell(parseDouble(p_ls[2]), 
                        parseDouble(args[3]));
                    
                    String cellID = cell_x+":"+cell_y;
                    String cellID_r = (cell_x+1)+":"+cell_y;
                    String cellID_dl = (cell_x-1)+":"+(cell_y+1);
                    String cellID_d = cell_x+":"+(cell_y+1);
                    String cellID_dr = (cell_x+1)+":"+(cell_y+1);         

                    PtXY crt_pnt_val = new PtXY();
                    String pnt_val = crt_pnt_val.set_PtXY(p_ls[0],
                            parseDouble(p_ls[1]),
                            parseDouble(p_ls[2]));
                
                    p_kv.add(new Tuple2<>(cellID,    pnt_val));
                    p_kv.add(new Tuple2<>(cellID_r,  pnt_val));
                    p_kv.add(new Tuple2<>(cellID_dl, pnt_val));
                    p_kv.add(new Tuple2<>(cellID_d,  pnt_val));
                    p_kv.add(new Tuple2<>(cellID_dr, pnt_val));
                    return p_kv.iterator();
            });

    	JavaPairRDD<String,ArrayList<String>> pKV = point_KV.combineByKey(
            grps -> {
                ArrayList<String> valList = new ArrayList<>();
                valList.add(grps);
                return valList;
            },
            (valList, grps) -> {valList.add(grps); return valList;},
            (valList1, valList2) -> {valList1.addAll(valList2); return valList1;}
            );
        
        // point pair output part
    	JavaPairRDD<String,ArrayList<Tuple2<String, String>>> pntSorted = pKV.mapValues(
            grps -> {
                Collections.sort(grps);
                ArrayList<Tuple2<String, String>> corList = new ArrayList<>();
                if (grps.size() > 1) { 
                    for(int i=0; i<grps.size()-1; i++){
                        String a1 = grps.get(i);
                        String[] pntxy_array1 = a1.split("\\s+");
                        for(int j=i+1; j<grps.size(); j++){
                            String a2 = grps.get(j);
                            String[] pntxy_array2 = a2.split("\\s+");
                            Double pnt_dist = cal_dist(pntxy_array1, pntxy_array2);
                            if (pnt_dist <= parseDouble(args[4])) {
                                corList.add(new Tuple2<>(pntxy_array1[0],pntxy_array2[0]));
                                //dstnList.add(pnt_dist);
                            }
                        }                       
                    } 
                }
                return corList;
            });
        pntSorted.persist(StorageLevels.MEMORY_AND_DISK_SER);

        JavaPairRDD<String, String> pntPair = pntSorted.flatMapToPair(
            xs -> {
                ArrayList<Tuple2<String,String>> valPart = xs._2;
                return valPart.iterator();
            });
        JavaPairRDD<String, String> pntDstnct = pntPair.distinct();
    	// Save the references back to the output file
    	pntDstnct.sortByKey().saveAsTextFile(args[2]);

        // statistics calculation part
        JavaPairRDD<String, ArrayList<MaxMinAve>> distSorted = pKV.mapValues(
            grps -> {
                Collections.sort(grps);
                Double input_dsnt = parseDouble(args[4]);
                ArrayList<MaxMinAve> dstnList = new ArrayList<>();
                    
                if (grps.size() > 1) { 
                    for(int i=0; i<grps.size()-1; i++){
                        String a1 = grps.get(i);
                        String[] pntxy_array1 = a1.split("\\s+");
                        for(int j=i+1; j<grps.size(); j++){
                            String a2 = grps.get(j);
                            String[] pntxy_array2 = a2.split("\\s+");
                            Double pnt_dist = cal_dist(pntxy_array1, pntxy_array2);
                            if (pnt_dist <= input_dsnt) {
                                MaxMinAve saveVal = new MaxMinAve(pnt_dist,1.0,pnt_dist,pnt_dist);
                                dstnList.add(saveVal);
                            }
                        }                       
                    } 
                }
                return dstnList;
            });

        JavaRDD<ArrayList<MaxMinAve>> dstnArray = distSorted.map(x->x._2);
        JavaPairRDD<Long, MaxMinAve> strt_stat = dstnArray.flatMapToPair( 
            grps-> {
                ArrayList<Tuple2<Long, MaxMinAve>> transit = new ArrayList<>();
                grps.forEach((s) -> {
                    transit.add(new Tuple2<>(1L,s));
                    });
                return transit.iterator();
            });
        JavaPairRDD<Long, MaxMinAve> comb_stat = strt_stat.reduceByKey((x,y) -> x.compVal(y));
        MaxMinAve extrt_stat = comb_stat.map(x->x._2).first();

        System.out.println("@ ------------------------------------------");
        System.out.println("@ The answer from my method.");
        System.out.println("@ Average distance = " + extrt_stat.sum/extrt_stat.cnt);
        System.out.println("@ Minimum distance = " + extrt_stat.minVal);
        System.out.println("@ Maximum distance = " + extrt_stat.maxVal);
        System.out.println("@ Calculated Them!");
        System.out.println("@ ------------------------------------------");

        JavaPairRDD<String, ArrayList<Tuple2<ArrayList<Double>,Double>>> distSorted1 = pKV.mapValues(
            grps -> {
                Collections.sort(grps);
                Double input_dsnt = parseDouble(args[4]);
                ArrayList<Tuple2<ArrayList<Double>,Double>> dstnList1 = new ArrayList<>();
                LtPair location_pair = new LtPair();
                    
                if (grps.size() > 1) { 
                    for(int i=0; i<grps.size()-1; i++){
                        String a1 = grps.get(i);
                        String[] pntxy_array1 = a1.split("\\s+");
                        for(int j=i+1; j<grps.size(); j++){
                            String a2 = grps.get(j);
                            String[] pntxy_array2 = a2.split("\\s+");
                            Double pnt_dist = cal_dist(pntxy_array1, pntxy_array2);
                            if (pnt_dist <= input_dsnt) {
                                ArrayList<Double> pnt_val = location_pair.set_LPair(pntxy_array1, pntxy_array2);
                                dstnList1.add(new Tuple2<>(pnt_val, pnt_dist));
                            }
                        }                       
                    } 
                }
                return dstnList1;
            });

        JavaPairRDD<ArrayList<Double>,Double> dstnArray1 = distSorted1.flatMapToPair(
            xs -> {
                ArrayList<Tuple2<ArrayList<Double>,Double>> valSide = xs._2;
                return valSide.iterator();
            });
        JavaPairRDD<ArrayList<Double>,Double> dstnDstnct1 = dstnArray1.distinct();
        JavaRDD<Double> extrt_dist1 = dstnDstnct1.map(x -> x._2);
        JavaDoubleRDD start_calcu1 = extrt_dist1.mapToDouble(x->(double)x);

        System.out.println("@ ------------------------------------------");
        System.out.println("@ The answer from JavaDoubleRDD.");
        System.out.println("@ Average distance = " + start_calcu1.mean().toString());
        System.out.println("@ Minimum distance = " + start_calcu1.min().toString());
        System.out.println("@ Maximum distance = " + start_calcu1.max().toString());
        System.out.println("@ Calculated Them!");
        System.out.println("@ ------------------------------------------");
    }
}


// sort /u/home/jiasun/COMP4333/Programming_Assignment_1/out_yarn_s/part* | diff - /u/home/mikegoss/PDCPublic/Programs/Program1/ShortSampleOutput.txt
// yarn logs -applicationId application_1581220586827_0098 > logs_short.tmp
// grep '^@' logs_short.tmp
// @ Average distance = 0.6260771098373045
// @ Minimum distance = 0.20622678483230272
// @ Maximum distance = 0.9086713049832704
// sort /u/home/jiasun/COMP4333/Programming_Assignment_1/out_yarn_f/part* | zdiff - /u/home/mikegoss/PDCPublic/Programs/Program1/FullSampleOutput.txt.gz
// yarn logs -applicationId application_1581220586827_0099 > logs_full.tmp
// grep '^@' logs_full.tmp
// @ Average distance = 1.0662638596588518
// @ Minimum distance = 0.0
// @ Maximum distance = 1.6
// sort /u/home/jiasun/COMP4333/Programming_Assignment_1/out_yarn_l/part* | zdiff - /u/home/mikegoss/PDCPublic/Programs/Program1/LargeSampleOutput.txt.gz
// yarn logs -applicationId application_1581220586827_0101 > logs_large.tmp
// grep '^@' logs_large.tmp
// @ Average distance = 1.000213583642862
// @ Minimum distance = 6.354044127512686E-4
// @ Maximum distance = 1.4999998900967746
