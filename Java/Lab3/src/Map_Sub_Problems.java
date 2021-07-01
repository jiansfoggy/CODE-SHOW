import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;
import java.util.Collections;
import static java.lang.Double.parseDouble;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

class PtXY implements Serializable {
    public String pnt_name;
    public Double x;
    public Double y;
    public String set_PtXY(String pnt_name, Double x, Double y){
    	this.pnt_name = pnt_name;
    	this.x = x;
    	this.y = y;
        String concat_them = this.pnt_name+" "+this.x+" "+this.y;
    	return concat_them.toString();
    }    
}

public class Map_Sub_Problems {
	/* 
  	 *  Specify four arguments:
  	 *    master - specify Spark master ("local" or "yarn")
  	 *    input - specify input file
	 *    output - specify output file
	 *    cellSize - specify a real number specifying the cell size (must be > 0).
    */   
	public static Integer cal_cell (Double x, Double cellSize){
		return (int)Math.floor(x / cellSize);
	}

	public static void main(String[] args) throws Exception {
    // Check arguments
		if (args.length != 4) {
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
                    /*
                    String cellID = String.valueOf(cell_x)+":"+String.valueOf(cell_y);
                    String cellID_r = String.valueOf(cell_x+1)+":"+String.valueOf(cell_y);
                    String cellID_dl = String.valueOf(cell_x-1)+":"+String.valueOf(cell_y+1);
                    String cellID_d = String.valueOf(cell_x)+":"+String.valueOf(cell_y+1);
                    String cellID_dr = String.valueOf(cell_x+1)+":"+String.valueOf(cell_y+1);
                    */
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

    	JavaPairRDD<String,Iterable<String>> pKV = point_KV.groupByKey();

    	JavaPairRDD<String,Iterable<String>> pntSorted = pKV.mapValues(
    		grps -> {
                    ArrayList<String> valList = new ArrayList<>();
                    for (String s : grps) valList.add(s);

                    Collections.sort(valList);
                    return valList;
        	});

    	JavaPairRDD<String,Iterable<String>> sorted_KV = pntSorted.sortByKey();

    	// Save the references back to the output file
    	sorted_KV.saveAsTextFile(args[2]);

    }
}

// cat /u/home/jiasun/COMP4333/Lab3/out_yarn_s1/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab3Output/out_short_1.txt | head -50
// cat /u/home/jiasun/COMP4333/Lab3/out_yarn_f1/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab3Output/out_full_1.txt | head -50
// cat /u/home/jiasun/COMP4333/Lab3/out_yarn_f3/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab3Output/out_full_3.txt | head -50


