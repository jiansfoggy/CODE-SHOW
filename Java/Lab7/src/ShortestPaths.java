import java.io.Serializable;
import java.util.ArrayList;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaRDDLike;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.EdgeDirection;
import org.apache.spark.graphx.EdgeTriplet;
import org.apache.spark.graphx.VertexRDD;
import org.apache.spark.graphx.Graph;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.StatCounter;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;
import scala.collection.Iterator;
import scala.collection.JavaConverters;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.AbstractFunction3;

public class ShortestPaths {
	public static void main(String[] args) {
		// Check arguments
		if (args.length != 4) {
      	System.err.println("usage: ShortestPaths master input output source");
      	System.exit(1);
    	}
    	// Create a Java Spark session and context
    	SparkSession spark = SparkSession.builder().master(args[0])
    								 	.appName("ShortestPaths").getOrCreate();
    	JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        // Load Our input data.
    	JavaRDD<String> input = sc.textFile(args[1]);
        // For bullet symbol 1, 2
        final LongAccumulator edgeCnt = sc.sc().longAccumulator("EdgeNumberCount");
        final LongAccumulator msgCnt  = sc.sc().longAccumulator("MsgNumberCount");
        // For bullet symbol 3
        long sourceVertex = Long.parseLong(args[3]);
        final Broadcast<Long> sV = sc.broadcast(sourceVertex);

    	// Decode each input line into an edge, building an RDD of the edges
    	JavaRDD<Edge<Long>> weightedEdges = input.map(new ParseEdge(edgeCnt));
    	// Build the graph from the edges. Initialize all vertices to the maximum
    	// possible value.
    	Graph graph = Graph.fromEdges(
    		weightedEdges.rdd(),          // source edges
    		Long.MAX_VALUE,				  // default vertex value
    		StorageLevel.MEMORY_ONLY(),	  // edge level
    		StorageLevel.MEMORY_ONLY(),   // vertex level
    		scala.reflect.ClassTag$.MODULE$.apply(Long.class),   // edge ClassTag
    		scala.reflect.ClassTag$.MODULE$.apply(Long.class));  // vertex ClassTag
    	graph.persist(StorageLevel.MEMORY_ONLY());

    	// Run iterative Pregel computation to find 
    	Graph shortestDistances = graph.ops().pregel(
            	Long.MAX_VALUE,             // initial message
            	Integer.MAX_VALUE,          // maximum number of iterations
            	EdgeDirection.Out(),        // which vertices must be active for send
            	new ReceiveMessage(sV.value()),  // process received message
            	new SendMessages(msgCnt),         // used to send message
            	new MergeMessages(),        // used to merge messages to a vertex
            	scala.reflect.ClassTag$.MODULE$.apply(Long.class)
    	);
    	shortestDistances.persist(StorageLevel.MEMORY_ONLY());
        Long edgeCount = shortestDistances.ops().numEdges();

        // Write output
        // map shortestDistances.vertices() out and convert them
        shortestDistances.vertices().saveAsTextFile(args[2]);
        // For bullet symbol 4
        JavaRDD<Tuple2<Object, Long>> verTpl  = shortestDistances.vertices().toJavaRDD();
        JavaDoubleRDD stat = verTpl.mapPartitionsToDouble(
            x->{
                ArrayList<Double> shrtDist = new ArrayList<>();
                while(x.hasNext()){
                    Tuple2<Object, Long> obLong = x.next();
                    shrtDist.add(new Double(obLong._2));
                    
                }
                return shrtDist.iterator();
            });
        //JavaDoubleRDD stat = verTpl.mapToDouble(x-> new Double(x._2));
    	StatCounter statCnt = stat.stats();
       
    	// For bullet symbol 5
    	System.out.println("> edgeCount = " + edgeCnt.value());
    	System.out.println("> edgeCount = " + edgeCount);
    	System.out.println("> messageCount = " + msgCnt.value());
    	System.out.println("> count() = " + statCnt.count());
    	System.out.println("> mean() = " + statCnt.mean());
    	System.out.println("> sum() = " + statCnt.sum());
    	System.out.println("> max() = " + statCnt.max());
    	System.out.println("> min() = " + statCnt.min());
    	System.out.println("> variance() = " + statCnt.variance());
    	System.out.println("> sampleVariance() = " + statCnt.sampleVariance());
    	System.out.println("> stdev() = " + statCnt.stdev());
    	System.out.println("> sampleStdev() = " + statCnt.sampleStdev());
	}

	static class ParseEdge implements Function<String, Edge<Long>> {
        LongAccumulator edgeCnt;
        ParseEdge(LongAccumulator eC) {
            edgeCnt = eC;
        }
		@Override
		public Edge<Long> call(String textLine) {
			String[] e = textLine.split(",");
			if (e.length!=3) {
				System.err.println("ERROR: malformed edge input: \""+textLine+"\"");
				System.exit(2);
			}
            edgeCnt.add(1);
			return new Edge<>(Long.valueOf(e[0]), Long.valueOf(e[1]), Long.valueOf(e[2]));
		}	
	}

	static class ReceiveMessage extends AbstractFunction3<Long, Long, Long, Long>
		implements Serializable {
			long sourceVertex;
			// Constructor caches source vertex
			ReceiveMessage(long sourceVertex_) {
				sourceVertex = sourceVertex_;
			}

			@Override
			public Long apply(Long vertexID, Long vertexData, Long message) {
				Long newData = vertexData; // default is unchanged value

				// If this vertex is the source vertex and this is the initial
				// message, set distance to 0
				if (vertexID == sourceVertex && vertexData != 0L) {
					newData = 0L;
				}
				// If incoming distance is < our current distance, update to
				// smaller value
				else if (message<vertexData) {
					newData=message; // modified value
				}
				// System.out.println("> Vertex "+vertexID+" received msg "+message
				// + ", new value is "+newData.toString());
				return newData;
			}
		};

	static class MergeMessages extends AbstractFunction2<Long,Long,Long>
		implements Serializable {
			@Override
			public Long apply(Long msg1, Long msg2) {
				// We just need the smaller of the messages (distance)
				return (msg1<=msg2)?msg1:msg2;
			}
		}

	static class SendMessages extends AbstractFunction1<EdgeTriplet<Long,Long>,
		Iterator<Tuple2<Long,Long>>> implements Serializable {
            LongAccumulator msgCnt;
            SendMessages(LongAccumulator mC) {
                msgCnt  = mC;
            }
			@Override
			public Iterator<Tuple2<Long,Long>> apply(EdgeTriplet<Long,Long> edge) {
				// Create array list for output messages
				ArrayList<Tuple2<Long,Long>> msgs = new ArrayList<>();
				// If distance from source to dest is shorter than existing dest value
				if (edge.srcAttr()!=Long.MAX_VALUE) {
					long distance = edge.srcAttr() + edge.attr();
					if (distance < edge.dstAttr()) {
						// Send new distance to destination
						msgs.add(new Tuple2<>(edge.dstId(), distance));
						msgCnt.add(1);
					}
				}
				// Return iterator over messages to send (must be a Scala iterator)
				return JavaConverters.asScalaIteratorConverter(msgs.iterator()).asScala();
			}
		}
	}

/*

compile:
    [javac] Compiling 1 source file to /u/home9/jiasun/COMP4333/Lab7/build
    [javac] /u/home9/jiasun/COMP4333/Lab7/src/ShortestPaths.java:60: warning: [unchecked] unchecked call to <A>pregel(A,int,EdgeDirection,Function3<Object,VD,A,VD>,Function1<EdgeTriplet<VD,ED>,Iterator<Tuple2<Object,A>>>,Function2<A,A,A>,ClassTag<A>) as a member of the raw type GraphOps
    [javac]     	Graph shortestDistances = graph.ops().pregel(
    [javac]     	                                            ^
    [javac]   where A,VD,ED are type-variables:
    [javac]     A extends Object declared in method <A>pregel(A,int,EdgeDirection,Function3<Object,VD,A,VD>,Function1<EdgeTriplet<VD,ED>,Iterator<Tuple2<Object,A>>>,Function2<A,A,A>,ClassTag<A>)
    [javac]     VD extends Object declared in class GraphOps
    [javac]     ED extends Object declared in class GraphOps
    [javac] /u/home9/jiasun/COMP4333/Lab7/src/ShortestPaths.java:77: warning: [unchecked] unchecked conversion
    [javac]     	JavaRDD<Tuple2<Object, Long>> verTpl = shortestDistances.vertices().toJavaRDD();
    [javac]     	                                                                             ^
    [javac]   required: JavaRDD<Tuple2<Object,Long>>
    [javac]   found:    JavaRDD
    [javac] 2 warnings

chmod u+x run_yarn_short_1.sh
./run_yarn_short_0.sh
sort out_yarn_short/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/short_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0220 > logs_short1.tmp
grep '^>' logs_short1.tmp
less logs_short.tmp

> edgeCount = 12
> edgeCount = 12
> messageCount = 5
> count() = 5
> mean() = 1.8
> sum() = 9.0
> max() = 5.0
> min() = 0.0
> variance() = 2.96
> sampleVariance() = 3.6999999999999997
> stdev() = 1.7204650534085253
> sampleStdev() = 1.9235384061671343

chmod u+x run_yarn_short_4.sh
./run_yarn_short_4.sh
sort out_yarn_short/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/short_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0219 > logs_short4.tmp
grep '^>' logs_short4.tmp
less logs_short.tmp

> edgeCount = 12
> edgeCount = 12
> messageCount = 6
> count() = 5
> mean() = 3.8
> sum() = 19.0
> max() = 6.0
> min() = 0.0
> variance() = 4.16
> sampleVariance() = 5.2
> stdev() = 2.039607805437114
> sampleStdev() = 2.280350850198276

./run_yarn_full_1.sh
sort out_yarn_full/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/full_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0216 > logs_full1.tmp
grep '^>' logs_full1.tmp
less logs_full.tmp

> edgeCount = 1000000
> edgeCount = 1000000
> messageCount = 854931
> count() = 80000
> mean() = 889.0866499999996
> sum() = 7.112693199999997E7
> max() = 1802.0
> min() = 0.0
> variance() = 21253.769466777507
> sampleVariance() = 21254.035142216784
> stdev() = 145.78672596219968
> sampleStdev() = 145.78763713777923

./run_yarn_full_80000.sh
sort out_yarn_full/part* | diff -w - /u/home/mikegoss/PDCPublic/data/Prog2SampleOutput/full_RDD_sorted.txt | head -20
yarn logs -applicationId application_1581706961583_0217 > logs_full80000.tmp
grep '^>' logs_full80000.tmp
less logs_full.tmp

> edgeCount = 1000000
> edgeCount = 1000000
> messageCount = 728674
> count() = 80000
> mean() = 951.8397749999986
> sum() = 7.614718199999988E7
> max() = 1985.0
> min() = 0.0
> variance() = 21166.864527949358
> sampleVariance() = 21167.12911706332
> stdev() = 145.48836561027605
> sampleStdev() = 145.48927492108592
*/