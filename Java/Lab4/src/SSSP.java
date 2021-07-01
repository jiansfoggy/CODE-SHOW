import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;
import java.util.Collections;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.graphx.EdgeDirection;
import org.apache.spark.graphx.EdgeTriplet;
import org.apache.spark.graphx.Graph;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.EdgeTriplet;
import org.apache.spark.graphx.GraphLoader;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.graphx.util.GraphGenerators;

import scala.Tuple2;
import scala.collection.Iterator;
import scala.collection.JavaConverters;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.AbstractFunction3;

public class SSSP{
    public static void main(String[] args){
        if (args.length != 4){
            System.err.println("usage: ConnectedComponets master input output");           
        }
        // Create a Java Spark Context
        SparkConf conf = new SparkConf().setMaster(args[0]).setAppName("Lab4");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load a graph from input
        JavaRDD<String> sc1 = sc.textFile(args[1]);
        JavaRDD<Edge<Long>> input = sc1.map(
            x -> {
                String fields[] = x.split(",");
                return new Edge(Long.parseLong(fields[0]), Long.parseLong(fields[1]),Long.parseLong(fields[2]));
          
            });

        Graph justEdges = Graph.fromEdges(input.rdd(), Long.MAX_VALUE,
            StorageLevel.MEMORY_ONLY(), StorageLevel.MEMORY_ONLY(),
            scala.reflect.ClassTag$.MODULE$.apply(Long.class),
            scala.reflect.ClassTag$.MODULE$.apply(Long.class));
        
        // Initialize all vertex data
        Long sourceID = Long.parseLong(args[3]);
        Graph graph = justEdges.mapVertices(
            new VertexInit(sourceID),
            scala.reflect.ClassTag$.MODULE$.apply(Long.class),
            null);

        graph.persist(StorageLevel.MEMORY_ONLY());

        // Find the sssp
        System.out.println("> Running Pregel with " + graph.edges().count() 
            + " edges and " + graph.vertices().count() + " vertices");

        int vertexNum = (int)graph.vertices().count();
        System.out.println("> Vertex Number is" + " " + vertexNum);
        Graph dijkstra = graph.ops().pregel(
            Long.parseLong(5000000),    // initial message
            30,                         // maximum number of iterations
            EdgeDirection.Either(),     // which vertices must be active for send
            new ReceiveMessages(),      // process received message
            new SendMessages(),         // used to send message
            new MergeMessages(),        // used to merge messages to a vertex
            scala.reflect.ClassTag$.MODULE$.apply(Long.class)
        );
        System.out.println("> Pregel complete.");
        dijkstra.vertices().saveAsTextFile(args[2]);

        System.out.println("> Done.");
    }

    static class VertexInit extends AbstractFunction2<Long, Long, Long>
        implements Serializable {
            long sourceID;
            VertexInit (long sID) {sourceID = sID;}
            @Override
            public Long apply(Long vertexID, Long oldData) {
                // Long.parseLong(args[3])
                if (vertexID == sourceID) {
                  return new Long(0);
                } else {
                  return new Long(Long.MAX_VALUE);
                }
              }
            }

    static class ReceiveMessages extends AbstractFunction3<Long,Long,Long,Long> 
          implements Serializable {
              @Override
              public Long apply(Long vertexID, Long vertexData, Long message) {
                  Long newData = vertexData;      // default is unchaged value
      
                  // If incoming component ID is < our current component ID, update to
                  // smaller value (ignore initial message of -1)
                  if (message >= 0 && message < vertexData) {
                      newData = new Long(message);  // modified value
                    }
                  System.out.println("> Vertex " + vertexID + " received msg " + message 
                      + ", new value is " + newData.toString());
                  return newData;
                }
            }

    static class MergeMessages extends AbstractFunction2<Long,Long,Long> 
          implements Serializable  {
              @Override
              public Long apply(Long msg1, Long msg2) {
                  // We just need the smaller of the messages (component IDs)
                  return (msg1 <= msg2) ? msg1 : msg2;
                }
            }

    static class SendMessages 
          extends AbstractFunction1<EdgeTriplet<Long,Long>,Iterator<Tuple2<Long,Long>>>
          implements Serializable {
              @Override
              public Iterator<Tuple2<Long,Long>> apply(EdgeTriplet<Long,Long> edge) {
                  // Create array list for output messages
                  ArrayList<Tuple2<Long,Long>> msgs = new ArrayList<>();
                  Long srcData = edge.srcAttr();
                  Long dstData = edge.dstAttr();
                  
                  System.out.println("> SendMessages called with "
                      + edge.srcId() + "(" + srcData + ") and " 
                      + edge.dstId() + "(" + dstData + ")");
      
                  // If component of source vertex is smaller than component of destination
                  if (srcData + edge.attr() <= dstData) {
                      // Send source component to destination
                      Long msgsout = srcData + edge.attr();
                      msgs.add(new Tuple2<Long,Long>(edge.dstId(), msgsout));
                      System.out.println("> Send fwd message " + edge.srcId() + "->" 
                          + edge.dstId() + " (" + msgsout.toString() + ")");
                      } 
                      
                  // Return iterator over messages to send (must be a Scala iterator)
                  return JavaConverters.asScalaIteratorConverter(msgs.iterator()).asScala();
              }
          }
}


// sort out_yarn_s0/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab4Output/Lab4SampleOutputShort0.txt
// yarn logs -applicationId application_1581706961583_0002 > logs_short0.tmp
// grep '^>' logs_short0.tmp
// sort out_yarn_s4/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab4Output/Lab4SampleOutputShort4.txt
// yarn logs -applicationId application_1581220586827_0115 > logs_short4.tmp
// grep '^>' logs_short4.tmp
// sort out_yarn_f1/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab4Output/Lab4SampleOutputFull1.txt | head -50
// yarn logs -applicationId application_1581706961583_0010 > logs_full1.tmp
// grep '^>' logs_full1.tmp
// sort out_yarn_f80000/part* | diff -b - /u/home/mikegoss/PDCPublic/Labs/Lab4Output/Lab4SampleOutputFull80000.txt | head -50
// yarn logs -applicationId application_1581706961583_0015 > logs_full80000.tmp
// yarn logs -applicationId application_1581706961583_0190 > logs_full80000.tmp
// grep '^>' logs_full80000.tmp | head -50

