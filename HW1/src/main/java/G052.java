import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.sources.In;
import org.apache.spark.storage.StorageLevel;
import scala.Array;
import scala.Int;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G052 {

    public static void main(String[] args) throws IOException {

        // HOMEWORK 1

        // Checking the correct number of parameters
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: K H S dataset_path");
        }

        // Setting up Spark
        SparkConf conf = new SparkConf(true).setAppName("G052");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        /**
         * Reading input
         */

        // number of partitions
        int K = Integer.parseInt(args[0]);

        // reading H
        int H = Integer.parseInt(args[1]);

        // reading country
        String S = args[2];

        // read dataset and subdivide it into K random partitions
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // setting global variables
        long numTransaction = rawData.count();
        System.out.println("Number of transaction: " + numTransaction);


        // RDD <P,C>
        JavaPairRDD<String, Integer> productCustomer;


        productCustomer = rawData
                .flatMapToPair((transaction) -> { // <- Map Phase R1

                    String[] tokens = transaction.split(",");

                    // parsing tokens (if you want, delete useless vars e.g. description, date and time, etc)
                    String transactionID = tokens[0];
                    String productID = tokens[1];
                    String description = tokens[2];
                    int quantity = Integer.parseInt(tokens[3]);
                    String invoiceDate = tokens[4];
                    double unitPrice = Double.parseDouble(tokens[5]);
                    int customerID = Integer.parseInt(tokens[6]);
                    String country = tokens[7];

                    // created since we need to return an iterator
                    ArrayList<Tuple2<Tuple2<String, Integer>, Integer>> pairList = new ArrayList<>();

                    // filtering the transaction as specified
                    if (quantity > 0 && (country.equalsIgnoreCase(S)) || (S).equalsIgnoreCase("all")) {

                        // 0 = fictitious value, useless to the program, will be lost at next phase (i.e. R1 reduce phase)
                        pairList.add(new Tuple2<>(new Tuple2<>(productID, customerID), 0));

                    }

                    return pairList.iterator();
                })
                .groupByKey() // <- Shuffling and grouping
                .flatMapToPair((element) -> { // <- Reduce phase R1

                    // list where to add each pair <Product, Customer>
                    ArrayList<Tuple2<String, Integer>> productCustomersPairs = new ArrayList<>();
                    productCustomersPairs.add(element._1());


                    return productCustomersPairs.iterator();

                }); // <- here start map phase round 2 (Points 3 - 6)

        // Debugging prints
        System.out.println("Number of transaction after R1: " + productCustomer.count());

        //RDD <P,Popularity>
        JavaPairRDD<String, Integer> productPopularity1;

        //Point 3 - mapPartitionsToPair
        productPopularity1 = productCustomer
                .mapPartitionsToPair((element) -> { // <- reduce phase R2

                    //hashMap to count the number of customer for each product
                    HashMap<String, Integer> counts = new HashMap<>();
                    while(element.hasNext()){
                        Tuple2<String,Integer> tuple = element.next();
                        counts.put(tuple._1(), 1 + counts.getOrDefault(tuple._1(), 0));
                    }

                    //ArrayList to store the new pairs (productId, popularity)
                    ArrayList<Tuple2<String, Integer>> productPopularityPairs = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : counts.entrySet()){
                        productPopularityPairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return productPopularityPairs.iterator();
                })
                .reduceByKey((x,y) -> x+y); // <- GroupByKey plus mapValues


        //Point 5 - extraction of the top H products
        if(H > 0){
            List<Tuple2<String, Integer>> topH = productPopularity1
                    //swap (key, value) to (value, key)
                    .mapToPair((element) -> element.swap())

                    //sort by descending order
                    .sortByKey(false)

                    //swap the previous (value, key) to (key, value) again
                    .mapToPair((element) -> element.swap())

                    //takes the first H elements
                    .take(H);
            System.out.println(topH);
        }

        //Point 6 - debug
        if(H == 0){
            List<Tuple2<String, Integer>> sorted = productPopularity1.sortByKey(true).collect();
            System.out.println(sorted);
        }

        /** WORDCOUNT CODE (to be deleted)

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // CHECKING NUMBER OF CMD LINE PARAMETERS
         // Parameters are: num_partitions, <path_to_file>
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         if (args.length != 2) {
         throw new IllegalArgumentException("USAGE: num_partitions file_path");
         }

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // SPARK SETUP
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         SparkConf conf = new SparkConf(true).setAppName("WordCount");
         JavaSparkContext sc = new JavaSparkContext(conf);
         sc.setLogLevel("WARN");

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // INPUT READING
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         // Read number of partitions
         int K = Integer.parseInt(args[0]);

         // Read input file and subdivide it into K random partitions
         JavaRDD<String> docs = sc.textFile(args[1]).repartition(K).cache();

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // SETTING GLOBAL VARIABLES
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         long numdocs, numwords;
         numdocs = docs.count();
         System.out.println("Number of documents = " + numdocs);
         JavaPairRDD<String, Long> wordCounts;
         Random randomGenerator = new Random();

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // STANDARD WORD COUNT with reduceByKey
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         wordCounts = docs
         .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
         String[] tokens = document.split(" ");
         HashMap<String, Long> counts = new HashMap<>();
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (String token : tokens) {
         counts.put(token, 1L + counts.getOrDefault(token, 0L));
         }
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .reduceByKey((x, y) -> x+y);    // <-- REDUCE PHASE (R1)
         numwords = wordCounts.count();
         System.out.println("Number of distinct words in the documents = " + numwords);

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // IMPROVED WORD COUNT (keys in [0,K-1]) with groupByKey
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         wordCounts = docs
         .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
         String[] tokens = document.split(" ");
         HashMap<String, Long> counts = new HashMap<>();
         ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
         for (String token : tokens) {
         counts.put(token, 1L + counts.getOrDefault(token, 0L));
         }
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(randomGenerator.nextInt(K), new Tuple2<>(e.getKey(), e.getValue())));
         }
         return pairs.iterator();
         })
         .groupByKey()    // <-- SHFFLE+GROUPING
         .flatMapToPair((element) -> { // <-- REDUCE PHASE (R1)
         HashMap<String, Long> counts = new HashMap<>();
         for (Tuple2<String, Long> c : element._2()) {
         counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
         }
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2)
         numwords = wordCounts.count();
         System.out.println("Number of distinct words in the documents = " + numwords);

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // IMPROVED WORD COUNT (keys in [0,K-1]) with groupBy
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         wordCounts = docs
         .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
         String[] tokens = document.split(" ");
         HashMap<String, Long> counts = new HashMap<>();
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (String token : tokens) {
         counts.put(token, 1L + counts.getOrDefault(token, 0L));
         }
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .groupBy((wordcountpair) -> randomGenerator.nextInt(K))  // <-- KEY ASSIGNMENT+SHFFLE+GROUPING
         .flatMapToPair((element) -> { // <-- REDUCE PHASE (R1)
         HashMap<String, Long> counts = new HashMap<>();
         for (Tuple2<String, Long> c : element._2()) {
         counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
         }
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2)
         numwords = wordCounts.count();
         System.out.println("Number of distinct words in the documents = " + numwords);

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // IMPROVED WORD COUNT with mapPartitions
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         wordCounts = docs
         .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
         String[] tokens = document.split(" ");
         HashMap<String, Long> counts = new HashMap<>();
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (String token : tokens) {
         counts.put(token, 1L + counts.getOrDefault(token, 0L));
         }
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .mapPartitionsToPair((element) -> {    // <-- REDUCE PHASE (R1)
         HashMap<String, Long> counts = new HashMap<>();
         while (element.hasNext()){
         Tuple2<String, Long> tuple = element.next();
         counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
         }
         ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
         for (Map.Entry<String, Long> e : counts.entrySet()) {
         pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
         }
         return pairs.iterator();
         })
         .groupByKey()     // <-- SHUFFLE+GROUPING
         .mapValues((it) -> { // <-- REDUCE PHASE (R2)
         long sum = 0;
         for (long c : it) {
         sum += c;
         }
         return sum;
         }); // Obs: one could use reduceByKey in place of groupByKey and mapValues
         numwords = wordCounts.count();
         System.out.println("Number of distinct words in the documents = " + numwords);

         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
         // COMPUTE AVERAGE WORD LENGTH
         // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

         int avgwordlength = wordCounts
         .map((tuple) -> tuple._1().length())
         .reduce((x, y) -> x+y);
         System.out.println("Average word length = " + avgwordlength/numwords);

         */

    }


}
