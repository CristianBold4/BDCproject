import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.*;

public class G052HW3 {

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Print input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end - start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end - start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers(JavaRDD<Vector> points, int k, int z, int L) {

        long initial_time1, final_time1, total_time1;
        long initial_time2, final_time2, total_time2;


        initial_time1 = System.nanoTime();
        //------------- ROUND 1 ---------------------------

        JavaRDD<Tuple2<Vector, Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k + z + 1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1


        //------------- ROUND 2 ---------------------------


        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k + z) * L);
        elems.addAll(coreset.collect());

        final_time1 = System.nanoTime();
        total_time1 = final_time1 - initial_time1;


        initial_time2 = System.nanoTime();
        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //
        ArrayList<Vector> vecs = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();

        for (Tuple2<Vector, Long> t : elems) {
            vecs.add(t._1());
            weights.add(t._2);
        }

        ArrayList<Vector> centers = SeqWeightedOutliers(vecs, weights, k, z , 2);
        final_time2 = System.nanoTime();

        total_time2 = final_time2 - initial_time2;

        System.out.println("Time Round 1: " + total_time1 / 10e6  + " ms");
        System.out.println("Time Round 2: " + total_time2 / 10e6  + " ms");

        return centers;

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT(ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius = 0;

        for (int iter = 1; iter < k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //
    // ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    //

    /**
     * Computes and returns the set of centers according to the algorithm implemented to solve the
     * k-center with z outliers problem.
     *
     * @param P     the set of points
     * @param W     the set of weights
     * @param k     the number of centers
     * @param z     the number of outliers
     * @param alpha the coefficient used by the algorithm
     * @return the set of k centers
     */
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, float alpha) {


        double r = Math.sqrt(Vectors.sqdist(P.get(0), P.get(1)));
        for (int i = 0; i <= k + z + 1; i++) {
            for (int j = i + 1; j <= k + z; j++) {
                r = Math.min(r, Math.sqrt(Vectors.sqdist(P.get(i), P.get(j))));
            }
        }
        r /= 2;

        System.out.println("Initial guess = " + r);

        int guess = 1;

        while (true) {

            ArrayList<Vector> Z = new ArrayList<>(P);
            ArrayList<Vector> S = new ArrayList<>();
            //The set of weights associated to each point in Z at the same index
            ArrayList<Long> weightsZ = new ArrayList<>(W);
            long Wz = 0;
            for (Long weight : W) {
                Wz += weight;
            }
            while ((S.size() < k) && (Wz > 0)) {
                long max = 0;
                int new_center_index = -1;
                for (int i = 0; i < P.size(); i++) {
                    Map<Integer, Vector> Bz = computeBall(Z, P.get(i), (1 + 2 * alpha) * r);
                    long ball_weight = 0;
                    for (Integer index : Bz.keySet()) {
                        ball_weight += weightsZ.get(index);
                    }
                    if (ball_weight > max) {
                        max = ball_weight;
                        new_center_index = i;
                    }
                }
                S.add(P.get(new_center_index));

                Map<Integer, Vector> Bz = computeBall(Z, P.get(new_center_index), (3 + 4 * alpha) * r);

                //Get all the points to be removed from Z according to Bz
                ArrayList<Integer> indexToRemove = new ArrayList<>(Bz.keySet());
                Collections.sort(indexToRemove);

                //Safely remove from last to first to avoid problems when shifting elements inside arraylist
                for (int i = indexToRemove.size() - 1; i >= 0; i--) {
                    Z.remove((int) indexToRemove.get(i));
                    Wz -= weightsZ.get(indexToRemove.get(i));
                    weightsZ.remove((int) indexToRemove.get(i));
                }
            }
            if (Wz <= z) {
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + guess);
                return S;
            } else {
                r *= 2;
                guess++;
            }
        }

    }

    /**
     * Computes the ball of {@code Z} with radius {@code r} centered at {@code ballCenter}
     *
     * @param Z          the subset of considered points
     * @param ballCenter the center of the ball
     * @param r          the radius
     * @return the set of points inside the given ball with their corresponding indexes
     */
    private static Map<Integer, Vector> computeBall(ArrayList<Vector> Z, Vector ballCenter, double r) {
        Map<Integer, Vector> Bz = new Hashtable<>();
        for (int i = 0; i < Z.size(); i++) {
            if (Math.sqrt(Vectors.sqdist(ballCenter, Z.get(i))) <= r) {
                Bz.put(i, Z.get(i));
            }
        }
        return Bz;
    }


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective(JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {

        //
        // ****** ADD THE CODE FOR computeObjective
        //

        double min_dist, current_dist;


        // -- list where to store distances
        ArrayList<Double> distances = new ArrayList<>();

        // -- compute all distances, for each z x in P, from S
        for (Vector x : points.collect()) {
            min_dist = Double.POSITIVE_INFINITY;
            for (Vector y : centers) {
                current_dist = euclidean(x, y);
                if (current_dist < min_dist) {
                    //System.out.println("Min distance -> " + current_dist);
                    min_dist = current_dist;
                }
            }

            distances.add(min_dist);
        }

        // -- Sort the distances
        Collections.sort(distances);

        //System.out.println(Arrays.toString(distances.toArray()));

        // -- return the largest distance excluding the z-largest ones
        return (distances.get(distances.size() - z - 1));


    }

}
