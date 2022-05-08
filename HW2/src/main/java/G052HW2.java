import org.apache.commons.math3.util.MathUtils;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


public class G052HW2 {

    // -- DRAFT @Cristian 
    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, double z) {

        // -- number of centers
        int k = S.size();
        double min_dist, current_dist;

        // -- list where to store distances
        ArrayList<Double> distances = new ArrayList<>();

        // -- compute all distances, for each z x in P, from S
        for (Vector x : P) {
            min_dist = Double.POSITIVE_INFINITY;
            for (Vector y : S) {
                current_dist = Math.sqrt(Vectors.sqdist(x, y));
                if (current_dist < min_dist) {
                    min_dist = current_dist;
                }
            }

            distances.add(min_dist);
        }

        // -- Sort the distances
        Collections.sort(distances);


        // -- return the largest distance excluding the z-largest ones
        int i = 0;
        while (distances.get(i) <= z) {
            i++;
        }

        return distances.get(i - 1);

    }

    /**
     * Computes and returns the set of centers according to the algorithm implemented to solve the
     * k-center with z outliers problem.
     *
     * @param P  the set of points
     * @param W  the set of weights
     * @param k  the number of centers
     * @param z  the number of outliers
     * @param alpha  the coefficient used by the algorithm
     * @return  the set of k centers
     */
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P,ArrayList<Long> W, int k,int z, float alpha){
        double r=Math.sqrt(Vectors.sqdist(P.get(0), P.get(1)));
        for(int i=0;i<=k+z+1;i++){
            for(int j=i+1;j<=k+z;j++){
                r=Math.min(r,Math.sqrt(Vectors.sqdist(P.get(i), P.get(j))));
            }
        }
        r/=2;
        //Just for testing purposes
        System.out.println("GUESS"+r);

        while(true){

            ArrayList<Vector> Z= new ArrayList<>(P);
            ArrayList<Vector> S=new ArrayList<>();
            //The set of weights associated to each point in Z at the same index
            ArrayList<Long> weightsZ=new ArrayList<>(W);
            long Wz=0;
            for(Long weight:W){
                Wz+=weight;
            }
            while((S.size()<k) && (Wz>0)){
                long max=0;
                int new_center_index=-1;
                for(int i=0;i<P.size();i++){
                    Map<Integer,Vector> Bz=computeBall(Z,P.get(i),(1+2*alpha)*r);
                    long ball_weight=0;
                    for(Integer index:Bz.keySet()){
                        ball_weight+=weightsZ.get(index);
                    }
                    if(ball_weight>max){
                        max=ball_weight;
                        new_center_index=i;
                    }
                }
                S.add(P.get(new_center_index));

                Map<Integer,Vector> Bz=computeBall(Z,P.get(new_center_index),(3+4*alpha)*r);

                //Get all the points to be removed from Z according to Bz
                ArrayList<Integer> indexToRemove = new ArrayList<>(Bz.keySet());
                Collections.sort(indexToRemove);

                //Safely remove from last to first to avoid problems when shifting elements inside arraylist
                for(int i=indexToRemove.size()-1;i>=0;i--){
                    Z.remove((int)indexToRemove.get(i));
                    Wz-=weightsZ.get(indexToRemove.get(i));
                    weightsZ.remove((int)indexToRemove.get(i));
                }
            }
            if(Wz<=z){
                System.out.println(r);
                return S;
            }
            else{
                r*=2;
                //Just for testing purposes
                System.out.println("GUESS"+r);
            }
        }

    }

    /**
     * Computes the ball of {@code Z} with radius {@code r} centered at {@code ballCenter}
     * @param Z  the subset of considered points
     * @param ballCenter  the center of the ball
     * @param r  the radius
     * @return  the set of points inside the given ball with their corresponding indexes
     */
    private static Map<Integer, Vector> computeBall(ArrayList<Vector> Z, Vector ballCenter, double r){
        Map<Integer,Vector> Bz=new Hashtable<>();
        for(int i=0;i<Z.size();i++){
            if(Math.sqrt(Vectors.sqdist(ballCenter, Z.get(i)))<=r){
                Bz.put(i,Z.get(i));
            }
        }
        return Bz;
    }


    /**
     * Converts a {@code String} to {@code Vector}
     * @param str the {@code String} to convert
     * @return  the {@code Vector} obtained from the given {@code String}
     * @throws IOException
     */
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    /**
     * Reads a file and creates the corresponding list of points
     * @param filename  the path to the file
     * @return  the list of points read
     * @throws IOException
     */
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


    public static void main(String[] args) throws IOException{

        //EVERYTHING HERE WAS JUST TO TEST THE FUNCTION
        ArrayList<Vector> points=readVectorsSeq("PATH");
        ArrayList<Long> weights=new ArrayList<>();
        for(Vector p:points){
            weights.add(1L);
        }
        SeqWeightedOutliers(points,weights,9,300,0);
        // -- HOMEWORK 2
    }

}
