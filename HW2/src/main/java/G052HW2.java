import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;
import java.util.Collections;


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

    public static void main(String[] args) {

        // -- HOMEWORK 2
    }

}
