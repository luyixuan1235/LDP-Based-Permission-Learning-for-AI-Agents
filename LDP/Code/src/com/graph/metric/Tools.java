package com.graph.metric;
import java.util.Random;

public class Tools {

    public static final double[][][] optimalPercentage = {
            {{0.7071, 0.8098, 0.8614, 0.8923, 0.9126, 0.9268, 0.9372, 0.9449}, {0.8064, 0.8758, 0.9071, 0.9225, 0.9279, 0.9259, 0.9188, 0.9080}}, 
            {{0.6929, 0.7927, 0.8460, 0.8792, 0.9016, 0.9175, 0.9291, 0.9379}, {0.6223, 0.7255, 0.7908, 0.8345, 0.8643, 0.8844, 0.8971, 0.9028}}, 
            {{0.6929, 0.7927, 0.8460, 0.8792, 0.9016, 0.9175, 0.9291, 0.9379}, {0.7195, 0.8129, 0.8617, 0.8913, 0.9097, 0.9120, 0.9232, 0.9199}}, 
            {{0.6929, 0.7927, 0.8460, 0.8792, 0.9016, 0.9175, 0.9291, 0.9379}, {0.7323, 0.8231, 0.8700, 0.8988, 0.9175, 0.9300, 0.9380, 0.9418}}, 

            {{0.6929, 0.7927, 0.8460, 0.8792, 0.9016, 0.9175, 0.9291, 0.9379}, {0.7323, 0.8231, 0.8700, 0.8988, 0.9175, 0.9300, 0.9380, 0.9418}}, 
            {{0.6929, 0.7927, 0.8460, 0.8792, 0.9016, 0.9175, 0.9291, 0.9379}, {0.9257, 0.9550, 0.9677, 0.9747, 0.9786, 0.9801, 0.9796, 0.9772}}  
    };

    public static final String[] inputFilename = {
            "dataset/facebook_combined",
            "dataset/Email-Enron",
            "dataset/CA-AstroPh-transform",
            "dataset/Brightkite_edges",
            "dataset/twitter_combined_transform",
            "dataset/gplus_combined_transform"};

    public static final int[] inputFileUserNum = {4039, 36692, 18772, 58228, 81306, 107614};

    public static double getOneDegree(boolean[][] mat, int node) {
        double degree = 0.0;
        for (int i = 0; i < mat[0].length; i++) {
            if (mat[node][i] == true && node != i) {
                degree++;
            }
        }
        return degree;
    }

    public static double[] getDegree(boolean[][] mat) {
        int userNum = mat[0].length; 
        double[] degree = new double[userNum]; 
       
        for (int i = 0; i < userNum; i++)
            degree[i] = getOneDegree(mat, i);
        return degree;
    }
   
   
    public static double getEdges(boolean[][] mat) {
        int userNum = mat[0].length;
        double[] degree = getDegree(mat);
        double edges = 0;
        for (int i = 0; i < userNum; i++)
            edges += degree[i];
        return edges / 2;
    }

    public static double[] addLaplaceNoise_Degree(double[] degree, double ep) {
        int userNum = degree.length;
        double[] noisyDegree = new double[userNum]; 

        for (int i = 0; i < userNum; i++) { 
            noisyDegree[i] = degree[i] + Tools.LaplaceDist(2.0, ep); 
            if (noisyDegree[i] < 1) { 
                noisyDegree[i] = 1;
            } else if (noisyDegree[i] > userNum - 1) {  
                noisyDegree[i] = userNum - 1;
            }
        }

        return noisyDegree; 
    }

   
    public static double LaplaceDist(double sensitivity, double ep) {
        double scale = sensitivity / ep;
        Random rng = new Random(42);
        double U = rng.nextDouble() - 0.5;
        return -scale * Math.signum(U) * Math.log(1 - 2 * Math.abs(U));
    }


    public static double mergeTwoNoisyDegree(double v1, double r1, double v2, double r2) {
        return v1 * r2 / (r1 + r2) + v2 * r1 / (r1 + r2);
    }

    public static double[] mergeTwoNoisyDegree_list(double[] v1, double r1, double[] v2, double[] r2) {
        int userNum = v1.length;
        double[] noisyDegree = new double[userNum];
        for (int i = 0; i < userNum; i++)
            noisyDegree[i] = mergeTwoNoisyDegree(v1[i], r1, v2[i], r2[i]);
        return noisyDegree;
    }

    public static double[] getVariance_list(double[] degree, double epsilon) {
        double userNum = degree.length;
        double[] variance = new double[(int) userNum];
        double p = Math.exp(epsilon) / (1 + Math.exp(epsilon));
        for (int i = 0; i < userNum; i++)
            variance[i] = userNum * (1 / (16 * (p - 0.5) * (p - 0.5)) - (degree[i] / userNum - 0.5) * (degree[i] / userNum - 0.5));
        return variance;
    }

    public static double getPopularDegree(double[] degree) {
        int userNum = degree.length;
        double[] count = new double[userNum];
        for (int i = 0; i < userNum; i++)
            count[(int) (degree[i])]++;
        double max = count[0];
        int maxIndex = 0;
        for (int i = 0; i < userNum; i++) {
            if (max < count[i]) {
                max = count[i];
                maxIndex = i;
            }
        }
        System.out.println("max=" + max);
        return maxIndex;
    }

}
