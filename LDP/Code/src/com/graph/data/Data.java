package com.graph.data;

import com.graph.metric.Tools;

import java.io.BufferedReader;
import java.io.FileReader;

public class Data {
    public static void readGraph(String filename, boolean[][] mat) throws Exception {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                mat[i][j] = false;
            }
        }
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = br.readLine()) != null) {
            String[] node = line.split(",");
            if (node.length == 1) {
                node = line.split(" ");
            }
            if (node.length == 1) {
                node = line.split("\t");
            }
            if (node.length != 2) {
                System.out.println("line !=2!!");
                throw new RuntimeException("File Input Error!" + line + "," + node.length + "," + node[0] + "," + "\n");
            }
            int sNode = Integer.parseInt(node[0]);
            int dNode = Integer.parseInt(node[1]);
            mat[sNode][dNode] = true;
            mat[dNode][sNode] = true;
        }
        br.close();
    }

    public static void main(String args[]) throws Exception {
        int dataset = 3;
        String filename = Tools.inputFilename[dataset - 1] + ".txt";
        int userNum = Tools.inputFileUserNum[dataset - 1];
        boolean[][] mat = new boolean[userNum][userNum];
        Data.readGraph(filename, mat);
        double[] degree = Tools.getDegree(mat);

        Tools.getPopularDegree(degree);
    }
}
