package examples;

import networks.MLP;
import utilities.Matrix;

public class XOR {

    public static void main(String[] args) {
        Matrix x = new Matrix(new float[][] {{0,0,1},
                                             {0,1,1},
                                             {1,0,1},
                                             {1,1,1}});
        
        Matrix y = new Matrix(new float[][] {{1, 0},{0, 1},{0, 1},{1, 0}});

        int inputs = x.columns;
        int hidden1 = 8;
        int hidden2 = 8;
        int outputs = y.columns;
        int epochs = 1000;
        float lr = 0.1f;
        
        int[] networkShape = {inputs, hidden1, hidden2, outputs};
        
        MLP nn = new MLP(networkShape, lr);
        
        nn.train(x, y, epochs);
        System.out.println(nn.predict(x));
    }
}
