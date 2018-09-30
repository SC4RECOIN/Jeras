package examples;

import layers.Dense;
import layers.Sequential;
import utilities.Matrix;

public class XORSequential {
	public static void main(String[] args) {
        Matrix x = new Matrix(new float[][] {{0,0,1},
                                             {0,1,1},
                                             {1,0,1},
                                             {1,1,1}});
        
        Matrix y = new Matrix(new float[][] {{1, 0},{0, 1},{0, 1},{1, 0}});

        int epochs = 5000;
        float lr = 0.1f;
        
        Sequential model = new Sequential();
        model.add(new Dense(8, "sigmoid", x.columns));
        model.add(new Dense(8, "sigmoid"));
        model.add(new Dense(y.columns, "softmax"));
        model.compile(lr);
        
        model.fit(x, y, epochs);
        System.out.println(model.predict(x));
    }
}

