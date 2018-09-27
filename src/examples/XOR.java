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

	    // inputs - hidden - outputs
	    int[] networkShape = {x.columns, 5, 5, y.columns};
	    MLP nn = new MLP(networkShape, 0.1f);
	    
	    nn.train(x, y, 1000);
	    System.out.println(nn.predict(x));
	}
}
