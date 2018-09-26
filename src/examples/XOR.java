package examples;

import networks.MLP;
import utilities.Matrix;

public class XOR {

	public static void main(String[] args) {
		Matrix x = new Matrix (new float[][] {{0,0,1},
							                  {0,1,1},
							                  {1,0,1},
							                  {1,1,1}});
		
	    Matrix y = new Matrix (new float[][] {{1, 0},{0, 1},{0, 1},{1, 0}});

	    int[] networkShape = {3, 8, 2};
	    MLP nn = new MLP(networkShape);
	    
	    nn.train(x, y, 4500);
	    System.out.println(nn.predict(x));
	}
}
