package examples;

import networks.MLP;
import utilities.Matrix;

public class XOR {

	public static void main(String[] args) {
		float[][] x = {{0,0,1},
	                   {0,1,1},
	                   {1,0,1},
	                   {1,1,1}};
		
	    float[][] y = {{1, 0},{0, 1},{0, 1},{1, 0}};
	    
	    int inputs = 3;
	    int[] hidden = {8};
	    int outputs = 2;
	    
	    MLP nn = new MLP(inputs, hidden, outputs);
	    
	    try {
	    	nn.train(new Matrix(x), new Matrix(y), 4500);
	    	System.out.println(nn.predict(new Matrix(x)));
	    	
	    } catch (Exception e) {
	    	e.printStackTrace();
	    }
	}
}
