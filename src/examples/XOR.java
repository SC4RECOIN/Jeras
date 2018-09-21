package examples;

import networks.MLP;

public class XOR {

	public static void main(String[] args) {
		float[][] x = {{0,0,1},
	                   {0,1,1},
	                   {1,0,1},
	                   {1,1,1}};
		
	    float[][] y ={{1, 0},{0, 1},{0, 1},{1, 0}};
	    
	    int inputs = 3;
	    int[] hidden = {8};
	    int outputs = 2;
	    
	    MLP nn = new MLP(inputs, hidden, outputs);
	    
	    try {
	    	nn.train(x, y, 3000);
	    	System.out.println(nn.predict(x));
	    	
	    } catch (Exception e) {
	    	e.printStackTrace();
	    }
	}
}
