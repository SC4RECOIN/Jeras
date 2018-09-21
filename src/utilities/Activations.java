package utilities;

public class Activations {
	
	public static float sigmoid(float x) {
	    return (float) (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}

	public static float[][] sigmoid(float[][] x, boolean deriv) {
	    float[][] result = new float[x.length][x[0].length];

	    for (int i = 0; i < x.length; i++) {
	        for (int j = 0; j < x[i].length; j++) {
	            float sigmoidCell = sigmoid(x[i][j]);

	            if (deriv == true) {
	                result[i][j] = sigmoidCell * (1 - sigmoidCell);
	            } else {
	                result[i][j] = sigmoidCell;
	            }
	        }
	    }
	    return result;
	}
	
	public static float[][] sigmoid(float[][] x) {
		return sigmoid(x, false);
	}
	
	public static float[][] sigmoidDerivative(float[][] x) {
		return sigmoid(x, true);
	}
}
