package utilities;

public class Activations {
	
	public static float sigmoid(float x) {
	    return (float) (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}

	public static Matrix sigmoid(Matrix x, boolean deriv) {
		float[][] result = new float[x.rows][x.columns];

	    for (int i = 0; i < x.rows; i++) {
	        for (int j = 0; j < x.columns; j++) {
	            float sigmoidCell = sigmoid(x.values[i][j]);

	            if (deriv == true) {
	                result[i][j] = sigmoidCell * (1 - sigmoidCell);
	            } else {
	                result[i][j] = sigmoidCell;
	            }
	        }
	    }
	    return new Matrix(result);
	}
	
	public static Matrix sigmoid(Matrix x) {
		return sigmoid(x, false);
	}
	
	public static Matrix sigmoidDerivative(Matrix x) {
		return sigmoid(x, true);
	}
}
