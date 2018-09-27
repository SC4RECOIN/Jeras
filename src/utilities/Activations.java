package utilities;

public class Activations {
	
	private static float sigmoid(float x) {
	    return (float) (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}

	private static Matrix sigmoid(Matrix x, boolean deriv) {
		float[][] result = new float[x.rows][x.columns];

	    for (int i = 0; i < x.rows; i++) {
	        for (int j = 0; j < x.columns; j++) {
	            float sig = sigmoid(x.values[i][j]);

	            if (deriv == true) {
	                result[i][j] = sig * (1 - sig);
	            } else {
	                result[i][j] = sig;
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

	private static Matrix exp(Matrix x) {
		float[][] result = new float[x.rows][x.columns];

	    for (int i = 0; i < x.rows; i++) {
	        for (int j = 0; j < x.columns; j++) {
	            result[i][j] = (float) Math.pow(Math.E, (x.values[i][j]));
	        }
	    }
	    return new Matrix(result);
	}
	
	public static Matrix softmax(Matrix x) throws Exception {
		Matrix exp = exp(x);
		Matrix sum = exp.sum();
		Matrix result = exp.div(sum);

		return result;
	}
}
