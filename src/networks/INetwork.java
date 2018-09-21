package networks;

import utilities.Matrix;

public interface INetwork {
	public void train(float[][] x, float[][] y, int epochs) throws Exception;
	public Matrix predict(float[][] x);
}
