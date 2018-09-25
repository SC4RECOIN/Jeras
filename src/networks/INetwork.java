package networks;

import utilities.Matrix;

public interface INetwork {
	public void train(Matrix x, Matrix y, int epochs);
	public Matrix predict(Matrix x);
}
