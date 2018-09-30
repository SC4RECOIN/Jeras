package layers;

import static utilities.Activations.sigmoid;
import static utilities.Activations.sigmoidDerivative;

import utilities.Array;
import utilities.Matrix;
import utilities.Initializers.Init;

public class Dense {
	
	private int inputs;
	private int outputs;
	private String activation;
	
	public Matrix w;
	public Array b;
	
	private int layerNum;

	public Dense(int outputs, String activation) {
		this.inputs = 0;
		this.outputs = outputs;
		this.activation = activation;
	}
	
	public Dense(int outputs, String activation, int inputs) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.activation = activation;
		
		w = new Matrix(inputs, outputs, Init.RANDOM);
		b = new Array(outputs, Init.RANDOM);
	}
	
	public int getInputs() { return this.inputs; }
	public int getOutputs() { return this.outputs; }
	
	public void setInputs(int inputs) {
		this.inputs = inputs;
		w = new Matrix(inputs, outputs, Init.RANDOM);
		b = new Array(outputs, Init.RANDOM);
	}
	
	public void setLayerNum(int num) {
		layerNum = num;
	}
	
	public Matrix feed(Matrix input) {
		Matrix output;
		try {
			output = sigmoid(input.dot(w).add(b));
		} catch (Exception e) {
			throw new RuntimeException("Error feeding layer " + layerNum + " (Dense)");
		}
		return output;
	}
	
	public Matrix backfeed(Matrix delta, Matrix layerInput, Dense lastLayer) {
		Matrix result;
    	try {
			result = delta.dot(w.T()).mult(sigmoidDerivative(layerInput.dot(lastLayer.w).add(lastLayer.b)));
		} catch (Exception e) {
			throw new RuntimeException("Error during backpropagation (layer " + layerNum + ":Dense)");
		}
    	return result;
    }
}
