package networks;

import static utilities.Activations.sigmoid;
import static utilities.Activations.sigmoidDerivative;
import static utilities.Functions.add;
import static utilities.Functions.dot;
import static utilities.Functions.multiply;
import static utilities.Functions.subtract;
import static utilities.Functions.transpose;

import java.util.ArrayList;

import utilities.Initializers.Init;
import utilities.Matrix;

public class MLP implements INetwork {
	
	private int _numInputs;
	private int[] _hiddenLayers;
	private int _numOutputs;
	
	private ArrayList<float[][]> _weights;
	private ArrayList<float[][]> _layerOutputs;
	
	private float[][] _x = null;
	private float[][] _y = null;
	
	private Init _initializer;
	
	public MLP(int numInputs, int[] hiddenLayers, int numOutputs, Init initializer) {
		_numInputs = numInputs;
		_hiddenLayers = hiddenLayers;
		_numOutputs = numOutputs;
		
		_weights = new ArrayList<float[][]>();
		_layerOutputs = new ArrayList<float[][]>();
		
		_initializer = initializer;
	
		initWeights();            
	}
	
	public MLP(int numInputs, int[] hiddenLayers, int numOutputs) {
		this(numInputs, hiddenLayers, numOutputs, Init.NORMAL);        
	}
	
	private void initWeights() {
		_weights.add((new Matrix(_numInputs, _hiddenLayers[0], _initializer)).values);
		for (int i = 1; i < _hiddenLayers.length - 1; i++) {
			_weights.add((new Matrix(_hiddenLayers[i], _hiddenLayers[i+1], _initializer)).values);
		}
		_weights.add((new Matrix(_hiddenLayers[_hiddenLayers.length-1], _numOutputs, _initializer)).values);
	}

	private void setTrainingData(float x[][], float[][] y) throws Exception {
		if (x[0].length != _numInputs) {
			throw new Exception("Data shape does not match network inputs");
		}
		_x = x;
		_y = y;
	}
	
	private void feedForward(float[][] x) {
		_layerOutputs.clear();
		_layerOutputs.add(sigmoid(dot(x, _weights.get(0))));
		for (int i = 1; i < _weights.size(); i++) {
			_layerOutputs.add(sigmoid(dot(_layerOutputs.get(i-1), _weights.get(i))));
		}
	}
	
	private void feedForward() {
		feedForward(_x);
	}
	
	private void backPropogation() throws Exception {
		ArrayList<float[][]> results = new ArrayList<float[][]>();
		
		// application of the chain rule to find derivative of the loss function with respect to weights
		float[][] subResult = multiply(multiply(subtract(_y, _layerOutputs.get(1)), 2), sigmoidDerivative(_layerOutputs.get(1)));
		results.add(dot(transpose(_layerOutputs.get(0)), subResult));
		results.add(dot(transpose(_x), multiply(dot(subResult, transpose(_weights.get(1))), sigmoidDerivative(_layerOutputs.get(0)))));

		_weights.set(1, add(_weights.get(1), results.get(0)));
		_weights.set(0, add(_weights.get(0), results.get(1)));
	}
	
	public void train(float[][] x, float[][] y, int epochs) throws Exception {
		setTrainingData(x, y);
		for (int i = 0; i < epochs; i++) {
			feedForward();
			backPropogation();
		}
	}
	
	public Matrix predict(float[][] x) {
		feedForward(x);
		return new Matrix(_layerOutputs.get(_layerOutputs.size() - 1));
	}
}
