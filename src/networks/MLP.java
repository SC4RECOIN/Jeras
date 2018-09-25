package networks;

import static utilities.Activations.sigmoid;
import static utilities.Activations.sigmoidDerivative;

import java.util.ArrayList;

import utilities.Initializers.Init;
import utilities.Matrix;

public class MLP implements INetwork {
	
	private int _numInputs;
	private int[] _hiddenLayers;
	private int _numOutputs;
	
	private ArrayList<Matrix> _weights;
	private ArrayList<Matrix> _layerOutputs;
	
	private Matrix _x = null;
	private Matrix _y = null;
	
	private Init _initializer;
	
	public MLP(int numInputs, int[] hiddenLayers, int numOutputs, Init initializer) {
		_numInputs = numInputs;
		_hiddenLayers = hiddenLayers;
		_numOutputs = numOutputs;
		_initializer = initializer;
		
		_weights = new ArrayList<Matrix>();
		_layerOutputs = new ArrayList<Matrix>();
	
		initWeights();            
	}
	
	public MLP(int numInputs, int[] hiddenLayers, int numOutputs) {
		this(numInputs, hiddenLayers, numOutputs, Init.NORMAL);        
	}
	
	private void initWeights() {
		_weights.add(new Matrix(_numInputs, _hiddenLayers[0], _initializer));
		for (int i = 1; i < _hiddenLayers.length - 1; i++) {
			_weights.add(new Matrix(_hiddenLayers[i], _hiddenLayers[i+1], _initializer));
		}
		_weights.add(new Matrix(_hiddenLayers[_hiddenLayers.length-1], _numOutputs, _initializer));
	}

	private void setTrainingData(Matrix x, Matrix y) {
		if (x.columns != _numInputs) {
			throw new RuntimeException("Data shape does not match network inputs");
		}
		_x = x;
		_y = y;
	}
	
	private void feedForward(Matrix x) {
		_layerOutputs.clear();
		_layerOutputs.add(sigmoid(x.dot(_weights.get(0))));
		for (int i = 1; i < _weights.size(); i++) {
			_layerOutputs.add(sigmoid(_layerOutputs.get(i-1).dot(_weights.get(i))));
		}
	}
	
	private void backPropogation() {
		ArrayList<Matrix> results = new ArrayList<Matrix>();
		
		try {
			// application of the chain rule to find derivative of the loss function with respect to weights
			Matrix subResult = _y.sub(_layerOutputs.get(1)).mult(2).mult(sigmoidDerivative(_layerOutputs.get(1)));
			results.add(_layerOutputs.get(0).T().dot(subResult));
			results.add(_x.T().dot(subResult.dot(_weights.get(1).T()).mult(sigmoidDerivative(_layerOutputs.get(0)))));
	
			_weights.set(1, _weights.get(1).add(results.get(0)));
			_weights.set(0, _weights.get(0).add(results.get(1)));
		
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void train(Matrix x, Matrix y, int epochs) {
		setTrainingData(x, y);
		for (int i = 0; i < epochs; i++) {
			feedForward(_x);
			backPropogation();
		}
	}
	
	public Matrix predict(Matrix x) {
		feedForward(x);
		return _layerOutputs.get(_layerOutputs.size() - 1);
	}
}
