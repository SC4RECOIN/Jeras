package layers;

import static utilities.Activations.sigmoidDerivative;

import java.util.ArrayList;

import utilities.Matrix;

public class Sequential {
	
	private boolean compiled;
	ArrayList<Dense> layers;
	private ArrayList<Matrix> out;
	private Matrix output;
	private int numLayers;
	
	private float _lr;

	public Sequential() {
		compiled = false;
		layers = new ArrayList<Dense>();
		out = new ArrayList<Matrix>();
	}

	public void add(Dense dense) {
		if (layers.size() == 0 && dense.getInputs() == 0) {
			throw new RuntimeException("You must specify number of inputs for the first layer");
		}
		layers.add(dense);
		dense.setLayerNum(layers.size());
	}

	public void compile(float lr) {
		_lr = lr;
		// set input dimensions based on previous layers output
		for (int i = 1; i < layers.size(); i++) {
			layers.get(i).setInputs(layers.get(i-1).getOutputs());
		}
		numLayers = layers.size();
		compiled = true;
	}
	
	private void forwardPropagate(Matrix x) {
        out.clear();

        try {
        	out.add(layers.get(0).feed(x));
            for (int i = 1; i < numLayers; i++) {
            	Dense nextLayer = layers.get(i);
            	Matrix layerInput = out.get(i-1);
            	out.add(nextLayer.feed(layerInput));
            }
            output = out.get(out.size()-1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void backPropagate(Matrix X, Matrix Y) {
        try {
            ArrayList<Matrix> deltas = new ArrayList<Matrix>();
            
            Matrix error = output.sub(Y);
            deltas.add(error);
            
            for (int i = out.size() - 1; i > 0; i--) {
                Matrix d = deltas.get(deltas.size() - 1);
                if (i == 1) {
                	Dense nextLayer = layers.get(i);
                	Dense lastLayer = layers.get(i-1);
                    deltas.add(nextLayer.backfeed(d, X, lastLayer));
                } else {
                	Dense nextLayer = layers.get(i);
                	Dense lastLayer = layers.get(i-1);
                	deltas.add(nextLayer.backfeed(d, out.get(i-2), lastLayer));
                }
            }
            
            // correct weights and biases using learning rate
            Matrix delta = deltas.get(deltas.size() - 1);
            Dense firstLayer = layers.get(0);
            firstLayer.w = firstLayer.w.sub(X.T().dot(delta).mult(_lr));
            firstLayer.b = firstLayer.b.sub(delta.sumaxis().mult(_lr));
            
            for (int i = 1; i < numLayers; i++) {
            	Dense layer = layers.get(i);
                delta = deltas.get(numLayers - i - 1);
                layer.w = layer.w.sub(out.get(i-1).T().dot(delta).mult(_lr));
                layer.b = layer.b.sub(delta.sumaxis().mult(_lr));
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

	public void fit(Matrix x, Matrix y, int epochs) {
		if (!compiled) {
			throw new RuntimeException("You must compile the model before training");
		}
		
		for (int i = 0; i < epochs; i++) {
			forwardPropagate(x);
			backPropagate(x, y);
		}
	}

	public Matrix predict(Matrix x) {
		forwardPropagate(x);
		return output;
	}
}
