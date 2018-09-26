package networks;

import static utilities.Activations.sigmoid;
import static utilities.Activations.softmax;
import static utilities.Activations.sigmoidDerivative;

import java.util.ArrayList;

import utilities.Array;
import utilities.Initializers.Init;
import utilities.Matrix;

public class MLP implements INetwork {
	
	private ArrayList<Matrix> w;
	private ArrayList<Array> b;
	private ArrayList<Matrix> out;
	
	private final float lr;

	public MLP(int[] networkShape, Init initializer) {
		// weights
		w = new ArrayList<Matrix>();
        for (int i = 0; i < networkShape.length - 1; i++) {
			w.add(new Matrix(networkShape[i], networkShape[i+1], initializer));
		} 

		// bias
        b = new ArrayList<Array>();
        for (int i = 1; i < networkShape.length; i++) {
			b.add(new Array(networkShape[i], initializer));
		}

		// learning rate
		lr = 0.0001f;

		// outputs from forward propagation
		out = new ArrayList<Matrix>();
	}
	
	public MLP(int[] networkShape) {
		this(networkShape, Init.NORMAL);        
	}

	private void forwardPropagate(Matrix x) {
		out.clear();

		try { 
			out.add(sigmoid(x.dot(w.get(0)).add(b.get(0)))); 
			out.add(softmax(out.get(0).dot(w.get(1)).add(b.get(1))));
			
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private void backPropagate(Matrix x, Matrix Y) {
		try {
			Matrix delta2 = out.get(1).sub(Y);
		    Matrix delta1 = delta2.dot(w.get(1).T()).mult(sigmoidDerivative(x.dot(w.get(0)).add(b.get(0))));
		    
		    System.out.println(delta2);
		    System.out.println(delta1);
		    System.exit(0);
	
		    w.set(1, w.get(1).sub(out.get(0).T().dot(delta2).mult(lr)));
		    w.set(0, w.get(0).sub(out.get(0).T().dot(delta1).mult(lr)));
	
		    b.set(1, b.get(1).sub(delta2.sum()).mult(lr));
		    b.set(0, b.get(0).sub(delta1.sum()).mult(lr));
		    
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void train(Matrix x, Matrix y, int epochs)  {
		for (int i = 0; i < epochs; i++) {
			forwardPropagate(x);
			backPropagate(x, y);
		}
	}
	
	public Matrix predict(Matrix x) {
		forwardPropagate(x);
		return out.get(out.size() - 1);
	}
}
