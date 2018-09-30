package layers;

import static utilities.Activations.sigmoid;
import static utilities.Activations.softmax;
import static utilities.Activations.sigmoidDerivative;

import utilities.Activations;
import utilities.Activations.Activation;
import utilities.Array;
import utilities.Initializers.Init;
import utilities.Matrix;

public class Dense {
    
    private int inputs;
    private int outputs;
    private Activation activation;
    
    public Matrix w;
    public Array b;
    
    private int layerNum;

    public Dense(int outputs, String activation) {
        this.inputs = 0;
        this.outputs = outputs;
        this.activation = Activations.getEnum(activation);
    }
    
    public Dense(int outputs, String activation, int inputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.activation = Activations.getEnum(activation);
        
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
        Matrix output = null;
        try {
            Matrix linear = input.dot(w).add(b);
            
            switch (activation) {
                case sigmoid:
                    output = sigmoid(linear);
                    break;
                case linear:
                    output = linear;
                    break;
                case softmax:
                    output = softmax(linear);
                    break;
            }
            
        } catch (Exception e) {
            throw new RuntimeException("Error feeding layer " + layerNum + " (Dense)");
        }
        return output;
    }
    
    public Matrix backfeed(Matrix delta, Matrix layerInput, Dense nextLayer) {
        Matrix result = null;
        try {
            switch (activation) {
                case sigmoid:
                    result = delta.dot(nextLayer.w.T()).mult(sigmoidDerivative(layerInput.dot(w).add(b)));
                    break;
                case linear:
                    result = delta.dot(nextLayer.w.T()).mult(layerInput.dot(w).add(b));
                    break;
                case softmax:
                    throw new RuntimeException("Softmax only supported on last layer");
            }
        
        } catch (Exception e) {
            throw new RuntimeException("Error during backpropagation (layer " + layerNum + ":Dense)");
        }
        return result;
    }
}
