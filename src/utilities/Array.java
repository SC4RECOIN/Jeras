package utilities;


import static utilities.Initializers.init;
import utilities.Initializers.Init;

public class Array {
	
	public  float[] values;
	public int length;
	
	public Array(int length, Init initializer) {
		this.length = length;
		values = init(length, initializer);
	}
	
	public Array(int length) {
		this(length, Init.RANDOM);
	}
	
	public Array(float[] arr) {
		this.length = arr.length;
		values = arr;
	}
	
	public Array sub(Array arr2) throws Exception {
		if (this.length != arr2.length) {
			throw new Exception("Cannot subtract Arrays of different lengths");
		}
		float[] result = new float[this.length];
		for (int i = 0; i < this.length; i++) {
			result[i] = this.values[i] - arr2.values[i];
		}
		return new Array(result);
	}
	
	public Array mult(float scalar) {
		float[] result = new float[this.length];
		for (int i = 0; i < this.length; i++) {
			result[i] = this.values[i] * scalar;
		}
		return new Array(result);
	}
}
