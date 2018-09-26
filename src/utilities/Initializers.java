package utilities;

import java.util.Random;

public class Initializers {
	
	public enum Init {
		RANDOM
	  , NORMAL
	}
	
	public static float[][] init(int rows, int cols, Init initializer) {
		switch (initializer) {
			case RANDOM:
				return random(rows, cols, false);
			case NORMAL:
				return random(rows, cols, true);
			default:
				return random(rows, cols, true);
		}
	}
	
	private static float[][] random(int inputs, int outputs, boolean normal) {
		Random rand = new Random();
		float[][] weights = new float[inputs][outputs];
		for (int i = 0; i < inputs; i++) {
			for (int j = 0; j < outputs; j++) {
				if (normal) {
					weights[i][j] = (float) rand.nextGaussian();
				} else {
					weights[i][j] = rand.nextFloat();
				}
			}
		}
		return weights;
	}
	
	public static float[] init(int length, Init initializer) {
		switch (initializer) {
			case RANDOM:
				return random(length, false);
			case NORMAL:
				return random(length, true);
			default:
				return random(length, true);
		}
	}
	
	private static float[] random(int length, boolean normal) {
		Random rand = new Random();
		float[] array = new float[length];
		for (int i = 0; i < length; i++) {
			if (normal) { array[i] = (float) rand.nextGaussian(); }
			else { array[i] = rand.nextFloat(); }
		}
		return array;
	}

	// also known as the Xavier uniform initializer
//	TODO
//	public glorotUniform() {
//		Nd4j.randn(order(), shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
//	}
}
