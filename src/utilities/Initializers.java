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

	// also known as the Xavier uniform initializer
//	TODO
//	public glorotUniform() {
//		Nd4j.randn(order(), shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
//	}
}
