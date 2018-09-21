package utilities;

public class Functions {

	public static float[][] dot(float[][] matrix1, float[][] matrix2) {
		float [][] result = new float[matrix1.length][matrix2[0].length];
	
		for (int i = 0; i < matrix1.length; i++) { 
		    for (int j = 0; j < matrix2[0].length; j++) { 
		        for (int k = 0; k < matrix1[0].length; k++) { 
		            result[i][j] += matrix1[i][k] * matrix2[k][j];
		        }
		    }
		}
		return result;
	}
	
	public static float[][] transpose(float[][] matrix) {
		float[][] result = new float[matrix[0].length][matrix.length];
        
		for (int i = 0; i < matrix.length; i++) {
        	for (int j = 0; j < matrix[0].length; j++) {
        		result[j][i] = matrix[i][j];
        	}
        }
        return result;
	}
	
	public static float[][] subtract(float[][] matrix1, float[][] matrix2) throws Exception {
	    if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length){
	        throw new Exception("Cannot subtract matrices of different sizes");
	    }
	    float[][] result = new float[matrix1.length][matrix1[0].length];

	    for (int i = 0; i < matrix1.length; i++){
	        for (int j = 0; j < matrix1[0].length; j++){
	        	result[i][j] = matrix1[i][j] - matrix2[i][j];
	        }
	    }
	    return result;
	}
	
	public static float[][] add(float[][] matrix1, float[][] matrix2) throws Exception {
	    if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length){
	        throw new Exception("Cannot add matrices of different sizes");
	    }
	    
	    float[][] result = new float[matrix1.length][matrix1[0].length];
	    for (int i = 0; i < matrix1.length; i++){
	        for (int j = 0; j < matrix1[0].length; j++){
	        	result[i][j] = matrix1[i][j] + matrix2[i][j];
	        }
	    }
	    return result;
	}
	
	public static float[][] multiply(float[][] matrix, int scalar) {
		float[][] result = new float[matrix.length][matrix[0].length];
		
		for (int i = 0; i < matrix.length; i++){
	        for (int j = 0; j < matrix[0].length; j++){
	        	result[i][j] = matrix[i][j] * scalar;
	        }
	    }
		return result;
	}
	
	public static float[][] multiply(float[][] matrix1, float[][] matrix2) throws Exception {
	    if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length){
	        throw new Exception("Cannot bulk multiply matrices of different sizes");
	    }
	    
	    float[][] result = new float[matrix1.length][matrix1[0].length];
	    for (int i = 0; i < matrix1.length; i++){
	        for (int j = 0; j < matrix1[0].length; j++){
	        	result[i][j] = matrix1[i][j] * matrix2[i][j];
	        }
	    }
	    return result;
	}
	
	public static float[][] matrixMultiply(float[][] matrix1, float[][] matrix2) throws Exception {
        int m1RowLength = matrix1.length;
        int m2RowLength = matrix2.length;    
        int m1ColLength = matrix1[0].length; 
        int m2ColLength = matrix2[0].length;
        
        if(m1ColLength != m2RowLength) {
        	throw new Exception("Incompatible matrices for multiplication");
        }
        
        float[][] result = new float[m1RowLength][m2ColLength];
        for(int i = 0; i < m1RowLength; i++) {         // rows from m1
            for(int j = 0; j < m2ColLength; j++) {     // columns from m2
                for(int k = 0; k < m1ColLength; k++) { // columns from m1
                	result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return result;
    }

	
	public static float[][] copyOf(float[][] original){
		float[][] copied = new float[original.length][original[0].length];
	    for(int i = 0; i < original.length; i++){
	        for(int j = 0; j < original[i].length; j++){
	        	float x = original[i][j];
	            copied[i][j] = x;
	        }
	    }
	    return copied;
	}
}
