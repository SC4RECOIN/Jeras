package utilities;

import utilities.Initializers.Init;

public class Matrix {
	
	public final float[][] values;
	public final int rows;
	public final int columns;
	
	public Matrix(float[][] matrix) {
		this.values = matrix;
		this.rows = matrix.length;
		this.columns = matrix[0].length;
	}
	
	public Matrix(int rows, int columns, Init initializer) {
		this.values = Initializers.init(rows, columns, initializer);
		this.rows = rows;
		this.columns = columns;
	}
	
	public Matrix(int rows, int columns) {
		this.values = Initializers.init(rows, columns, Init.NORMAL);
		this.rows = rows;
		this.columns = columns;
	}
	
	public Matrix transpose() {
		float[][] result = new float[this.values[0].length][this.values.length];
		for (int i = 0; i < rows; i++) {
        	for (int j = 0; j < columns; j++) {
        		result[j][i] = values[i][j];
        	}
        }
        return new Matrix(result);
	}
	
	public Matrix T() {
		return transpose();
	}
	
	public Array sum() {
		float[] result = new float[this.columns];
		for (int i = 0; i < columns; i++) {
			result[i] = 0;
	        for (int j = 0; j < rows; j++){
	        	result[i] += this.values[j][i];
	        }
	    }
		return new Array(result);
	}
	
	public Matrix sub(Matrix matrix2) throws Exception {
	    if (this.rows != matrix2.rows || this.columns != matrix2.columns){
	        throw new Exception("Cannot subtract matrices of different sizes");
	    }
	    
	    float[][] result = new float[rows][columns];
	    for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = this.values[i][j] - matrix2.values[i][j];
	        }
	    }
	    return new Matrix(result);
	}
	
	public Matrix add(Matrix matrix2) throws Exception {
	    if (this.rows != matrix2.rows || this.columns != matrix2.columns) {
	        throw new Exception("Cannot add matrices of different sizes");
	    }
	    
	    float[][] result = new float[rows][columns];
	    for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = this.values[i][j] + matrix2.values[i][j];
	        }
	    }
	    return new Matrix(result);
	}
	
	public Matrix add(float[] arr) throws Exception {
	    if (this.columns != arr.length) {
	        throw new Exception("Cannot add array of length " + arr.length + " with matrix with " + this.columns + " columns");
	    }
	    
	    float[][] result = new float[rows][columns];
	    for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = this.values[i][j] + arr[j];
	        }
	    }
	    return new Matrix(result);
	}
	
	public Matrix add(Array arr) throws Exception {
		return this.add(arr.values);
	}
	
	public Matrix mult(float scalar) {
		float[][] result = new float[rows][columns];
		
		for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = values[i][j] * scalar;
	        }
	    }
		return new Matrix(result);
	}
	
	public Matrix mult(int scalar) {
		return mult((float) scalar);
	}
	
	public Matrix mult(Matrix matrix2) throws Exception {
		if (this.rows != matrix2.rows || this.columns != matrix2.columns) {
	        throw new Exception("Cannot bulk multiply matrices of different sizes");
	    }
	    
	    float[][] result = new float[rows][columns];
	    for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = this.values[i][j] * matrix2.values[i][j];
	        }
	    }
	    return new Matrix(result);
	}
	
	public Matrix div(Array arr) throws Exception {
		if (this.columns != arr.length) {
	        throw new Exception(String.format("Cannot divide Array(%d) with Matrix(%d,%d)", arr.length, this.rows, this.columns));
	    }
	    
	    float[][] result = new float[rows][columns];
	    for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = this.values[i][j] / arr.values[j];
	        }
	    }
	    return new Matrix(result);
	}
	
	public Matrix matmul(Matrix matrix2) throws Exception {
        if(this.columns != matrix2.rows) {
        	throw new Exception("Incompatible matrices for multiplication");
        }
        
        float[][] result = new float[this.rows][matrix2.columns];
        for(int i = 0; i < this.rows; i++) {         
            for(int j = 0; j < matrix2.columns; j++) {    
                for(int k = 0; k < this.columns; k++) { 
                	result[i][j] += this.values[i][k] * matrix2.values[k][j];
                }
            }
        }
        return new Matrix(result);
    }
	
	public Matrix dot(Matrix matrix2) {
		float [][] result = new float[rows][matrix2.columns];
	
		for (int i = 0; i < this.rows; i++) { 
		    for (int j = 0; j < matrix2.columns; j++) { 
		        for (int k = 0; k < this.columns; k++) { 
		            result[i][j] += this.values[i][k] * matrix2.values[k][j];
		        }
		    }
		}
		return new Matrix(result);
	}
	
	@Override
    public String toString() {
		String output = "[";
        for (int i = 0; i < rows; i++) {
        	output += "[";
        	for (int j = 0; j < columns; j++) {
        		if (this.values[i][j] > 0) { output += " "; }
        		output += String.format("%.3f ", this.values[i][j]); 
        	}
        	if (i == rows-1) { output += "]"; }
        	output += "]\n ";
        }
        return output;
    }
}
