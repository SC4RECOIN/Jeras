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
	
	public Matrix mult(int scalar) {
		float[][] result = new float[rows][columns];
		
		for (int i = 0; i < rows; i++){
	        for (int j = 0; j < columns; j++){
	        	result[i][j] = values[i][j] * scalar;
	        }
	    }
		return new Matrix(result);
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
		String output = "[ ";
        for (int i = 0; i < rows; i++) {
        	for (int j = 0; j < columns; j++) {
        		output += String.format("%.3f ", this.values[i][j]); 
        	}
        	if (i == rows-1) { output += "]"; }
        	output += "\n  ";
        }
        return output;
    }
}
