package cs678.bptt;

import cs678.tools.MLSystemManager;
import cs678.tools.Matrix;

public class Main 
{
    public static void main( String[] args ) throws Exception
    {
		MLSystemManager ml = new MLSystemManager();
		Matrix matrix = new Matrix();
		matrix.loadArff("data/activity.arff");
		matrix.normalize();
		Matrix small = new Matrix(matrix, 0, 3, matrix.rows(), 3);
		small.print();
    }
}
