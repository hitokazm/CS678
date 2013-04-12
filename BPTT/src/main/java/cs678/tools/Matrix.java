// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
package cs678.tools;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

public class Matrix implements Serializable {
	private static final long serialVersionUID = 1L;
	
	// Data
	ArrayList< double[] > m_data;

	
	// Meta-data
	ArrayList< String > m_attr_name;
	ArrayList< TreeMap<String, Integer> > m_str_to_enum;
	ArrayList< TreeMap<Integer, String> > m_enum_to_str;

	double mean;
	double sd;
	
	static double MISSING = Double.MAX_VALUE; // representation of missing values in the dataset

	// Creates a 0x0 matrix. You should call loadARFF or setSize next.
	public Matrix() {}

	// Creates a rowsxcols empty matrix. 
	public Matrix(int rows, int cols){
		this.setSize(rows, cols);
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int col = 0; col < cols; col++){
			String name = "attr" + (col+1);
			m_attr_name.add(name);
			TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
			TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
			m_str_to_enum.add(ste);
			m_enum_to_str.add(ets);
		}
	}

	public void setOutputClass(String name, int maxValue){
		m_attr_name.add(name);
		TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
		TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
		for(int v = 0; v <= maxValue; v++){
			ste.put(String.valueOf(v), new Integer(v));
			ets.put(new Integer(v), String.valueOf(v));
		}
		m_str_to_enum.add(ste);
		m_enum_to_str.add(ets);
	}
	
	// Copies the specified portion of that matrix into this matrix
	public Matrix(Matrix that, int rowStart, int colStart, int rowCount, int colCount) {
		m_data = new ArrayList< double[] >();
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[colCount];
			for(int i = 0; i < colCount; i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int i = 0; i < colCount; i++) {
			m_attr_name.add(that.attrName(colStart + i));
			m_str_to_enum.add(that.m_str_to_enum.get(colStart + i));
			m_enum_to_str.add(that.m_enum_to_str.get(colStart + i));
		}
	}
	
	// Adds a copy of the specified portion of that matrix to this matrix
	public void add(Matrix that, int rowStart, int colStart, int rowCount) throws Exception {
		if(colStart + cols() > that.cols())
			throw new Exception("out of range");
		for(int i = 0; i < cols(); i++) {
			if(that.valueCount(colStart + i) != valueCount(i))
				throw new Exception("incompatible relations");
		}
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[cols()];
			for(int i = 0; i < cols(); i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
	}

	// Resizes this matrix (and sets all attributes to be continuous)
	public void setSize(int rows, int cols) {
		m_data = new ArrayList<double[]>();
		for(int j = 0; j < rows; j++) {
			double[] row = new double[cols];
			m_data.add(row);
		}
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int i = 0; i < cols; i++) {
			m_attr_name.add("");
			m_str_to_enum.add(new TreeMap<String, Integer>());
			m_enum_to_str.add(new TreeMap<Integer, String>());
		}
	}

	// Loads from an ARFF file
	public void loadArff(String filename) throws Exception, FileNotFoundException {
		m_data = new ArrayList<double[]>();
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		boolean READDATA = false;
		Scanner s = new Scanner(new File(filename));
		while (s.hasNext()) {
			String line = s.nextLine().trim();
			if (line.length() > 0 && line.charAt(0) != '%') {
				if (!READDATA) {
					
					Scanner t = new Scanner(line);
					String firstToken = t.next().toUpperCase();
					
					String datasetName = "";
					
					if (firstToken.equals("@RELATION")) {
						datasetName = t.nextLine();
					}
					
					if (firstToken.equals("@ATTRIBUTE")) {
						TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
						m_str_to_enum.add(ste);
						TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
						m_enum_to_str.add(ets);

						Scanner u = new Scanner(line);
						if (line.indexOf("'") != -1) u.useDelimiter("'");
						u.next();
						String attributeName = u.next();
						if (line.indexOf("'") != -1) attributeName = "'" + attributeName + "'";
						m_attr_name.add(attributeName);

						int vals = 0;
						String type = u.next().trim().toUpperCase();
						if (type.equals("REAL") || type.equals("CONTINUOUS") || type.equals("INTEGER")) {
						}
						else {
							try {
								String values = line.substring(line.indexOf("{")+1,line.indexOf("}"));
								Scanner v = new Scanner(values);
								v.useDelimiter(",");
								while (v.hasNext()) {
									String value = v.next().trim();
									if(value.length() > 0)
									{
										ste.put(value, new Integer(vals));
										ets.put(new Integer(vals), value);
										vals++;
									}
								}
							}
							catch (Exception e) {
								throw new Exception("Error parsing line: " + line + "\n" + e.toString());
							}
						}
					}
					if (firstToken.equals("@DATA")) {
						READDATA = true;
					}
				}
				else {
					double[] newrow = new double[cols()];
					int curPos = 0;

					try {
						Scanner t = new Scanner(line);
						t.useDelimiter(",");
						while (t.hasNext()) {
							String textValue = t.next().trim();
							//System.out.println(textValue);

							if (textValue.length() > 0) {
								double doubleValue;
								int vals = m_enum_to_str.get(curPos).size();
								
								//Missing instances appear in the dataset as a double defined as MISSING
								if (textValue.equals("?")) {
									doubleValue = MISSING;
								}
								// Continuous values appear in the instance vector as they are
								else if (vals == 0) {
									doubleValue = Double.parseDouble(textValue);
								}
								// Discrete values appear as an index to the "name" 
								// of that value in the "attributeValue" structure
								else {
									doubleValue = m_str_to_enum.get(curPos).get(textValue);
									if (doubleValue == -1) {
										throw new Exception("Error parsing the value '" + textValue + "' on line: " + line);
									}
								}
								
								newrow[curPos] = doubleValue;
								curPos++;
							}
						}
					}
					catch(Exception e) {
						throw new Exception("Error parsing line: " + line + "\n" + e.toString());
					}
					m_data.add(newrow);
				}
			}
		}
	}

	// Returns the number of rows in the matrix
	public int rows() { return m_data.size(); }

	// Returns the number of columns (or attributes) in the matrix
	public int cols() { return m_attr_name.size(); }

	// Returns the specified row
	public double[] row(int r) { return m_data.get(r); }

	// Returns the element at the specified row and column
	public double get(int r, int c) { return m_data.get(r)[c]; }

	// Sets the value at the specified row and column
	public void set(int r, int c, double v) { row(r)[c] = v; }

	// Returns the name of the specified attribute
	public String attrName(int col) { return m_attr_name.get(col); }

	// Set the name of the specified attribute
	public void setAttrName(int col, String name) { m_attr_name.set(col, name); }

	// Returns the name of the specified value
	public String attrValue(int attr, int val) { return m_enum_to_str.get(attr).get(val); }

	// Returns the number of values associated with the specified attribute (or column)
	// 0=continuous, 2=binary, 3=trinary, etc.
	public int valueCount(int col) { return m_enum_to_str.get(col).size(); }

	// Shuffles the row order
	public void shuffle(Random rand) {
		for(int n = rows(); n > 0; n--) {
			int i = rand.nextInt(n);
			double[] tmp = row(n - 1);
			m_data.set(n - 1, row(i));
			m_data.set(i, tmp);
		}
	}

	// Returns the mean of the specified column
	public double columnMean(int col) {
		double sum = 0;
		int count = 0;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				sum += v;
				count++;
			}
		}
		return sum / count;
	}

	public double columnSD(int col){
		double mean = columnMean(col);
		return this.columnSD(col, mean);
	}
	
	public double columnSD(int col, double mean){
		double sqSum = 0.0;
		for(int row = 0; row < rows(); row++){
			sqSum += Math.pow(get(row,col)-mean, 2.0);
		}
		return Math.sqrt(sqSum/(double) rows());
	}
	
	// Returns the min value in the specified column
	public double columnMin(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v < m)
					m = v;
			}
		}
		return m;
	}

	// Returns the max value in the specified column
	public double columnMax(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v > m)
					m = v;
			}
		}
		return m;
	}

	// Returns the most common value in the specified column
	public double mostCommonValue(int col) {
		TreeMap<Double, Integer> tm = new TreeMap<Double, Integer>();
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				Integer count = tm.get(v);
				if(count == null)
					tm.put(v, new Integer(1));
				else
					tm.put(v, new Integer(count.intValue() + 1));
			}
		}
		int maxCount = 0;
		double val = MISSING;
		Iterator< Entry<Double, Integer> > it = tm.entrySet().iterator();
		while(it.hasNext())
		{
			Entry<Double, Integer> e = it.next();
			if(e.getValue() > maxCount)
			{
				maxCount = e.getValue();
				val = e.getKey();
			}
		}
		return val;
	}

	public void normalize() {
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				double min = columnMin(i);
				double max = columnMax(i);
				//System.out.println("min and max: " + min + "  " + max);
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						if(max == min)
							set(j, i, 0.5);
						else
							set(j, i, (v - min) / (max - min));
				}
			}
		}
	}

	
	
	public void normalize(boolean standardNormal) {
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				double min = columnMin(i);
				double max = columnMax(i);
				
				double mean = columnMean(i);
				double sd = columnSD(i, mean);
				
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						if(max == min){
							if(standardNormal)
								set(j, i, 0);
							else
								set(j, i, 0.5);
						}
						else{
							if(standardNormal){
								set(j, i, (v - mean) / sd);
							}
							else{
								set(j, i, (v - min) / (max - min));
							}
						}
				}
			}
		}
	}

	public void normalize(double[] min, double[] max){
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						if(max[i] == min[i])
							set(j, i, 0.5);
						else
							set(j, i, (v - min[i]) / (max[i] - min[i]));
				}
			}
		}		
	}
	
	public void normalize(double[] means, double[] sds, double[] max, double[] min, boolean standardNormal){
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				double sum = 0;
				for(int j = 0; j < rows(); j++){
					sum += this.get(j, i);
				}
				double mean = sum / (double) rows();
				sum = 0;
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						if(max[i] == min[i]){
							if(standardNormal){
								set(j, i, 0.0);
							}
							else{
								set(j, i, 0.5);
							}
						}
						else{
							if(standardNormal){
								set(j, i, (v - means[i]) / sds[i]);
							}
							else{
								set(j, i, (v - max[i])/ (max[i] - min[i]));
							}
						}
				}
			}
		}		
	}

	public void print() {
		System.out.println("@RELATION Untitled");
		for(int i = 0; i < m_attr_name.size(); i++) {
			System.out.print("@ATTRIBUTE " + m_attr_name.get(i));
			int vals = valueCount(i);
			if(vals == 0)
				System.out.println(" CONTINUOUS");
			else
			{
				System.out.print(" {");
				for(int j = 0; j < vals; j++) {
					if(j > 0)
						System.out.print(", ");
					System.out.print(m_enum_to_str.get(i).get(j));
				}
				System.out.println("}");
			}
		}
		System.out.println("@DATA");
		for(int i = 0; i < rows(); i++) {
			double[] r = row(i);
			for(int j = 0; j < r.length; j++) {
				if(j > 0)
					System.out.print(",");
				if(valueCount(j) == 0)
					System.out.print(r[j]);
				else
					System.out.print(m_enum_to_str.get(j).get((int)r[j]));
			}
			System.out.println("");
		}
	}
	

	public String export() {
		Formatter formatter = new Formatter(new StringBuilder());
		
		formatter.format("@RELATION Untitled\n");
		for(int i = 0; i < m_attr_name.size(); i++) {
			formatter.format("@ATTRIBUTE " + m_attr_name.get(i));
			int vals = valueCount(i);
			if(vals == 0)
				formatter.format(" CONTINUOUS\n");
			else
			{
				formatter.format(" {");
				for(int j = 0; j < vals; j++) {
					if(j > 0)
						formatter.format(", ");
					formatter.format(m_enum_to_str.get(i).get(j));
				}
				formatter.format("}\n");
			}
		}
		formatter.format("@DATA\n");
		for(int i = 0; i < rows(); i++) {
			double[] r = row(i);
			for(int j = 0; j < r.length; j++) {
				if(j > 0)
					formatter.format(",");
				if(valueCount(j) == 0)
					formatter.format("%.1f", r[j]);
				else
					formatter.format(m_enum_to_str.get(j).get((int)r[j]));
			}
			formatter.format("\n");
		}
		return formatter.toString();
	}
	
	
	// I added this mothod for decision tree project (Feb. 18, 2012)
	public void removeRow(int row){
		this.m_data.remove(row);
	}
	
	public void addRow(double[] row){
		double[] toBeAdded = new double[row.length];
		for(int i = 0; i < row.length; i++){
			toBeAdded[i] = row[i];
		}
		this.m_data.add(toBeAdded);
	}
	
	// Copies the data framework of the source matrix into this matrix 
	// this doesn't copy rows 
	// (I added this constructor for decision tree project, Feb. 18, 2012)
	public Matrix(Matrix that, int colStart, int colCount) {
		m_data = new ArrayList< double[] >();
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int i = 0; i < colCount; i++) {
			m_attr_name.add(that.attrName(colStart + i));
			m_str_to_enum.add(that.m_str_to_enum.get(colStart + i));
			m_enum_to_str.add(that.m_enum_to_str.get(colStart + i));
		}
	}
	
	// get the actual value count in the column
	// I added this for decision tree
	public int getActualValueCount(int col) {
		
		Set<Double> types = new HashSet<Double>();
		
		for(double[] data : this.m_data){
			if(!types.contains(data[col])){
				types.add(data[col]);
			}
		}
	
		return types.size();
	}
	
	// add an unknown value at attribute column col
	// I added this for decision tree
	public void addAttributeUnknownValue(int col){

//		System.out.println(m_enum_to_str.toString());
//		System.out.println(m_str_to_enum.toString());
//
//
//		System.out.println("Value Count before: " + valueCount(col));
		
		m_enum_to_str.get(col).put(new Integer(valueCount(col)), "unknown");
		m_str_to_enum.get(col).put("unknown", new Integer(valueCount(col)-1));

//		System.out.println(m_enum_to_str.toString());
//		System.out.println(m_str_to_enum.toString());
//		
//		System.out.println("Value Count after: " + valueCount(col));
		
		for(double[] row : m_data){
			if(row[col] == Double.MAX_VALUE)
				row[col] = (double) valueCount(col)-1;
		}
		
	}
	
	public boolean containUnknownValue(int col){
		
		for(double[] row : m_data){
			if(row[col] == Double.MAX_VALUE)
				return true;
		}
		return false;
		
	}
	
}
