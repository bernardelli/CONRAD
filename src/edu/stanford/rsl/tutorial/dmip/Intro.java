package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector.VectorNormType;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.IJ;
import ij.ImageJ;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;


/**
 * Introduction to the CONRAD Framework
 * Exercise 1 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */

public class Intro {
	
	
	public static void gridIntro(){
				
		//Define the image size
		int imageSizeX = 256;
		int imageSizeY = 256;
	
		//Define an image
		//Hint: Import the package edu.stanford.rsl.conrad.data.numeric.Grid2D
		//DONE
		Grid2D image = new Grid2D(imageSizeX, imageSizeY);
	
		//Draw a circle
		int radius = 50;
		//Set all pixels within the circle to 100
		int insideVal = 100;
	

		for (int i = 0; i < imageSizeX; i++){
			for (int j = 0; j < imageSizeY; j++){
				if((Math.pow(i-imageSizeX/2, 2) + Math.pow(j-imageSizeY/2, 2)) < Math.pow(radius,2))
				image.setAtIndex(i, j, insideVal);
			}
		}
		
		//Show ImageJ GUI
		ImageJ ij = new ImageJ();
		//Display image				
		image.show("look at me");
		
		//Copy an image
		Grid2D copy = new Grid2D(image);
		copy.show("Copy of circle");
		
		
		//Load an image from file
		String filename = "C:/Users/rafael/workspace/Recostruction/CONRAD/src/edu/stanford/rsl/tutorial/dmip/mr12.dcm";
		Grid2D mrImage = ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
		mrImage.show();
		
		//convolution
		Convolver conv = new Convolver();
		
		//define the kernel. Try simple averaging 3x3 filter
		int kw = 3;
		int kh = 3;
		float[] kernel = new float[kw*kh];
		for(int i = 0; i < kernel.length; i++)
		{	
			kernel[i] = 1.f / (kw*kh);
		}
		
		conv.convolve(ImageUtil.wrapGrid2D(mrImage), kernel, kw, kh);
		mrImage.show("convoluted");
		
		//write an image to disk, check the supported output formats
		String outFilename ="C:/Users/rafael/workspace/Recostruction/CONRAD/src/edu/stanford/rsl/tutorial/dmip/mr12_out.gif";
		
		IJ.save(ImageUtil.wrapGrid(mrImage, null), outFilename); //quando eu tento tif fica tudo branco, mas gif funciona :)
	}
	
	
	public static void signalIntro()
	{
		//How can I plot a sine function sin(2*PI*x)?
		double stepSize = 0.01;
		int plotLength = 500;
		
		double[] y = new double[plotLength];
		
		for(int i = 0; i < y.length; i++)
		{
			y[i] = Math.sin(2.0 * Math.PI * stepSize * i);
			
		}
		
		VisualizationUtil.createPlot(y).show();
		double[] x = new double [plotLength];
		for(int i = 0; i < x.length; i++)
		{
			x[i] = (double) i * stepSize;
		}
		
		VisualizationUtil.createPlot(x, y, "sin(x)", "x", "y").show();		
		
	}
	
	public static void basicIntro()
	{
		//Display text
		System.out.println("Creating a vector: v1 = [1.0; 2.0; 3.0]");
		
		//create column vector
		SimpleVector v1 = new SimpleVector(1.f, 2.f, 3.f);
		System.out.println("v1 = " + v1.toString());
		
		//create a randomly initialized vector
		SimpleVector vRand = new SimpleVector(3);
		vRand.randomize(0, 2);
		System.out.println("vRand = " + vRand.toString());
		
		//create matrix M 3x3  1 2 3; 4 5 6; 7 8 9
		SimpleMatrix M = new SimpleMatrix(3,3);
		M.setColValue(0, new SimpleVector(1,4,7));
		M.setColValue(1, new SimpleVector(2,5,8));
		M.setColValue(2, new SimpleVector(3,6,9));
		
		System.out.println("M = " + M.toString());
		
		//determinant of M
		System.out.println("Determinant of matrix m: " + M.determinant() );
		
		//transpose M
		SimpleMatrix Mt = new SimpleMatrix(3,3);
		Mt = M.transposed();
		
		System.out.println("M^T: " + Mt.toString() );
		//copy matrix
		SimpleMatrix Mcopy = new SimpleMatrix(M);
	
		Mcopy.transpose();
				
		//transpose M inplace
		System.out.println("M after clone attribuition: " + M.toString() );
		System.out.println("Mcopy: " + Mcopy.toString() );
		//get size
		int numRows = M.getRows();
		int numCols = M.getCols();
		
		//access elements of M
		System.out.println("M: ");
		for(int i = 0 ; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				double element = M.getElement(i, j); 
				System.out.print(element + " ");
			}
			System.out.println();
		}
		
		//Create 3x3 Matrix of 1's
		SimpleMatrix Mones = new SimpleMatrix(3,3);
		Mones.ones();
		//Create a 3x3 Matrix of 0's
		SimpleMatrix Mzeros = new SimpleMatrix(3,3);
		Mones.zeros();
		//Create a 3x3 Identity matrix
		SimpleMatrix Midentity = new SimpleMatrix(3,3);
		Midentity.identity();
		
		//Matrix multiplication
		SimpleMatrix ResMat = SimpleOperators.multiplyMatrixProd(Mt, M);
		System.out.println("M^T * M = " + ResMat.toString());
		

		//Matrix vector multiplication
		SimpleVector resVec = SimpleOperators.multiply(ResMat, v1);
		System.out.println("M * v1 = " + resVec.toString());
		
		
		//Extract the last column vector from matrix M
		SimpleVector colVector = M.getCol(2);
		System.out.println("M(0)(2) = " + colVector.toString());
		
		//Extract the 1x2 subvector from the last column of matrix M
		int [] selectRows = {0, 1, 2};
		int [] selectCols = {1, 2};
		SimpleMatrix subVector = M.getSubMatrix(selectRows, selectCols);
		System.out.println("[m(0)(2); m(1)(2)] = " + subVector.toString());
		
		//Matrix elementwise multiplication
		SimpleMatrix MsquaredElem = SimpleOperators.multiplyElementWise(M, M);
		System.out.println("M squared Elements: " + MsquaredElem.toString());
		
		//round vectors
		SimpleVector vRandCopy = new SimpleVector(vRand);
		System.out.println("vRand         = " + vRandCopy.toString());
		
		vRandCopy.floor();
		System.out.println("vRand.floor() = " + vRandCopy.toString());
		
		vRand.ceil();
		System.out.println("vRand.ceil()  = " + vRand.toString());
		
		//min, max, mean
		//double minV1 = v1.min();
		//double maxV1 = v1.max();
		//System.out.println("Min(v1) = " + minV1 + " Max(v1) = " + maxV1);
		
		//for matrices: iterate over row or column vectors
		SimpleVector maxVec = new SimpleVector(M.getCols());
		for(int i = 0; i < M.getCols(); i++)
		{
			maxVec.setElementValue(i, M.getCol(i).max());
		}
		double maxM = maxVec.max();
		System.out.println("Max(M) = " + maxM);
		
		
		
		//Norms
		//matrix L1
		double matrixNormL1 = M.norm(MatrixNormType.MAT_NORM_L1);
		//vector L2
		double vecNormL2 = colVector.normL2();
		System.out.println("||M||_L1 = " + matrixNormL1);
		System.out.println("||colVec||_2 = " + vecNormL2);
		
		//get normalized vector
		SimpleVector normVector = colVector.normalizedL2();
		//normalize vector in-place
		colVector.normalizeL2();
		System.out.println("Normalized colVector: " + colVector.toString());
		
		
		//SVD
		SimpleMatrix A = new SimpleMatrix(3,3);
		A.setRowValue(0, new SimpleVector(11, 10,  14));
		A.setRowValue(1, new SimpleVector(12, 11, -13));
		A.setRowValue(2, new SimpleVector(14, 13, -66));
				
		System.out.println("A = " + A.toString());
		
		//print singular matrix
		DecompositionSVD svd = new DecompositionSVD(A);
		System.out.println("S= " + svd.getS().toString());
		System.out.println("U= " + svd.getU().toString());
		System.out.println("V= " + svd.getV().toString());
		//get condition number
		System.out.println("Condition number of A: " + svd.cond() );
		
		//Re-compute A = U * S * V^T
		SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		SimpleMatrix A2 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		System.out.println("U * S * V^T: " + A2.toString());
		
	}

	public static void main(String arg[])
	{
		basicIntro();
		gridIntro();
		signalIntro();
	}
}
