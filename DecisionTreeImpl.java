import java.util.ArrayList;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl{

	private DecTreeNode root;
	//ordered list of attributes
	private List<String> mTrainAttributes; 
	//
	private ArrayList<ArrayList<Double>> mTrainDataSet;
	//Min number of instances per leaf.
	private int minLeafNumber = 10;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() 
	{
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(ArrayList<ArrayList<Double>> trainDataSet, ArrayList<String> trainAttributeNames, int minLeafNumber) 
	{
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = trainDataSet;
		this.minLeafNumber = minLeafNumber;
		this.root = buildTree(this.mTrainDataSet);
	}

	/**
	 * Recursively builds a decision tree starting at the root 
	 * @param dataSet - training set the tree is built from
	 * @return node - the newly created node of the decision tree
	 */
	private DecTreeNode buildTree(ArrayList<ArrayList<Double>> dataSet)
	{
		// Not enough examples. Return leaf node.
		if (0 == dataSet.size() || dataSet.size() <= this.minLeafNumber)
		{
			int class0Count = 0;
			int class1Count = 0;

			for (int example = 0; example < dataSet.size(); ++example)
			{
				if (0 == classify(dataSet.get(example)))
				{
					++class0Count;
				}
				else
				{
					++class1Count;
				}
			}
			if (class0Count > class1Count)
			{
				return new DecTreeNode(0, "leaf", 0.0);
			}
			else
			{
				return new DecTreeNode(1, "leaf", 0.0);
			}
		}
		// Check if all examples have same classification. If so, return leaf node.
		int prevClassNum = -1;
		int currClassNum = -1;
		boolean sameClassification = true;;
		for (int example = 0; example < dataSet.size(); ++example)
		{
			currClassNum = classify(dataSet.get(example));
			if (-1 == prevClassNum)
			{
				prevClassNum = currClassNum;
			}
			else
			{
				if (currClassNum != prevClassNum)
				{
					sameClassification = false;
					break;
				}
			}
		}
		// All examples have same classification.
		if (sameClassification)
		{
			return new DecTreeNode(currClassNum, "leaf", 0.0);
		}
		// End of base cases. Start creating threshold node.
		DecTreeNode node = null;
		double bestSplitPoint = 0.0;
		ArrayList<Double> splitPoints = null;
		ArrayList<DataBinder> dataBinderList = null;
		ArrayList<Double> candidateThresholds = new ArrayList<Double>();
		// Find best splitPoint for each attribute
		for (int attribute = 0; attribute < this.mTrainAttributes.size(); ++attribute)
		{
			dataBinderList = generateDataBinderList(attribute, dataSet);
			splitPoints = generateSplitPoints(dataBinderList, dataSet);
			if (!splitPoints.isEmpty())
			{
				bestSplitPoint = calculateBestSplitPoint(splitPoints, dataBinderList);
			}
			else
			{
				bestSplitPoint = 0.0;
			}
			candidateThresholds.add(bestSplitPoint);
		}
		// Determine next threshold value. Save attribute index for array of threshold vals
		int attributeIndex = 0;
		double bestInfoGain = 0.0;
		double currInfoGain = 0.0;
		for (int attribute = 0; attribute < candidateThresholds.size(); ++attribute)
		{
			if (candidateThresholds.get(attribute) != 0.0)
			{
				dataBinderList = generateDataBinderList(attribute, dataSet);
				currInfoGain = calculateInfoGain(candidateThresholds.get(attribute), dataBinderList);
				// Update variables if needed
				if (currInfoGain >= bestInfoGain)
				{
					bestInfoGain = currInfoGain;
					attributeIndex = attribute;
				}
			}
		}
		ArrayList<ArrayList<Double>> leftTreeDataSet = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> rightTreeDataSet = new ArrayList<ArrayList<Double>>();
		// Generate new dataSets to partition examples for child nodes
		for (int example = 0; example < dataSet.size(); ++example)
		{
			if (dataSet.get(example).get(attributeIndex) <= candidateThresholds.get(attributeIndex))
			{
				leftTreeDataSet.add(dataSet.get(example));
			}
			else
			{
				rightTreeDataSet.add(dataSet.get(example));
			}
		}
		// Construct node to return
		node = new DecTreeNode(0, this.mTrainAttributes.get(attributeIndex), candidateThresholds.get(attributeIndex));
		// Create left child node
		DecTreeNode leftChild = buildTree(leftTreeDataSet);
		// Create right child node
		DecTreeNode rightChild = buildTree(rightTreeDataSet);
		node.left = leftChild;
		node.right = rightChild;

		return node;
	}

	/**
	 * Computes and prints the information gain for each attribute threshold at the root
	 * @param dataSet - data set of the tree
	 * @param trainAttributeNames - names of the training attributes
	 * @param minLeafNumber - minimum number of examples remaining in data set until threshold node becomes a leaf
	 */
	public void rootInfoGain(ArrayList<ArrayList<Double>> dataSet, ArrayList<String> trainAttributeNames, int minLeafNumber)
	{
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = dataSet;
		this.minLeafNumber = minLeafNumber;

		double bestSplitPoint = 0.0;
		ArrayList<Double> splitPoints = null;
		ArrayList<DataBinder> dataBinderList = null;
		ArrayList<Double> candidateThresholds = new ArrayList<Double>();
		// Find best splitPoint for each attribute
		for (int attribute = 0; attribute < trainAttributeNames.size(); ++attribute)
		{
			dataBinderList = generateDataBinderList(attribute, dataSet);
			splitPoints = generateSplitPoints(dataBinderList, dataSet);
			if (!splitPoints.isEmpty())
			{
				bestSplitPoint = calculateBestSplitPoint(splitPoints, dataBinderList);
			}
			else
			{
				bestSplitPoint = 0.0;
			}
			candidateThresholds.add(bestSplitPoint);
		}
		double currInfoGain = 0.0;
		for (int attribute = 0; attribute < candidateThresholds.size(); ++attribute)
		{
			dataBinderList = generateDataBinderList(attribute, dataSet);
			currInfoGain = calculateInfoGain(candidateThresholds.get(attribute), dataBinderList);
			System.out.println(this.mTrainAttributes.get(attribute) + " " + String.format("%.6f", currInfoGain));
		}

		//modify this example print statement to work with your code to output attribute names and info gain. Note the %.6f output format.
		//for(int i = 0; i < bestSplitPointList.size(); i++)
		//{
		//	System.out.println(this.mTrainAttributes.get(i) + " " + String.format("%.6f", bestSplitPointList.get(i).get(0)));
		//}	
	}

	/**
	 * Calculates the information gain of a threshold for a given attribute on the data set
	 * @param threshold
	 * @param dataBinderList - list of DataBinder objects for a given attribute
	 * @return infoGain - the information gain
	 */
	public double calculateInfoGain(double threshold, ArrayList<DataBinder> dataBinderList)
	{
		double infoGain = 0.0;
		double part1Entropy = 0.0;
		double part2Entropy = 0.0;
		double entropy = 0.0;
		double part1CondEntropy = 0.0;
		double part2CondEntropy = 0.0;
		double condEntropy = 0.0; // conditional entropy
		double numExamples = (double) dataBinderList.size();
		double greaterThan = 0.0; // count of examples w/ attribute vals > thresh
		double lessThanOrEqual = 0.0; // count of examples w/ attribute vals <= thresh
		double greaterThanClass0 = 0.0; // count of ex. > thresh that have class 0
		double greaterThanClass1 = 0.0; // count of ex. > thresh that have class 1
		double lessThanClass0 = 0.0; // count of ex. <= thresh that have class 0
		double lessThanClass1 = 0.0; // count of ex. <= thresh that have class 1
		for (int example = 0; example < dataBinderList.size(); ++example)
		{
			// Example has attribute val > threshold
			if (dataBinderList.get(example).getArgItem() > threshold)
			{
				++greaterThan;
				if (0 == classify(dataBinderList.get(example).getData()))
				{
					++greaterThanClass0;
				}
				else
				{
					++greaterThanClass1;
				}
			}
			// Example has attribute val <= threshold
			else
			{
				++lessThanOrEqual;
				if (0 == classify(dataBinderList.get(example).getData()))
				{
					++lessThanClass0;
				}
				else
				{
					++lessThanClass1;
				}
			}
		}
		// Calculate conditional entropy
		part1CondEntropy = (lessThanOrEqual / numExamples) * (((-1.0) * lessThanClass0 / lessThanOrEqual) * (Math.log10(lessThanClass0 / lessThanOrEqual) / Math.log10(2.0)) + ((-1.0) * lessThanClass1 / lessThanOrEqual) * (Math.log10(lessThanClass1 / lessThanOrEqual) / Math.log10(2.0)));
		part2CondEntropy = (greaterThan / numExamples) * (((-1.0) * greaterThanClass0 / greaterThan) * (Math.log10(greaterThanClass0 / greaterThan) / Math.log10(2.0)) + ((-1.0) * greaterThanClass1 / greaterThan) * (Math.log10(greaterThanClass1 / greaterThan) / Math.log10(2.0)));
		if (Double.isNaN(part1CondEntropy))
		{
			part1CondEntropy = 0.0;
		}
		if (Double.isNaN(part2CondEntropy))
		{
			part2CondEntropy = 0.0;
		}
		condEntropy = part1CondEntropy + part2CondEntropy;
		// Calculate attribute entropy
		part1Entropy = ((-1.0) * (lessThanClass0 + greaterThanClass0) / numExamples) * (Math.log10((lessThanClass0 + greaterThanClass0) / numExamples) / Math.log10(2.0));
		part2Entropy = ((-1.0) * (lessThanClass1 + greaterThanClass1) / numExamples) * (Math.log10((lessThanClass1 + greaterThanClass1) / numExamples) / Math.log10(2.0));
		if (Double.isNaN(part1Entropy))
		{
			part1Entropy = 0.0;
		}
		if (Double.isNaN(part2Entropy))
		{
			part2Entropy = 0.0;
		}
		entropy = part1Entropy + part2Entropy;
		infoGain = entropy - condEntropy;
		return infoGain;
	}
	/**
	 * Calculates the best split point for a given attribute and corresponding list
	 * @param splitPoints
	 * @param dataBinderList - list of DataBinder objects for a given attribute
	 * @return bestSplitPoint - the split point with the lowest entropy
	 */
	public double calculateBestSplitPoint(ArrayList<Double> splitPoints, ArrayList<DataBinder> dataBinderList)
	{
		double bestSplitPoint = 0.0;
		double part1CondEntropy = 0.0;
		double part2CondEntropy = 0.0;
		double condEntropy = 0.0; // conditional entropy
		double temp = 1000.0;
		double numExamples = (double) dataBinderList.size();
		double greaterThan = 0.0; // count of examples w/ attribute vals > split
		double lessThanOrEqual = 0.0; // count of examples w/ attribute vals <= split
		double greaterThanClass0 = 0.0; // count of ex. > split that have class 0
		double greaterThanClass1 = 0.0; // count of ex. > split that have class 1
		double lessThanClass0 = 0.0; // count of ex. <= split that have class 0
		double lessThanClass1 = 0.0; // count of ex. <= split that have class 1
		for (int splitPoint = 0; splitPoint < splitPoints.size(); ++splitPoint)
		{
			for (int example = 0; example < dataBinderList.size(); ++example)
			{
				// Example has attribute val > split
				if (dataBinderList.get(example).getArgItem() > splitPoints.get(splitPoint))
				{
					++greaterThan;
					if (0 == classify(dataBinderList.get(example).getData()))
					{
						++greaterThanClass0;
					}
					else
					{
						++greaterThanClass1;
					}
				}
				// Example has attribute val <= split
				else
				{
					++lessThanOrEqual;
					if (0 == classify(dataBinderList.get(example).getData()))
					{
						++lessThanClass0;
					}
					else
					{
						++lessThanClass1;
					}
				}
			}
			part1CondEntropy = (lessThanOrEqual / numExamples) * (((-1.0) * lessThanClass0 / lessThanOrEqual) * (Math.log10(lessThanClass0 / lessThanOrEqual) / Math.log10(2.0)) + ((-1.0) * lessThanClass1 / lessThanOrEqual) * (Math.log10(lessThanClass1 / lessThanOrEqual) / Math.log10(2.0)));
			part2CondEntropy = (greaterThan / numExamples) * (((-1.0) * greaterThanClass0 / greaterThan) * (Math.log10(greaterThanClass0 / greaterThan) / Math.log10(2.0)) + ((-1.0) * greaterThanClass1 / greaterThan) * (Math.log10(greaterThanClass1 / greaterThan) / Math.log10(2.0)));
			
			if (Double.isNaN(part1CondEntropy))
			{
				part1CondEntropy = 0.0;
			}
			if (Double.isNaN(part2CondEntropy))
			{
				part2CondEntropy = 0.0;
			}
			condEntropy = part1CondEntropy + part2CondEntropy;
			if (condEntropy <= temp)
			{
				temp = condEntropy;
				bestSplitPoint = splitPoints.get(splitPoint);
			}
			// Reset values to calculate conditional entropy of next split point
			greaterThan = 0.0;
			lessThanOrEqual = 0.0;
			greaterThanClass0 = 0.0;
			greaterThanClass1 = 0.0;
			lessThanClass0 = 0.0;
			lessThanClass1 = 0.0;
		}
		return bestSplitPoint;
	}
	
	/**
	 * Generates a list of split points for a given attribute
	 * @param dataBinderList - a list of DataBinder objects for a given attribute
	 * @param dataSet
	 * @return splitPoints - list of split points
	 */
	public ArrayList<Double> generateSplitPoints(ArrayList<DataBinder> dataBinderList, ArrayList<ArrayList<Double>> dataSet)
	{
		int prevClassification = -1;
		int currClassification = -1;
		double average = 0.0;
		ArrayList<Double> splitPoints = new ArrayList<Double>();
		for (int example = 0; example < dataSet.size(); ++example)
		{
			currClassification = classify(dataBinderList.get(example).getData());
			// First iteration
			if (-1 == prevClassification)
			{
				;
			}
			else
			{
				// Add average value if classes of two consecutive examples differ
				if (prevClassification != currClassification)
				{
					average = (dataBinderList.get(example).getArgItem() + dataBinderList.get(example - 1).getArgItem()) / 2.0;
					splitPoints.add(average);
				}
			}
			prevClassification = currClassification;
		}
		return splitPoints;
	}
	/**
	 * Generates a sorted list of DataBinder objects 
	 * @param attribute - attribute for the DataBinder objects
	 * @param dataSet - data set for the DataBinder objects
	 * @return dataBinderList - sorted list of DataBinder objects
	 */
	public ArrayList<DataBinder> generateDataBinderList(int attribute, ArrayList<ArrayList<Double>> dataSet)
	{
		ArrayList<DataBinder> dataBinderList = new ArrayList<DataBinder>();
		for (int example = 0; example < dataSet.size(); ++example)
		{
			dataBinderList.add(new DataBinder(attribute, dataSet.get(example)));
		}
		Collections.sort(dataBinderList, new DataBinder());
		return dataBinderList;
	}
	/**
	 * Finds the true value of an example's (instance) class label
	 * @param instance - a list of double values each corresponding to an example's (instance) attributes
	 * @return true value of an example's class label
	 */
	public static int classify(List<Double> instance) 
	{
		double doubleResult = instance.get(instance.size() - 1);
		int result = (int) doubleResult;
		return result;
	}
	/**
	 * Predicts the class label of an example (instance) using the constructed decision tree
	 * @param instance - a list of double values each corresponding to an examples's (instance) attributes
	 * @return prediction of example's class label
	 */
	public int predictClass(List<Double> instance)
	{
		double threshold = 0.0;
		DecTreeNode currNode = root;
		int attribute = 0;
		// Iterate until leaf node is reached
		while (!currNode.isLeaf())
		{
			String attributeString = currNode.attribute;
			// Trim off "A" and cast to int
			attributeString = attributeString.substring(1);
			attribute = Integer.parseInt(attributeString);
			attribute = attribute - 1;
			threshold = currNode.threshold;
			// Traverse right side of tree
			if (instance.get(attribute) > threshold)
			{
				currNode = currNode.right;
			}
			// Traverse left side of tree
			else
			{
				currNode = currNode.left;
			}
		}
		return currNode.classLabel;
	}

	/**
	 * Print the decision tree in the specified format
	 */
	public void print() 
	{
		printTreeNode("", this.root);
	}

	/**
	 * Recursively prints the tree structure, left subtree first, then right subtree.
	 */
	public void printTreeNode(String prefixStr, DecTreeNode node) 
	{
		String printStr = prefixStr + node.attribute;

		System.out.print(printStr + " <= " + String.format("%.6f", node.threshold));
		if(node.left.isLeaf())
		{
			System.out.println(": " + String.valueOf(node.left.classLabel));
		}
		else
		{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%.6f", node.threshold));
		if(node.right.isLeaf())
		{
			System.out.println(": " + String.valueOf(node.right.classLabel));
		}
		else
		{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}
	/**
	 * Print accuracy based on inputs
	 * @param numEqual - number of equal occurrences of class value
	 * @param numTotal - total number of examples in data set
	 * @return accuracy - the accuracy calculation
	 */
	public double printAccuracy(int numEqual, int numTotal)
	{
		double accuracy = (double) numEqual / (double) numTotal;
		System.out.println(accuracy);
		return accuracy;
	}


	/**
	 * Private class to facilitate instance sorting by argument position since java doesn't like passing variables to comparators through
	 * nested variable scopes.
	 * */
	private class DataBinder implements Comparator<DataBinder>
	{

		public ArrayList<Double> mData;
		public int i;
		public DataBinder(){}
		public DataBinder(int i, ArrayList<Double> mData)
		{
			this.mData = mData;
			this.i = i;
		}
		public double getArgItem()
		{
			return mData.get(i);
		}
		public ArrayList<Double> getData()
		{
			return mData;
		}
		@Override
		public int compare(DataBinder a, DataBinder b)
		{
			// Attribute value of a is bigger than b
			if (a.getArgItem() > b.getArgItem())
			{
				return 1;
			}
			// Attribute value of b is bigger than a
			else if (a.getArgItem() < b.getArgItem())
			{
				return -1;
			}
			// Attribute values of a and b are the same
			else
			{
				// a should come before b if it has less than or equal class value
				if (DecisionTreeImpl.classify(a.getData()) < DecisionTreeImpl.classify(b.getData()))
				{
					return -1;
				}
				else if (DecisionTreeImpl.classify(a.getData()) == DecisionTreeImpl.classify(b.getData()))
				{
					return 0;
				}
				// a should come after b if has the greater class value
				else
				{
					return 1;
				}
			}
		}
	}

}
